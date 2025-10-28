import torch
import torch.nn as nn
import trimesh
from tqdm.auto import tqdm
import numpy as np
from scipy import ndimage

from src.utils.contact_mapping import calculate_object_points, interpret_contact_points, apply_transformation
from src.utils.geometry import rot6d_to_matrix, rotation_matrix_to_angle_axis
from src.utils.renderer_out import MySoftSilhouetteRenderer
from src.utils.structs import HandParams, ObjectParams
from src.utils.sdf.sdf.sdf_loss import SDFLoss


class Phase_2_Optimizer(nn.Module):
    def __init__(self,
        rotation_init,
        translation_init,
        scaling_init,
        human_points,
        object_points,
        hand_params,
        object_params,
        contact_mapping,
        render_size,
        cam_intrinsic,
    ):
        super(Phase_2_Optimizer, self).__init__()

        self.rotation = nn.Parameter(rotation_init.clone().float(), requires_grad=True)
        self.translation = nn.Parameter(translation_init.clone().float(), requires_grad=True)
        self.scaling = nn.Parameter(scaling_init.clone().float(), requires_grad=True)
        # self.register_buffer('scaling', torch.tensor([1.0], device='cuda'))

        self.register_buffer('human_points', human_points)
        self.register_buffer('object_points', object_points)
        self.register_buffer('hum_vertices', hand_params.vertices)
        self.register_buffer('hum_centroid_offset', hand_params.centroid_offset)
        # self.register_buffer('hum_bbox', hand_params.bbox)
        self.register_buffer('obj_vertices', object_params.vertices)
        self.register_buffer('obj_faces', object_params.faces)
        self.register_buffer('obj_mask', object_params.mask.float() if object_params.mask is not None else None)
        self.register_buffer('obj_init_scaling', torch.tensor([1.0], device='cuda'))
        self.contact_transfer_map = contact_mapping
        self.render_size = render_size

        self.renderer = MySoftSilhouetteRenderer(render_size, object_params.faces, cam_intrinsic)

        # dist_mat = ndimage.distance_transform_edt(1 - self.obj_mask.cpu().numpy())
        # self.register_buffer('dist_mat', torch.tensor(dist_mat, device='cuda'))

        # SDF collision loss setup
        self.sdf_loss = SDFLoss(hand_params.faces, robustifier=1.0)


    def calculate_contact_loss(self, upd_obj_vertices):
        new_object_points = calculate_object_points(upd_obj_vertices, self.contact_transfer_map, self.obj_faces)
        loss = torch.nn.functional.mse_loss(self.human_points, new_object_points)
        return {"loss_contact": loss}
    
    def calculate_scale_loss(self):
        # current scale is the biggest dimension of the object (from the 3 axes)
        loss = torch.nn.functional.mse_loss(self.obj_init_scaling, self.scaling)
        return {"loss_scale": loss}
    
    def calculate_collision_loss(self, upd_obj_vertices):
        loss = self.sdf_loss(self.hum_vertices, upd_obj_vertices)
        return {"loss_collision_p2": loss}
    
    def calculate_centroid(self, mask):
        coords = torch.nonzero(mask, as_tuple=False)
        if coords.nelement() == 0:
            return torch.tensor([mask.shape[0] / 2, mask.shape[1] / 2], device=mask.device)
        weights = mask[coords[:, 0], coords[:, 1]]
        # Ensure no in-place operations modify 'coords' or 'weights'
        centroid = torch.sum(coords * weights.unsqueeze(1), dim=0) / torch.sum(weights)
        return centroid
    
    def distance_penalty(self, current_mask):
        # distance penalty term (distance between the centroids of the white regions in the masks)
        centroid_target = self.calculate_centroid(self.obj_mask)
        centroid_current = self.calculate_centroid(current_mask)
        # Calculate the distance and normalize
        distance = torch.norm(centroid_target - centroid_current)
        distance = distance / current_mask.shape[0]  # Use assignment instead of /= which is in-place
        return distance

    def calculate_silhouette_loss_iou(self, upd_obj_vertices, distance_penalty=0):
        current_mask = self.renderer.render(
            upd_obj_vertices + self.hum_centroid_offset    # we move hand to centroid when loading, so need to translate it back
        )
        intersection = torch.sum(current_mask * self.obj_mask)
        union = torch.sum((current_mask + self.obj_mask).clamp(0, 1))
        loss = 1 - intersection / union

        if distance_penalty > 0:
            distance = self.distance_penalty(current_mask)
            loss += distance_penalty * distance

        return {"loss_silhouette": loss}

    def forward(self, loss_weights: dict):
        upd_obj_vertices = apply_transformation(self.obj_vertices, self.rotation, self.translation, self.scaling)
        
        loss_dict = {}
        if loss_weights["lw_contact"] > 0:
            loss_dict.update(self.calculate_contact_loss(upd_obj_vertices))
        if loss_weights["lw_silhouette"] > 0 and self.obj_mask is not None: # for samples without object masks, we cannot calculate mask loss
            loss_dict.update(self.calculate_silhouette_loss_iou(upd_obj_vertices, distance_penalty=loss_weights["lw_silhouette_distance"]))
        if loss_weights["lw_scale"] > 0:
            loss_dict.update(self.calculate_scale_loss())
        if loss_weights["lw_collision_p2"] > 0:
            loss_dict.update(self.calculate_collision_loss(upd_obj_vertices))

        return loss_dict
   

def optimize_phase2_image(
    hand_params: HandParams, object_params: ObjectParams, contact_mapping: dict, render_size: list, cam_intrinsic: torch.Tensor, 
    sparse_dense_mapping: dict=None, **kwargs,
):
    object_mesh = trimesh.Trimesh(vertices=object_params.vertices.detach().cpu().numpy(), faces=object_params.faces.detach().cpu().numpy())
    hand_mesh = trimesh.Trimesh(vertices=hand_params.vertices.detach().cpu().numpy(), faces=hand_params.faces.detach().cpu().numpy())

    human_points, object_points = interpret_contact_points(
        contact_mapping, hand_mesh.vertices, object_mesh.vertices, object_mesh, hand_params.left_right,
        sparse_dense_map=sparse_dense_mapping,
    )
    
    rotation_init = torch.tensor([1.01, 0.01, 0.01, 1.01, 0.01, 0.01], requires_grad=True).cuda()
    translation_init = torch.tensor([0.0, 0.0, 0.0], requires_grad=True).cuda()
    scaling_init = torch.tensor([1.0], requires_grad=True).cuda()

    loss_weights = kwargs["loss_weights"]
    nr_phase_2_steps = kwargs["nr_phase_2_steps"]
    lr_rotation_phase_2 = kwargs.get("lr_rotation_phase_2", 0.04)
    lr_translation_phase_2 = kwargs.get("lr_translation_phase_2", 0.03)
    lr_scaling_phase_2 = kwargs.get("lr_scaling_phase_2", 0.02)

    model = Phase_2_Optimizer(
        rotation_init,
        translation_init,
        scaling_init,
        human_points,
        object_points,
        hand_params,
        object_params,
        contact_mapping,
        render_size,
        cam_intrinsic,
    )
    model.cuda()

    # optimizer with separate learning rates for each parameter
    optimizer = torch.optim.Adam([
        {'params': [model.rotation], 'lr': lr_rotation_phase_2},
        {'params': [model.translation], 'lr': lr_translation_phase_2},
        {'params': [model.scaling], 'lr': lr_scaling_phase_2},
    ])

    loop = tqdm(total=nr_phase_2_steps)
    for i in range(nr_phase_2_steps):
        optimizer.zero_grad()
        loss_dict = model(loss_weights)
        loss_dict_weighted = {
            k: loss_dict[k] * loss_weights[k.replace("loss", "lw")] for k in loss_dict
        }
        loss = sum(loss_dict_weighted.values())
        loss.backward() # remove retain_graph=True for memory reasons
        optimizer.step()
        loop.set_description(f'loss: {loss.item():.3g}')
        loop.update()

        if i % 50 == 0:
            loss_str = " | ".join([f"{k}: {loss_dict_weighted[k].item():.3g}" for k in loss_dict_weighted])
            print(loss_str)
            # print("gradients: ", model.rotation.grad, model.translation.grad, model.scaling.grad)


    object_parameters = {}
    object_parameters["rotation"] = rotation_matrix_to_angle_axis(rot6d_to_matrix(model.rotation).detach())
    object_parameters["translation"] = model.translation.unsqueeze(0).detach()
    object_parameters["scaling"] = model.scaling.detach()
    transformed_obj_vertices = apply_transformation(object_params.vertices, model.rotation, model.translation, model.scaling)
    object_parameters["vertices"] = transformed_obj_vertices.detach()

    return object_parameters
