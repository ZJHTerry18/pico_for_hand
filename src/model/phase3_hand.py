import torch
import torch.nn as nn
import trimesh
from tqdm.auto import tqdm
import numpy as np
from smplx import MANO

from src.constants import MANO_MODEL_PATH
from src.utils.contact_mapping import calculate_hand_points, interpret_contact_points, select_pose_parameters
from src.utils.renderer_out import MySoftSilhouetteRenderer
from src.utils.structs import HandParams, ObjectParams
from src.utils.sdf.sdf.sdf_loss import SDFLoss


class Phase_3_Optimizer(nn.Module):
    def __init__(self,
        mano_params,
        hand_points,
        object_points,
        contact_mapping,
        body_pose_indices_to_opt, 
        hand_params,
        object_params,
        render_size,
        cam_intrinsic,
        sparse_dense_mapping,
    ):
        super(Phase_3_Optimizer, self).__init__()

        self.register_buffer('mano_betas', torch.tensor(mano_params['betas']).float().cuda())
        self.register_buffer('mano_global_orient', torch.tensor(mano_params['global_orient']).float().cuda())
        self.register_buffer('mano_transl', torch.tensor(mano_params['transl']).float().cuda())
        self.register_buffer('mano_pose', torch.tensor(mano_params['hand_pose']).float().cuda())
        self.hand_lr = hand_params.left_right
        self.sparse_dense_mapping = sparse_dense_mapping

        self.body_pose_indices_to_opt = body_pose_indices_to_opt

        self.mano_pose_opt = nn.Parameter(
            # torch.tensor(mano_params['hand_pose'][:, self.body_pose_indices_to_opt]).float().cuda(),
            torch.tensor(mano_params['hand_pose']).float().cuda(), 
            requires_grad=True)
        self.register_buffer('mano_pose_init', torch.tensor(mano_params['hand_pose']).float().cuda())

        self.register_buffer('hand_points', hand_points)
        self.register_buffer('object_points', object_points)
        self.contact_transfer_map = contact_mapping

        mano_model = MANO(MANO_MODEL_PATH, is_rhand=mano_params['is_rhand'], flat_hand_mean=True, use_pca=False)
        self.mano_model = mano_model.cuda()

        self.register_buffer('hum_vertices', hand_params.vertices)
        self.register_buffer('hum_faces', hand_params.faces)
        self.register_buffer('hum_centroid_offset', hand_params.centroid_offset)
        # self.register_buffer('hum_bbox', hand_params.bbox)
        self.register_buffer('hum_mask', hand_params.mask.float() if hand_params.mask is not None else None)
        self.register_buffer('obj_vertices', object_params.vertices)

        # # center the MANO mesh
        # newverts = self.get_hand_verts(remove_offset=False)
        # temp_mesh = trimesh.Trimesh(vertices=newverts.detach().cpu().numpy(), faces=hand_params.faces.detach().cpu().numpy())
        # self.mano_offset = torch.tensor(temp_mesh.centroid).float().cuda()

        self.renderer = MySoftSilhouetteRenderer(render_size, hand_params.faces, cam_intrinsic)

        # SDF collision loss setup
        self.sdf_loss = SDFLoss(hand_params.faces, robustifier=1.0)


    def get_mano_pose(self):
        # Recombine the optimized and constant parameters
        # full_pose = self.mano_pose_init.clone()
        # full_pose[:, self.body_pose_indices_to_opt] = self.smplx_body_pose_opt
        full_pose = self.mano_pose_opt
        return full_pose

    def get_hand_verts(self, remove_offset=True):
        output = self.mano_model(
            betas=self.mano_betas.unsqueeze(0),
            hand_pose=self.get_mano_pose().unsqueeze(0),
            global_orient=self.mano_global_orient.unsqueeze(0),
            transl=self.mano_transl.unsqueeze(0),
        )
        verts_hand = output.vertices[0]
        if remove_offset:
            verts_hand = verts_hand - self.hum_centroid_offset
        return verts_hand


    def calculate_contact_loss(self, upd_hand_vertices):
        new_hand_points = calculate_hand_points(
            upd_hand_vertices, self.contact_transfer_map, self.hand_lr, self.sparse_dense_mapping)
        loss = torch.nn.functional.mse_loss(new_hand_points, self.object_points)
        return {"loss_contact": loss}

    def calculate_collision_loss(self, upd_hand_vertices):
        loss = self.sdf_loss(upd_hand_vertices, self.obj_vertices)
        return {"loss_collision_p3": loss}
    
    def calculate_pose_reg_loss(self):
        # loss = torch.nn.functional.mse_loss(self.mano_pose_opt, self.mano_pose_init[:, self.body_pose_indices_to_opt])
        loss = torch.nn.functional.mse_loss(self.mano_pose_opt, self.mano_pose_init)
        # if self.left_hand_opt:
        #     loss += torch.nn.functional.mse_loss(self.smplx_left_hand_pose, self.smplx_left_hand_pose_init)
        # if self.right_hand_opt:
        #     loss += torch.nn.functional.mse_loss(self.smplx_right_hand_pose, self.smplx_right_hand_pose_init)
        return {"loss_pose_reg": loss}
    

    def calculate_silhouette_loss_iou(self, upd_hand_vertices):
        current_mask = self.renderer.render(
            upd_hand_vertices + self.hum_centroid_offset
        )
        intersection = torch.sum(current_mask * self.hum_mask)
        union = torch.sum((current_mask + self.hum_mask).clamp(0, 1))
        loss = 1 - intersection / union
        return {"loss_silhouette_hand": loss}

    def calculate_silhouette_loss_l2(self, upd_hand_vertices):
        loss = torch.tensor(0.0).cuda()
        pred_mask = self.renderer.render(upd_hand_vertices + self.hum_centroid_offset)
        loss = torch.nn.functional.mse_loss(pred_mask, self.hum_mask)
        return {"loss_silhouette_hand": loss}


    def forward(self, loss_weights: dict):
        upd_hand_vertices = self.get_hand_verts()

        loss_dict = {}
        if loss_weights["lw_contact"] > 0:
            loss_dict.update(self.calculate_contact_loss(upd_hand_vertices))
        if loss_weights["lw_collision_p3"] > 0:
            loss_dict.update(self.calculate_collision_loss(upd_hand_vertices))
        if loss_weights["lw_pose_reg"] > 0:
            loss_dict.update(self.calculate_pose_reg_loss())
        if loss_weights["lw_silhouette_hand"] > 0 and self.hum_mask is not None:
            loss_dict.update(self.calculate_silhouette_loss_iou(upd_hand_vertices))

        return loss_dict
   

def optimize_phase3_hand(
    hand_params: HandParams, object_params: ObjectParams, contact_mapping: dict, render_size: list, cam_intrinsic: torch.Tensor,
    sparse_dense_mapping: dict=None, **kwargs,
):
    object_mesh = trimesh.Trimesh(vertices=object_params.vertices.detach().cpu().numpy(), faces=object_params.faces.detach().cpu().numpy())
    hand_mesh = trimesh.Trimesh(vertices=hand_params.vertices.detach().cpu().numpy(), faces=hand_params.faces.detach().cpu().numpy())

    hand_points, object_points = interpret_contact_points(
        contact_mapping, hand_mesh.vertices, object_mesh.vertices, object_mesh, hand_params.left_right,
        sparse_dense_map=sparse_dense_mapping,
    )

    # # TODO: select which hand pose parameters to optimize - the ones in contact. Not doing this now.
    # body_pose_indices_to_opt, left_hand_opt, right_hand_opt = select_pose_parameters(contact_mapping)
    # if len(body_pose_indices_to_opt) == 0 and not left_hand_opt and not right_hand_opt:
    #     print("--> No contacting limbs to optimize! Skipping phase 3.")
    #     hand_parameters = {}
    #     hand_parameters["vertices"] = hand_params.vertices.detach()
    #     return hand_parameters, {}
    # print("Optimizing body pose indices:", body_pose_indices_to_opt)
    # print("Optimizing left hand:", left_hand_opt)
    # print("Optimizing right hand:", right_hand_opt)

    loss_weights = kwargs["loss_weights"]
    nr_phase_3_steps = kwargs["nr_phase_3_steps"]
    lr_phase_3 = kwargs.get("lr_phase_3", 0.01)

    model = Phase_3_Optimizer(
        hand_params.mano_params,
        hand_points,
        object_points,
        contact_mapping,
        None,
        hand_params,
        object_params,
        render_size,
        cam_intrinsic,
        sparse_dense_mapping,
    )
    model.cuda()

    # optimizer params
    opt_params = [
        {'params': [model.mano_pose_opt], 'lr': lr_phase_3},
    ]
    # if left_hand_opt:
    #     opt_params.append({'params': [model.smplx_left_hand_pose], 'lr': lr_phase_3})
    # if right_hand_opt:
    #     opt_params.append({'params': [model.smplx_right_hand_pose], 'lr': lr_phase_3})

    # optimizer with separate learning rates for each parameter
    optimizer = torch.optim.Adam(opt_params)

    loop = tqdm(total=nr_phase_3_steps)
    for i in range(nr_phase_3_steps):
        optimizer.zero_grad()
        loss_dict = model(loss_weights)
        loss_dict_weighted = {
            k: loss_dict[k] * loss_weights[k.replace("loss", "lw")] for k in loss_dict
        }
        loss = sum(loss_dict_weighted.values())
        loss.backward() # remove retain_graph=True for memory reasons
        optimizer.step()
        # loop.set_description(f'loss: {loss.item():.3g}')
        # loop.update()

        if i % 50 == 0:
            loss_str = " | ".join([f"{k}: {loss_dict_weighted[k].item():.3g}" for k in loss_dict_weighted])
            print(loss_str)
            # print('hand', torch.mean(model.mano_pose_opt.grad), torch.mean(model.mano_pose_opt))


    hand_parameters = {}
    updated_hand_vertices = model.get_hand_verts()
    hand_parameters["vertices"] = updated_hand_vertices.detach()
    hand_parameters["mano_pose_init"] = model.mano_pose_init
    hand_parameters["global_orient"] = model.mano_global_orient
    hand_parameters["transl"] = model.mano_transl
    hand_parameters["mano_pose_opt"] = model.mano_pose_opt.detach()
    hand_parameters["mano_betas"] = model.mano_betas
    hand_parameters["loss"] = {k: v.item() for k, v in loss_dict_weighted.items()} 

    return hand_parameters
