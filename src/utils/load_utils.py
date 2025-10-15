import cv2
import numpy as np
import trimesh
import os
import torch

from src.constants import IMAGE_SIZE, SMPLX_FACES_PATH
from src.utils.structs import HandParams, ObjectParams
from src.utils.renderer_out import MySoftSilhouetteRenderer


def load_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # resize image
    h, w = image.shape[:2]
    r = min(IMAGE_SIZE / w, IMAGE_SIZE / h)
    w = int(r * w)
    h = int(r * h)
    image = cv2.resize(image, (w, h))

    return image


def load_hand_params(hand_inference_file: str, hand_detection_file: str = None, imgsize: np.ndarray = None) -> HandParams:
    hand_mesh = trimesh.load(hand_inference_file)
    left_right = "left" if "left" in hand_inference_file else "right"

    ## For now, directly use the posed hand vertices instead of MANO
    # # save the centroid offset
    centroid_offset = hand_mesh.centroid
    # # center the mesh
    # hand_mesh.apply_translation(-centroid_offset)

    ## For now, skip the part of parameterization and mask loading
    # smplx_params = {
    #     'betas': human_npz['hps_betas'],
    #     'body_pose': human_npz['hps_body_pose'],
    #     'global_orient': human_npz['hps_global_orient'],
    #     'right_hand_pose': human_npz['hps_right_hand_pose'],
    #     'left_hand_pose': human_npz['hps_left_hand_pose'],
    #     'jaw_pose': human_npz['hps_jaw_pose'],
    #     'leye_pose': human_npz['hps_leye_pose'],
    #     'reye_pose': human_npz['hps_reye_pose'],
    #     'expression': human_npz['hps_expression']
    # }

    # detection = np.load(human_detection_file)
    # mask = np.array(detection['mask']).astype(float)
    # # resize to image size
    # mask = cv2.resize(mask, (imgsize[1], imgsize[0]))

    hand_params = HandParams(
        vertices = hand_mesh.vertices,
        faces = hand_mesh.faces,
        centroid_offset = centroid_offset.copy(),
        left_right=left_right,
        # bbox = human_npz['bbox'][0],
        # mask = mask,
        # smplx_params = smplx_params
    )

    hand_params.to_cuda()
    return hand_params
    

def load_object_params(object_mesh_file: str, object_detection_file: str = None, imgsize: np.ndarray = None,
                       trans_mat=None, load_obj_mask=False, cam_intrinsic=None) -> ObjectParams:
    obj_mesh = trimesh.load(object_mesh_file)
    gt_vertices = torch.from_numpy(obj_mesh.vertices).float()

    # render object mask
    if load_obj_mask:
        faces = torch.from_numpy(obj_mesh.faces).float().cuda()
        vertices = torch.from_numpy(obj_mesh.vertices).float().cuda()
        renderer = MySoftSilhouetteRenderer(img_shape=imgsize, faces=faces, cam_intrinsic=cam_intrinsic)
        mask = renderer.render(vertices).cpu().numpy()
    else:
        mask = None

    # initialize object pose, according to how the HOI dataset defines it
    if trans_mat is None:
        # rotate object 90 degrees around x-axis (mostly upright in objaverse)
        obj_mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0]))
        # center the mesh
        obj_mesh.apply_translation(-obj_mesh.centroid)
    else:
        rot_mat = torch.eye(4, dtype=torch.float)
        rot_mat[:3, :3] = trans_mat[:3, :3]
        trans_vec = trans_mat[:3, 3]
        obj_mesh.apply_translation(trans_vec)
        obj_mesh.apply_transform(rot_mat)
    # # load object mask and resize to image size
    # detection = np.load(object_detection_file)
    # mask = np.array(detection['mask']).astype(float)
    # mask = cv2.resize(mask, (imgsize[1], imgsize[0]))

    object_params = ObjectParams(
        vertices = obj_mesh.vertices,
        faces = obj_mesh.faces,
        mask = mask,
        scale = obj_mesh.extents.max()
    )

    object_params.to_cuda()
    return object_params, gt_vertices

