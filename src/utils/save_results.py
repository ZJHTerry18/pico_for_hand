import os
import numpy as np
import trimesh
import json
import torch
from glob import glob
from PIL import Image
import copy

from src.constants import COLOR_HUMAN_BLUE, COLOR_OBJECT_RED
from src.utils.structs import HandParams, ObjectParams
from src.utils.renderer_out import render_overlaid_view, render_side_views
from src.utils.geometry import axis_angle_to_matrix

from scipy.spatial.transform import Rotation

def save_phase_results(
    img_filename: str,
    output_folder: str,
    sample: dict,
    object_phase_params: dict,
    phase: int,
    do_eval: bool=False,
):
    # common folder, already exists if not first image
    os.makedirs(output_folder, exist_ok=True)
    
    # save the combined mesh
    hand_params = copy.deepcopy(sample["hand_params"])
    object_params = copy.deepcopy(sample["object_params"])
    hand_params.to_cpu()
    object_params.to_cpu()
    mesh_h = trimesh.Trimesh(vertices=hand_params.vertices, faces=hand_params.faces)
    mesh_h.visual.face_colors = COLOR_HUMAN_BLUE
    mesh_o = trimesh.Trimesh(vertices=object_params.vertices, faces=object_params.faces)
    mesh_o.visual.face_colors = COLOR_OBJECT_RED
    # re-apply the centroid offset of hand mesh
    centroid_offset = hand_params.centroid_offset
    mesh_h.apply_translation(centroid_offset)
    mesh_o.apply_translation(centroid_offset)
    mesh = mesh_h + mesh_o
    mesh_h.export(os.path.join(output_folder, f'pred_hand_mesh_phase{phase}.obj'))
    mesh_o.export(os.path.join(output_folder, f'pred_obj_mesh_phase{phase}.obj'))
    mesh.export(os.path.join(output_folder, f'pred_hoi_mesh_phase{phase}.obj'))
    # save rendered views
    # if phase == 3:
    #     visualize_human_object_results(img, img_filename, mesh, hand_params, output_folder)

    # save the outputs of the phase
    object_pose_data = {}
    ## ini-pose: the transform from canonical pose to PICO initial pose
    ini_rot = object_params.rotation_offset
    ini_trans = object_params.centroid_offset
    object_pose_data["init"] = {
        "rot": ini_rot.tolist(),
        "trans": ini_trans.tolist(),
    }

    ## pred-pose: the transfrom from PICO initial pose to PICO fitted pose (only for single phase)
    object_preds = {}
    object_preds["rot"] = (-object_phase_params["rotation"].squeeze()).tolist() # in PICO, rotation matrix is right-multiplied. But we want left-multiplied
    object_preds["trans"] = object_phase_params["translation"].squeeze().tolist()
    object_pose_data["pred"] = object_preds

    ## post-pose: the transform from PICO initial hand coordinate to GT hand coordinate
    object_pose_data["post"] = {
        "rot": [0., 0., 0.],
        "trans": centroid_offset.tolist()
    }

    if do_eval:
        metrics = sample["metrics"][f"phase{phase}"]
        out_path = os.path.join(output_folder, f'pred_phase{phase}.json')
        object_pose_data["gt"] = sample["gt_obj_pose"]
        save_data = {
            "pose": object_pose_data,
            "metrics": metrics
        }
    else:
        out_path = os.path.join(output_folder, f'pred_phase{phase}_noeval.json')
        save_data = {
            "pose": object_pose_data,
        }

    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=4)

    return

def postprocess_results(
    output_folder: str,
    do_eval: bool=False, 
):
    suffix = "" if do_eval else "_noeval"
    poses = dict()
    poses["pred"] = []
    for phase in range(1, 4):
        phase_json = os.path.join(output_folder, f"pred_phase{phase}{suffix}.json")
        if not os.path.exists(phase_json):
            continue

        with open(phase_json, "r") as f:
            phase_pose = json.load(f)["pose"]
        if "init" not in poses.keys():
            poses["init"] = phase_pose["init"]
        if "post" not in poses.keys():
            poses["post"] = phase_pose["post"]
        poses["pred"].append(phase_pose["pred"])
    
        # calculate the aggregated pose up to this phase
        R = torch.eye(4)
        init_mat = torch.eye(4)
        init_mat[:3, :3] = axis_angle_to_matrix(torch.FloatTensor(poses["init"]["rot"]))
        init_mat[:3, 3] = torch.FloatTensor(poses["init"]["trans"])
        R = init_mat @ R

        for pred in poses["pred"]:
            pred_mat = torch.eye(4)
            pred_mat[:3, :3] = axis_angle_to_matrix(torch.FloatTensor(pred["rot"]))
            pred_mat[:3, 3] = torch.FloatTensor(pred["trans"])
            R = pred_mat @ R
        
        post_mat = torch.eye(4)
        post_mat[:3, :3] = axis_angle_to_matrix(torch.FloatTensor(poses["post"]["rot"]))
        post_mat[:3, 3] = torch.FloatTensor(poses["post"]["trans"])
        R = post_mat @ R

        phase_save_r = {"transform": R.tolist()}
        r_save_json = os.path.join(output_folder, f"transform_phase{phase}.json")
        with open(r_save_json, "w") as f:
            json.dump(phase_save_r, f)
    
    return

def exist_results(output_path, do_eval, cfg):
    suffix = "" if do_eval else "_noeval"
    phases = []
    if not cfg.skip_phase_1:
        phases.append(1)
    if not cfg.skip_phase_2:
        phases.append(2)
    if not cfg.skip_phase_3:
        phases.append(3)

    check_file_templates = [
        "pred_hand_mesh_phase{}.obj",
        "pred_obj_mesh_phase{}.obj",
        "pred_hoi_mesh_phase{}.obj",
        f"pred_phase{{}}{suffix}.json",
        "transform_phase{}.json"
    ]

    for ph in phases:
        for cf in check_file_templates:
            check_path = os.path.join(output_path, cf.format(ph))
            if not os.path.exists(check_path):
                return False
    return True


def visualize_human_object_results(img, img_filename, mesh, hand_params, output_folder):
    ##### save the images
    frontal = visualize_frontal_overlaid(img, mesh, hand_params.centroid_offset, hand_params.bbox)
    top_down, left_image, right_image, back_image, front2 = render_side_views(mesh, img)
    
    # compose all 6 images into one
    top_row = np.concatenate((img, frontal, top_down), axis=1)
    bottomrow = np.concatenate((left_image, right_image, back_image), axis=1)
    combined = np.concatenate((top_row, bottomrow), axis=0)
    combined = Image.fromarray(combined)
    combined.save(os.path.join(output_folder, img_filename + '.jpg'))


def visualize_frontal_overlaid(img, mesh, human_offset, human_bbox):
    # bring mesh to camera frame
    mesh.apply_translation(human_offset.detach().cpu().numpy())

    # render overlaid view
    vis_img = render_overlaid_view(img, mesh, human_bbox.detach().cpu().numpy())
    vis_img = np.clip(vis_img, 0, 255).astype(np.uint8)
    return vis_img
