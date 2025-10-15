import os
import os.path as osp
import json
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import Dataset

from src.dataset.meta import DIAMETERS
from src.utils.load_utils import load_image, load_hand_params, load_object_params
from src.utils.contact_mapping import load_contact_mapping, load_sparse_dense_mapping
from src.utils.geometry import axis_angle_to_matrix


class EpicDataset(Dataset):
    def __init__(self, data_dir : str, file_list : str):
        self.data_dir = data_dir
        self.file_list = file_list
        self.dataset_samples = []

        # Prepare data
        self.prepare_data_list()
    
    def prepare_data_list(self,):
        with open(self.file_list, "r") as f:
            file_names = f.readlines()
        folders = sorted([x for x in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, x))])
        id2folder = defaultdict(list)
        for folder in folders:
            # check if there exists annotations
            contact_ann_file = os.path.join(self.data_dir, folder, "corresponding_contacts.json")
            if not os.path.exists(contact_ann_file):
                continue
            
            # save the id of the video (remove annotator mark)
            id = "_".join(folder.split("_")[:-1])
            id2folder[id].append(folder)

        for n in file_names:
            n = n.strip()
            self.dataset_samples.extend(id2folder[n])
    
    def __len__(self,):
        return len(self.dataset_samples)
    
    def __getitem__(self, index):
        folder_name = self.dataset_samples[index]
        return folder_name

class ArcticDataset(Dataset):
    def __init__(self, data_dir: str, file_list: str=None, cfg=None):
        self.data_dir = data_dir
        self.file_list = file_list
        self.cfg = cfg
        self.dataset_samples = []

        self._prepare_data_list()
        self._load_meta()
    
    def _load_meta(self,):
        self.diameters = DIAMETERS["arctic"]

    def _prepare_data_list(self,):
        folders = sorted([x for x in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, x))])

        if self.file_list is not None:
            with open(self.file_list, "r") as f:
                file_names = f.readlines()
            for fol in folders:
                if fol in file_names:
                    self.dataset_samples.append(fol)
        else:
            self.dataset_samples.extend(folders)
    
    def _check_file(self, file, isdir=False):
        return osp.exists(file)
        
    def __len__(self,):
        return len(self.dataset_samples)

    def __getitem__(self, index):
        folder_name = self.dataset_samples[index]
        folder_path = osp.join(self.data_dir, folder_name)

        # load the hand, object and contact mapping
        lr_flag = "left" if "left" in folder_name else "right"
        image_path = osp.join(folder_path, "rgb.jpg")
        hand_mesh_path = osp.join(folder_path, f"{lr_flag}_hand_mesh.obj")
        obj_mesh_path = osp.join(folder_path, "object_posed_mesh.obj")
        contact_path = osp.join(folder_path, "corresponding_contacts.json")
        pos_to_canon = osp.join(folder_path, "obj_pose.json")
        camera_intrinsic_path = osp.join(folder_path, "cam_intrinsic.json")
        assert self._check_file(image_path)
        assert self._check_file(hand_mesh_path)
        assert self._check_file(obj_mesh_path), f"{obj_mesh_path} does not exist"
        assert self._check_file(contact_path)
        assert self._check_file(pos_to_canon)

        # get pose initialization matrix
        with open(pos_to_canon, "r") as f:
            d = json.load(f)
        pos2can_rot = -torch.FloatTensor(d["rot"])
        pos2can_trans = -torch.FloatTensor(d["trans"])
        init_mat = torch.zeros((4, 4), dtype=torch.float)
        init_mat[:3, :3] = axis_angle_to_matrix(pos2can_rot)
        init_mat[:, 3] = torch.cat((pos2can_trans, torch.FloatTensor([1.0])))

        # get camera intrinsic matrix
        cam_intrinsic = None
        render_img_size = [1000, 1000]
        if not (self.cfg.skip_phase_2 and self.cfg.skip_phase_3):
            assert self._check_file(camera_intrinsic_path), f"Camera intrinsic {camera_intrinsic_path} does not exist"
            with open(camera_intrinsic_path, "r") as f:
                cam_intrinsic = json.load(f)
            cx = int(cam_intrinsic[0][2])
            cy = int(cam_intrinsic[1][2])
            render_img_size = [cy * 2, cx * 2]
            cam_intrinsic = torch.FloatTensor(cam_intrinsic)
        
        # get object meta
        object_cls = folder_name.split("_")[1]
        diameter = self.diameters[object_cls]["diameter"]

        meta_info = {
            "diameter": diameter
        }

        # load image, hand parameters, object parameters, contact
        img = load_image(image_path)
        hand_params = load_hand_params(hand_mesh_path)
        object_params, gt_obj_vertices = load_object_params(
            obj_mesh_path, imgsize=render_img_size, trans_mat=init_mat, 
            load_obj_mask=(not self.cfg.skip_phase_2), cam_intrinsic=cam_intrinsic
        )
        contact_mapping = load_contact_mapping(contact_path, convert_to_smplx=False)

        # ## TEST: visualize the rendered masks
        # mask = object_params.mask.cpu().numpy()
        # mask = (mask * 255).astype(np.uint8)
        # import cv2
        # os.makedirs("temp", exist_ok=True)
        # cv2.imwrite(f"temp/{folder_name}.png", mask)

        sample = dict()
        sample["img"] = img
        sample["hand_params"] = hand_params
        sample["object_params"] = object_params
        sample["contact_mapping"] = contact_mapping
        sample["gt_pose"] = d # GT pose in 6-DoF
        sample["gt_vertices"] = gt_obj_vertices
        sample["render_size"] = render_img_size
        sample["cam_intrinsic"] = cam_intrinsic
        sample["meta"] = meta_info
        sample["metrics"] = dict()

        return sample, folder_name

DATASET_FACTORY = {
    "epic": EpicDataset,
    "arctic": ArcticDataset
}