import os
import os.path as osp
import json
import numpy as np
from collections import defaultdict
from glob import glob
import torch
from torch.utils.data import Dataset

from src.dataset.meta import DIAMETERS
from src.utils.load_utils import load_image, load_hand_params, load_object_params
from src.utils.contact_mapping import load_contact_mapping, load_sparse_dense_mapping
from src.utils.geometry import axis_angle_to_matrix


class EpicDataset(Dataset):
    def __init__(self, data_dir : str, file_list : str=None, start_idx: int=0, end_idx: int=10**9,
                 cfg=None):
        self.data_dir = data_dir
        self.file_list = file_list
        self.cfg = cfg
        self.dataset_samples = []

        # Prepare data
        self.prepare_data_list(start_idx, end_idx)
    
    def prepare_data_list(self, start_idx, end_idx):
        # with open(self.file_list, "r") as f:
        #     file_names = f.readlines()
        # folders = sorted([x for x in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, x))])
        # id2folder = defaultdict(list)
        # for folder in folders:
        #     # check if there exists annotations
        #     contact_ann_file = os.path.join(self.data_dir, folder, "corresponding_contacts.json")
        #     if not os.path.exists(contact_ann_file):
        #         continue
            
        #     # save the id of the video (remove annotator mark)
        #     id = "_".join(folder.split("_")[:-1])
        #     id2folder[id].append(folder)

        # for n in file_names:
        #     n = n.strip()
        #     self.dataset_samples.extend(id2folder[n])
        
        # only load the best annotation for each id
        with open(self.file_list, "r") as f:
            file_list = json.load(f)
        for id, sample in file_list.items():
            if sample["best_dir"] is None:
                continue
            folder = sample["best_dir"].rstrip("/")
            contact_ann_file = os.path.join(self.data_dir, folder, "corresponding_contacts.json")
            if not os.path.exists(contact_ann_file):
                continue
            self.dataset_samples.append(folder)
        
        self.dataset_samples = self.dataset_samples[start_idx:end_idx]
    
    def __len__(self,):
        return len(self.dataset_samples)

    def _check_file(self, file, isdir=False):
        return osp.exists(file)
    
    def __getitem__(self, index):
        folder_name = self.dataset_samples[index]
        folder_path = osp.join(self.data_dir, folder_name)

        # load the hand, object and contact mapping
        lr_flag = "left" if "left" in folder_name else "right"
        image_path = glob(osp.join(folder_path, "frame_*.jpg"))[0]
        hand_mesh_path = osp.join(folder_path, f"{lr_flag}_hand_posed_mesh.ply")
        obj_mesh_path = osp.join(folder_path, "object.obj")
        contact_path = osp.join(folder_path, "corresponding_contacts.json")
        hand_npz_path = osp.join(folder_path, "wilor_output.pkl")
        hand_mask_path = osp.join(folder_path, "hand_mask.png")
        obj_mask_path = osp.join(folder_path, "object_mask.png")
        assert self._check_file(image_path)
        assert self._check_file(hand_mesh_path)
        assert self._check_file(obj_mesh_path), f"{obj_mesh_path} does not exist"
        assert self._check_file(contact_path)

        # get camera intrinsic matrix
        cam_intrinsic = None
        render_img_size = [456, 256]
        # TODO
        if not (self.cfg.skip_phase_2 and self.cfg.skip_phase_3):
            assert self._check_file(hand_npz_path), f"{hand_npz_path} does not exist"
            hand_npz = np.load(hand_npz_path, allow_pickle=True)[lr_flag]
            fl = hand_npz['focal_length'].item()
            cx, cy = hand_npz['img_size'][0], hand_npz['img_size'][1]
            render_img_size = [int(cy * 2), int(cx * 2)]
            cam_intrinsic = torch.FloatTensor([[fl, 0, cx], [0, fl, cy], [0, 0, 1]])

        # TODO: get objectmeta
        meta_info = dict()

        # load image, hand parameters, object parameters, contact
        img = load_image(image_path)
        load_hand_mask = (not self.cfg.skip_phase_3) and (self._check_file(hand_mask_path))
        load_obj_mask = (not self.cfg.skip_phase_2) and (self._check_file(obj_mask_path))
        load_mano = (not self.cfg.skip_phase_3)
        hand_params = load_hand_params(
            hand_mesh_path, hand_detection_file=hand_mask_path, imgsize=render_img_size, 
            lr_flag=lr_flag, center=True, load_hand_mask=load_hand_mask, load_mano=load_mano,
            cam_intrinsic=cam_intrinsic, hand_npz=hand_npz,
        )
        object_params, _ = load_object_params(
            obj_mesh_path, object_detection_file=obj_mask_path, imgsize=render_img_size, 
            trans_mat=None, load_obj_mask=load_obj_mask, cam_intrinsic=cam_intrinsic
        )
        contact_mapping = load_contact_mapping(contact_path, convert_to_smplx=False)
        sparse_dense_mapping = load_sparse_dense_mapping("./sparse_dense_mapping.json")

        # ## TEST: visualize the rendered masks
        # obj_mask = object_params.mask.cpu().numpy()
        # obj_mask = (obj_mask * 255).astype(np.uint8)
        # hand_mask = hand_params.mask.cpu().numpy()
        # hand_mask = (hand_mask * 255).astype(np.uint8)
        # print(obj_mask.shape, hand_mask.shape)
        # import cv2
        # os.makedirs("temp", exist_ok=True)
        # cv2.imwrite(f"temp/{folder_name}_obj.png", obj_mask)
        # cv2.imwrite(f"temp/{folder_name}_hand.png", hand_mask)

        sample = dict()
        sample["img"] = img
        sample["hand_params"] = hand_params
        sample["object_params"] = object_params
        sample["contact_mapping"] = contact_mapping
        sample["sparse_dense_mapping"] = sparse_dense_mapping
        sample["render_size"] = render_img_size
        sample["cam_intrinsic"] = cam_intrinsic
        sample["meta_info"] = meta_info
        sample["metrics"] = dict()

        return sample, folder_name

class ArcticDataset(Dataset):
    def __init__(self, data_dir: str, file_list: str=None, start_idx: int=0, end_idx: int=10**9, 
                 cfg=None):
        self.data_dir = data_dir
        self.file_list = file_list
        self.cfg = cfg
        self.dataset_samples = []

        self._prepare_data_list(start_idx, end_idx)
        self._load_meta()
    
    def _load_meta(self,):
        self.diameters = DIAMETERS["arctic"]

    def _prepare_data_list(self, start_idx, end_idx):
        folders = sorted(glob(osp.join(self.data_dir, "*/*/*/*")))
        folders = [f for f in folders if self._check_data(f)]

        if self.file_list is not None:
            with open(self.file_list, "r") as f:
                file_names = f.readlines()
            for fol in folders:
                if osp.basename(fol) in file_names:
                    self.dataset_samples.append(fol)
        else:
            self.dataset_samples.extend(folders)
        self.dataset_samples = self.dataset_samples[start_idx:end_idx]
    
    def _check_file(self, file, isdir=False):
        return osp.exists(file)

    def _check_data(self, folder_path):
        lr_path = osp.join(folder_path, "left_right.txt")
        return self._check_file(lr_path)

    def __len__(self,):
        return len(self.dataset_samples)

    def __getitem__(self, index):
        folder_path = self.dataset_samples[index]
        folder_name = folder_path[len(self.data_dir):].lstrip("/")

        # load the hand, object and contact mapping
        # TODO: change the method of checking left and right
        lr_path = osp.join(folder_path, "left_right.txt")
        with open(lr_path, "r") as f:
            lr_flag = f.readlines()[0].strip()
        assert lr_flag in ["left", "right"]
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
        object_cls = folder_path.split("/")[-3].split("_")[0]
        diameter = self.diameters[object_cls]["diameter"]

        meta_info = {
            "diameter": diameter
        }

        # load image, hand parameters, object parameters, contact
        img = load_image(image_path)
        hand_params = load_hand_params(hand_mesh_path, lr_flag)
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
        sample["gt_obj_pose"] = d # GT pose in 6-DoF
        sample["gt_obj_vertices"] = gt_obj_vertices
        sample["render_size"] = render_img_size
        sample["cam_intrinsic"] = cam_intrinsic
        sample["meta_info"] = meta_info
        sample["metrics"] = dict()

        return sample, folder_name

DATASET_FACTORY = {
    "epic": EpicDataset,
    "arctic": ArcticDataset
}