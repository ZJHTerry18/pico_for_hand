import torch
import numpy as np
from typing import Union

from src.evaluation.metrics import compute_v2v_dist_no_reduce

def eval_v2v_success(v3d_gt: torch.Tensor, v3d_pred: torch.Tensor, meta_info: dict):
    '''
    Evaluate the success rate between predicted mesh and ground-truth mesh by Fan. et al [1]
    Args:
        v3d_gt: vertices of ground-truth object, (N, 3)
        v3d_pred: vertices of predicted object, (N, 3)
        meta_info: dict, containing the diameter of the object
    Return:
        metric_dict: dict, success rate for different alphas (typically [0.05])
    
    [1] Fan, et al. "ARCTIC: A dataset for dexterous bimanual hand-object manipulation." CVPR. 2023.
    '''

    if 'part_ids' not in meta_info:
        bottom_idx = torch.ones((v3d_gt.shape[0],), dtype=torch.long) * 2
    else:
        bottom_idx = meta_info["part_ids"] == 2
    bottom_idx = bottom_idx.nonzero().view(-1)
    
    v3d_root_gt = v3d_gt[bottom_idx].mean(dim=0)
    v3d_root_pred = v3d_pred[bottom_idx].mean(dim=0)

    v3d_cam_gt_ra = v3d_gt - v3d_root_gt[None, :]
    v3d_cam_pred_ra = v3d_pred - v3d_root_pred[None, :]

    v2v_ra = compute_v2v_dist_no_reduce(v3d_cam_gt_ra, v3d_cam_pred_ra)

    diameter = meta_info["diameter"]
    alphas = [0.01, 0.03, 0.05, 0.1]
    metric_dict = dict()
    for alpha in alphas:
        v2v_rate_ra = (v2v_ra < diameter * alpha).astype(np.float32)
        success = v2v_rate_ra.sum()
        v2v_rate_ra = success / v2v_rate_ra.shape[0]
        # percentage
        metric_dict[f"success_rate/{alpha:.2f}"] = v2v_rate_ra
    metric_dict = {k: v * 100 for k, v in metric_dict.items()}
    return metric_dict