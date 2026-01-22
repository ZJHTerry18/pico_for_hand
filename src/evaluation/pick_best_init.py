import os
import os.path as osp
import json
import sys
from tqdm import tqdm
import shutil

result_dir = '/home/u5gi/jiahezhao25.u5gi/jiahe/data/epic-grasps/pico_v3/pico_stage3_1007videos_wilorspace/2026-01-15_pico_stage3_1007videos_wilorspace_con8.0_sil0.03-occ_pen0.01_sc1.0_reg0.05_maskv2_multiprior'
separate_save = '/home/u5gi/jiahezhao25.u5gi/jiahe/data/epic-grasps/pico_v3/pico_stage3_1007videos_wilorspace/2026-01-15_pico_stage3_1007videos_wilorspace_con8.0_sil0.03-occ_pen0.01_sc1.0_reg0.05_maskv2_multiprior-selbest'
os.makedirs(separate_save, exist_ok=True)

sample_folders = [x for x in sorted(os.listdir(result_dir)) if osp.isdir(osp.join(result_dir, x))]
for sf in tqdm(sample_folders):
    try:
        sample_dir = osp.join(result_dir, sf)
        init_fols = [x for x in sorted(os.listdir(sample_dir)) if osp.isdir(osp.join(sample_dir, x))]

        losses_list = []
        for inif in init_fols:
            loss_file = osp.join(sample_dir, inif, "pred_phase2_noeval.json")
            with open(loss_file, 'r') as f:
                loss_dict = json.load(f)["loss"]
            total_loss = sum(list(loss_dict.values()))
            losses_list.append(total_loss)
        # choose the one with the smallest total loss
        min_loss_ind = min(enumerate(losses_list), key=lambda x: x[1])[0]
        best_init_fol = init_fols[min_loss_ind]

        with open(f"{sample_dir}/{best_init_fol}.txt", "w") as f:
            f.write("")

        if separate_save:
            src_fol = osp.join(sample_dir, best_init_fol)
            tgt_fol = osp.join(separate_save, sf)
            shutil.copytree(src_fol, tgt_fol)
    except:
        continue
    