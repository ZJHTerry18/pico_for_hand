import sys
import os
import os.path as osp
import argparse
from tqdm import tqdm
import glob
import copy
import torch

from src.config_packs import CONFIGS_FACTORY
from src.dataset.hand_dataset import DATASET_FACTORY
from src.utils.save_results import save_phase_results
from src.config_packs import default_config, default_loss_weights
from src.model.phase1_contact import optimize_phase1_contact
from src.model.phase2_image import optimize_phase2_image
# from src.phase3_human import optimize_phase3_human
from src.evaluation.eval_modules import eval_v2v_success


def main(dataset, args, cfg = None, loss_weights = None):
    if cfg is None:
        cfg = default_config
    if loss_weights is None:
        loss_weights = default_loss_weights

    for i, data in enumerate(dataset):
        sample, folder_name = data

        output_path = osp.join(args.output_dir, folder_name)
        if glob.glob(os.path.join(output_path, "*.obj")) and not args.rewrite:
            print(f"--> Skipping sample '{folder_name}' as it has already been processed.")
            return

        # sample["lr_rotation_phase_1"] = cfg.lr_rotation_phase_1
        # sample["lr_translation_phase_1"] = cfg.lr_translation_phase_1
        # sample["nr_phase_1_steps"] = cfg.nr_phase_1_steps
        # sample["nr_phase_2_steps"] = cfg.nr_phase_2_steps
        # sample["loss_weights"] = loss_weights
        kwargs = {**sample, **vars(cfg)}
        # try: # fitting part starts here
        if not cfg.skip_phase_1:
            p1_object_params = optimize_phase1_contact(**kwargs)
            sample["object_params"].vertices = p1_object_params["vertices"]
            torch.cuda.empty_cache()
            # evaluate stage 1
            metrics = eval_v2v_success(sample["gt_vertices"], sample["object_params"].vertices.detach().cpu(), sample["meta"])
            sample["metrics"]["phase1"] = metrics
            print(metrics)
            # save results
            save_phase_results(folder_name, output_path, sample, p1_object_params, phase=1)


        if not cfg.skip_phase_2:
            p2_object_params = optimize_phase2_image(**kwargs)
            sample["object_params"].vertices = p2_object_params['vertices']
            # sample["object_params"].scale = p2_object_params['scaling']
            metrics = eval_v2v_success(sample["gt_vertices"], sample["object_params"].vertices.detach().cpu(), sample["meta"])
            sample["metrics"]["phase2"] = metrics
            print(metrics)
            save_phase_results(folder_name, output_path, sample, p2_object_params, phase=2)

            torch.cuda.empty_cache()            


        # if not cfg.skip_phase_3:
        #     p3_hand_params = optimize_phase3_human(hand_params, object_params, contact_mapping, img, loss_weights, cfg.nr_phase_3_steps)
        #     hand_params.vertices = p3_hand_params['vertices']
        #     save_phase_results(
        #         img_filename, output_folder, img,
        #         hand_params, object_params,
        #         phase = 3,
        #     )
        # except Exception as e:
        #     print(f"Sample {folder_name} induces error: {e}. Skip for now.")
        #     continue



if __name__ == "__main__":
    parser = argparse.ArgumentParser("PICO-fit-for-hand, input parameters")
    parser.add_argument("--dataset", "-d", type=str, help="dataset name")
    parser.add_argument("--data_dir", "-i", type=str, help="dataset directory")
    parser.add_argument("--file_list", "-l", type=str, default=None, help="list (.txt) of all the data to process")
    parser.add_argument("--output_dir", "-o", type=str, help="output directory")
    parser.add_argument("--rewrite", "-r", action="store_true", help="rewrite the outputs even if they exist")
    args = parser.parse_args()

    # Setup configurations and dataset
    cfg = CONFIGS_FACTORY[args.dataset]
    hand_dataset = DATASET_FACTORY[args.dataset](
        data_dir=args.data_dir,
        file_list=args.file_list,
        cfg=cfg,
    )
    
    main(hand_dataset, args, cfg=cfg)