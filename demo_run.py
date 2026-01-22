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
from src.utils.save_results import save_phase_results, postprocess_results, exist_results
from src.config_packs import default_config, default_loss_weights
from src.model.phase1_contact import optimize_phase1_contact
from src.model.phase2_image import optimize_phase2_image
from src.model.phase3_hand import optimize_phase3_hand
from src.evaluation.eval_modules import eval_v2v_success, eval_contact_dev, eval_mrrpe

def run_sample(folder_name, output_path, sample, args, **kwargs):
    # save the initial pose to check the object pose initialization
    save_phase_results(folder_name, output_path, sample, phase=0, do_eval=args.do_eval)
    
    if not cfg.skip_phase_1:
        p1_object_params = optimize_phase1_contact(**kwargs)
        sample["object_params"].vertices = p1_object_params["vertices"]
        torch.cuda.empty_cache()
        # evaluate stage 1
        if args.do_eval:
            sample["metrics"]["phase1"] = evaluation(sample)
        # save results
        save_phase_results(folder_name, output_path, sample, object_phase_params=p1_object_params, phase=1, do_eval=args.do_eval)

    if not cfg.skip_phase_2:
        p2_object_params = optimize_phase2_image(**kwargs)
        sample["object_params"].vertices = p2_object_params['vertices']
        torch.cuda.empty_cache()
        # evaluate stage 2
        if args.do_eval:
            sample["metrics"]["phase2"] = evaluation(sample)
        # save results
        save_phase_results(folder_name, output_path, sample, object_phase_params=p2_object_params, phase=2, do_eval=args.do_eval)           

    if not cfg.skip_phase_3:
        p3_hand_params = optimize_phase3_hand(**kwargs)
        sample["hand_params"].vertices = p3_hand_params['vertices']
        torch.cuda.empty_cache()
        # evaluate stage 3
        if args.do_eval:
            sample["metrics"]["phase3"] = evaluation(sample)
        save_phase_results(folder_name, output_path, sample, hand_phase_params=p3_hand_params, phase=3, do_eval=args.do_eval)            

    postprocess_results(output_path, args.do_eval)

def main(dataset, args, cfg = None, loss_weights = None):
    if cfg is None:
        cfg = default_config
    if loss_weights is None:
        loss_weights = default_loss_weights

    for i, datas in enumerate(dataset):
        single_data = False
        if not isinstance(datas, list):
            single_data = True
            datas = [datas]
        for init_i, data in enumerate(datas):
            sample, folder_name = data
            if not folder_name == "P01_01_43455_44975_left_pan.mp4":
                continue
            kwargs = {**sample, **vars(cfg)}
            if single_data:
                output_path = osp.join(args.output_dir, folder_name)
            else:
                output_path = osp.join(args.output_dir, folder_name, f"init{init_i}")

            # check whether to skip this task
            if not args.rewrite and exist_results(output_path, args.do_eval, cfg):
                print(f"--> Skipping sample '{folder_name}' as it has already been processed.")
                torch.cuda.empty_cache()
                continue
            
            if args.debug:
                run_sample(folder_name, output_path, sample, args, **kwargs)    
            else:
                try:
                    run_sample(folder_name, output_path, sample, args, **kwargs)
                except Exception as e:
                    print(f"Sample {folder_name} induces error: {e}. Skip for now.")
                    continue

def evaluation(sample):
    metrics = {}
    v2v_success = eval_v2v_success(
        sample["gt_obj_vertices"], 
        sample["object_params"].vertices.detach().cpu(),
        sample["meta_info"],
    )
    cdev = eval_contact_dev(
        sample["hand_params"].vertices.detach().cpu(),
        sample["object_params"].vertices.detach().cpu(),
        sample["contact_mapping"],
    )
    mrrpe = eval_mrrpe(
        sample["hand_params"].vertices.detach().cpu(), # we don't change hand at this moment, so pred equals to gt
        sample["hand_params"].vertices.detach().cpu(),
        sample["gt_obj_vertices"],
        sample["object_params"].vertices.detach().cpu(),
        sample["meta_info"]
    )
    metrics.update(v2v_success)
    metrics.update(cdev)
    metrics.update(mrrpe)

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser("PICO-fit-for-hand, input parameters")
    parser.add_argument("--dataset", "-d", type=str, help="dataset name")
    parser.add_argument("--data_dir", "-i", type=str, help="dataset directory")
    parser.add_argument("--file_list", "-l", type=str, default=None, help="list (.txt) of all the data to process")
    parser.add_argument("--output_dir", "-o", type=str, help="output directory")
    parser.add_argument("--do_eval", "-e", action="store_true", help="do evaluation")
    parser.add_argument("--rewrite", "-r", action="store_true", help="rewrite the outputs even if they exist")
    parser.add_argument("--start", type=int, default=0, help="start index of the dataset")
    parser.add_argument("--end", type=int, default=10**9, help="end index of the dataset")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Setup configurations and dataset
    cfg = CONFIGS_FACTORY[args.dataset]
    hand_dataset = DATASET_FACTORY[args.dataset](
        data_dir=args.data_dir,
        file_list=args.file_list,
        start_idx=args.start,
        end_idx=args.end,
        cfg=cfg,
    )
    print(f"From {args.start} - {args.end}: {len(hand_dataset)} valid samples loaded.")
    
    main(hand_dataset, args, cfg=cfg)