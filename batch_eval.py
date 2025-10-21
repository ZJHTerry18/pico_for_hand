import os
import os.path as osp
import json
from collections import defaultdict
import numpy as np
import argparse

def eval_result(folder_list, phase):
    all_results = defaultdict(list)
    for folder in folder_list:
        pred_res_file = osp.join(folder, f"pred_phase{phase}.json")

        if osp.exists(pred_res_file): # directly read from json
            with open(pred_res_file, "r") as f:
                res = json.load(f)
            metric_res = res["metrics"]
            for metric in metric_res.keys():
                all_results[metric].append(metric_res[metric])
        else:
            raise NotImplementedError("Non-json evaluation methods not implemented yet")
    
    print(f"== Phase {phase} Evaluation Results ==")
    for metric in all_results.keys():
        avg_value = np.mean(all_results[metric])
        print(f"{metric}: {avg_value:.2f}")
    print("=======================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, required=True, help="input directory")
    args = parser.parse_args()

    folder_list = [osp.join(args.i, x) for x in os.listdir(args.i) if osp.isdir(osp.join(args.i, x))]
    folder_list = sorted(folder_list)

    for phase in range(1, 4):
        eval_result(folder_list, phase)