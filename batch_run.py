import sys
import os
import argparse
from tqdm import tqdm
import glob

from dataset.hand_dataset import HandDataset
from src.utils.load_utils import load_image, load_hand_params, load_object_params
from src.utils.contact_mapping import load_contact_mapping, load_sparse_dense_mapping
from src.utils.save_results import save_phase_results
from src.config_packs import default_config, default_loss_weights
from src.phase1_contact import optimize_phase1_contact
# from src.phase2_image import optimize_phase2_image
# from src.phase3_human import optimize_phase3_human


def main(
        input_folder: str,
        output_folder: str,
        cfg = None,
        loss_weights = None
    ):
    if cfg is None:
        cfg = default_config
    if loss_weights is None:
        loss_weights = default_loss_weights

    img_filename = [x for x in os.listdir(input_folder) if x.endswith('.jpg')][0]
    img = load_image(os.path.join(input_folder, img_filename))

    output_end = output_folder.split("/")[-1]
    if glob.glob(os.path.join(output_folder, "*.obj")):
        print(f"--> Skipping folder '{output_end}' as it has already been processed.")
        return

    # determine left or right hand
    lr_flag = "left" if "left" in input_folder else "right"
    hand_inference_file = lr_flag + "_" + cfg.hand_inference_file
    hand_params = load_hand_params(
        os.path.join(input_folder, hand_inference_file),
        # os.path.join(input_folder, cfg.human_detection_file),
        # img.shape[:2]
    )
    object_params = load_object_params(
        os.path.join(input_folder, cfg.object_mesh_file),
        # os.path.join(input_folder, cfg.object_detection_file),
        # img.shape[:2]
    )
    contact_mapping = load_contact_mapping(
        os.path.join(input_folder, cfg.contact_mapping_file),
        convert_to_smplx=False
    )
    sparse_dense_mapping = load_sparse_dense_mapping(
        cfg.sparse_dense_mapping_file
    )

    
    if not cfg.skip_phase_1:
        p1_object_params = optimize_phase1_contact(hand_params, object_params, contact_mapping, sparse_dense_mapping, cfg.nr_phase_1_steps)
        object_params.vertices = p1_object_params['vertices']
        save_phase_results(
            output_end, output_folder, img,
            hand_params, object_params,
            phase = 1,
        )


    if not cfg.skip_phase_2:
        p2_object_params = optimize_phase2_image(hand_params, object_params, contact_mapping, img, loss_weights, cfg.nr_phase_2_steps)
        object_params.vertices = p2_object_params['vertices']
        object_params.scale = p2_object_params['scaling']
        save_phase_results(
            img_filename, output_folder, img,
            hand_params, object_params,
            phase = 2,
        )


    if not cfg.skip_phase_3:
        p3_hand_params = optimize_phase3_human(hand_params, object_params, contact_mapping, img, loss_weights, cfg.nr_phase_3_steps)
        hand_params.vertices = p3_hand_params['vertices']
        save_phase_results(
            img_filename, output_folder, img,
            hand_params, object_params,
            phase = 3,
        )



if __name__ == "__main__":
    parser = argparse.ArgumentParser("PICO-fit-for-hand, input parameters")
    parser.add_argument("--input_dir", "-d", type=str, help="dataset directory")
    parser.add_argument("--list", "-l", type=str, help="list (.txt) of all the data to process")
    parser.add_argument("--output_dir", "-o", type=str, help="output directory")
    args = parser.parse_args()

    # Set custom configuration
    cfg = default_config
    ## Currently we only run phase 1
    cfg.skip_phase_1 = False
    cfg.nr_phase_1_steps = 150
    cfg.skip_phase_2 = True
    cfg.skip_phase_3 = True

    hand_dataset = HandDataset(
        data_dir=args.input_dir,
        file_list=args.list,
    )
    
    for i, data in tqdm(enumerate(hand_dataset), total=len(hand_dataset)):
        try:
            # Load one sample for optimization
            input_folder = os.path.join(args.input_dir, data) # __getitem__ returns the folder name
            output_folder = os.path.join(args.output_dir, data)
            main(input_folder, output_folder, cfg=cfg)
        except Exception as e:
            print(f"Error encountered during optimizing data {data}: {e}. Skip this sample")