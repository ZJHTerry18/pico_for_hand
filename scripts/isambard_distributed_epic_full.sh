#!/bin/bash
#SBATCH --job-name=parallel_pico_epic
#SBATCH --nodes=20
#SBATCH --ntasks=1280
#SBATCH --gpus-per-node=4
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/epic-%j.out
#SBATCH --error=logs/epic-%j.err
#SBATCH --mail-user=jiahe.zhao@bristol.ac.uk
#SBATCH --mail-type=ALL

GPUS_PER_NODE=4

source ~/.bashrc
conda activate pico
mkdir -p logs

TOTAL_SAMPLES=2400
INPUT_DIR="/home/u5gi/jiahezhao25.u5gi/jiahe/data/epic-grasps/epic-contact_2026-02-17_full"
OUTPUT_DIR="/home/u5gi/jiahezhao25.u5gi/jiahe/data/epic-grasps/pico_v3/pico_stage3_fullvideos/2026-02-19_pico_stage3_rand100videos_wilorspace_maskv3_newsf_multimixed_[template]"
# FILE_LIST="/home/u5gi/jiahezhao25.u5gi/jiahe/data/epic-grasps/stage2_annotated_id_20251010.txt"
FILE_LIST="/home/u5gi/jiahezhao25.u5gi/jiahe/data/epic-grasps/best_dirs_annotations_full.json"

N_TASKS=$SLURM_NTASKS
if [ -z "$N_TASKS" ]; then
    echo "Error: SLURM_NTASKS not set. Did you run this script outside of Slurm?"
    exit 1
fi
CHUNK_SIZE=$((( TOTAL_SAMPLES / N_TASKS ) + 1))

echo "Total Samples: $TOTAL_SAMPLES"
echo "Total Tasks: $N_TASKS"
echo "Chunk Size (samples per task): $CHUNK_SIZE"
echo "Starting job on $SLURM_NNODES nodes."
echo "Start Time: $(date +%Y-%m-%d\ %H:%M:%S)"

srun bash -c "
    GLOBAL_RANK=\$SLURM_PROCID
    LOCAL_RANK=\$SLURM_LOCALID
    DEVICE_ID=\$(( LOCAL_RANK % $GPUS_PER_NODE ))
    export CUDA_VISIBLE_DEVICES=\$DEVICE_ID
    START_IDX=\$(( GLOBAL_RANK * $CHUNK_SIZE ))
    END_IDX=\$(( START_IDX + $CHUNK_SIZE ))

    echo \"Task \$GLOBAL_RANK (Node: \$LOCAL_RANK) using GPU: \$DEVICE_ID assigned range \$START_IDX to \$END_IDX\"

    python batch_run_generic.py \
        -d epic \
        -i $INPUT_DIR -o $OUTPUT_DIR -l $FILE_LIST \
        --start \$START_IDX --end \$END_IDX
"

echo "End Time: $(date +%Y-%m-%d\ %H:%M:%S)"