#!/bin/bash
#SBATCH --job-name=parallel_pico_epic
#SBATCH --nodes=16
#SBATCH --ntasks=64
#SBATCH --gpus-per-node=4
#SBATCH --time=05:00:00
#SBATCH --output=logs/epic-%j.out
#SBATCH --error=logs/epic-%j.err
#SBATCH --mail-user=jiahe.zhao@bristol.ac.uk
#SBATCH --mail-type=ALL

source ~/.bashrc
conda activate pico
mkdir -p logs

TOTAL_SAMPLES=321
INPUT_DIR="/home/s5a/jiahezhao25.s5a/jiahe/data/epic-grasps/2025-09-08_gemini_pro"
OUTPUT_DIR="/home/s5a/jiahezhao25.s5a/jiahe/data/epic-grasps/2025-10-24_pico_stage2_321videos_wilorspace/2025-10-24_pico_stage2_321videos_wilorspace_con8.0_pen1.0"
# FILE_LIST="/home/s5a/jiahezhao25.s5a/jiahe/data/epic-grasps/stage2_annotated_id_20251010.txt"
FILE_LIST="/home/s5a/jiahezhao25.s5a/jiahe/data/epic-grasps/best_dirs_annotations.json"

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
    START_IDX=\$(( GLOBAL_RANK * $CHUNK_SIZE ))
    END_IDX=\$(( START_IDX + $CHUNK_SIZE ))

    echo \"Task \$GLOBAL_RANK (Local GPU: \$LOCAL_RANK) assigned range \$START_IDX to \$END_IDX\"

    python batch_run_generic.py \
        -d epic \
        -i $INPUT_DIR -o $OUTPUT_DIR -l $FILE_LIST -r --debug \
        --start \$START_IDX --end \$END_IDX 
"

echo "End Time: $(date +%Y-%m-%d\ %H:%M:%S)"