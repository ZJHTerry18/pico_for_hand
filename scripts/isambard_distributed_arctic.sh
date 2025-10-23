#!/bin/bash
#SBATCH --job-name=parallel_pico_arctic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=05:00:00
#SBATCH --output=logs/arctic-%j.out
#SBATCH --error=logs/arctic-%j.err
#SBATCH --mail-user=jiahe.zhao@bristol.ac.uk
#SBATCH --mail-type=ALL

source ~/.bashrc
conda activate pico
mkdir -p logs

TOTAL_SAMPLES=138
INPUT_DIR="/projects/s5a/jiahezhao/codes/JointTransformer/outputs/render_out_v2.2_test"
OUTPUT_DIR="/projects/s5a/jiahezhao/codes/JointTransformer/outputs/pico_out_v2.2-test_phase2_3metric_isambard"

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
        -d arctic \
        -i $INPUT_DIR -o $OUTPUT_DIR -r --do_eval \
        --start \$START_IDX --end \$END_IDX 
"

echo "End Time: $(date +%Y-%m-%d\ %H:%M:%S)"