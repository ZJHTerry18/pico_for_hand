input_dir="/projects/s5a/jiahezhao/codes/JointTransformer/outputs/render_out_v1"
output_dir="/projects/s5a/jiahezhao/codes/JointTransformer/outputs/pico_out_v1_phase2_3metric_isambard"

python batch_run_generic.py \
    -d arctic \
    -i $input_dir -o $output_dir -r