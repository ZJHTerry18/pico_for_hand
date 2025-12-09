input_dir="/projects/s5a/jiahezhao/codes/JointTransformer/outputs/render_out_v2.2"
output_dir="/projects/s5a/jiahezhao/codes/JointTransformer/outputs/pico_out_v2.2_test1"

python batch_run_generic.py \
    -d arctic \
    -i $input_dir -o $output_dir -r