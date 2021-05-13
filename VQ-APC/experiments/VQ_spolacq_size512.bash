eval "$(conda shell.bash hook)"

cd ..

bash segmentation.bash may-13_size512epoch2000_run2 tp 36 100 512 ./logs/apr-17_codesize512_lr10-4_2000epochs.dir/apr-17_codesize512_lr10-4_2000epochs__epoch_2000.model

conda activate spolacq

cd ../spolacq_mod/scripts/

python main.py --data_name=may-13_VQ-spolacq_size512epoch2000_run2 \
--recog_dict=../../VQ-APC/results/may-13_size512epoch2000_run2_tp/recog_results_dict.pkl
