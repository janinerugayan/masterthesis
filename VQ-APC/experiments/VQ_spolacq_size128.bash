eval "$(conda shell.bash hook)"

cd ..

bash segmentation.bash may-13_size128epoch2000 tp 36 100 128 ./logs/apr-19_codesize128_1208-2000epochs.dir/apr-19_codesize128_1208-2000epochs__epoch_2000.model

conda activate spolacq

cd ../spolacq_mod/scripts/

python main.py --data_name=may-13_VQ-spolacq_size128epoch2000 \
--recog_dict=../../VQ-APC/results/may-13_size128epoch2000_tp/recog_results_dict.pkl
