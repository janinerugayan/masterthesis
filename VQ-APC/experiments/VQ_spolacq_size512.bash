cd ..

bash VQ-APC_segmentation.bash may-12_size512epoch2000 ag 36 100 ./logs/apr-17_codesize512_lr10-4_2000epochs.dir/apr-17_codesize512_lr10-4_2000epochs__epoch_2000.model

cd ../spolacq_mod/scripts/

python main.py --data_name=may-12_VQ-spolacq_size512epoch2000 \
--recog_dict=../../VQ-APC/results/may-12_size512epoch2000_ag/recog_results_dict.pkl
