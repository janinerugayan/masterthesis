cd ..

bash VQ-APC_segmentation.bash may-12_size128epoch2000 ag 36 100 ./logs/apr-19_codesize128_1208-2000epochs.dir/apr-19_codesize128_1208-2000epochs__epoch_2000.model

cd ../spolacq_mod/scripts/

python main.py --data_name=may-12_VQ-spolacq_size128epoch2000 \
--recog_dict=../../VQ-APC/results/may-12_size128epoch2000_ag/recog_results_dict.pkl 
