# $1 = run number, $2 = wordseg algo, $3 = dur_weight, $4 = data name

eval "$(conda shell.bash hook)"

cd ..

bash segmentation_ver2.bash $4_$2_dw$3_run$1 $2 $3 100 256 ./logs/may-6_codesize256_lr10-4_2000epochs.dir/may-6_codesize256_lr10-4_2000epochs__epoch_1000.model

conda activate spolacq

cd ../spolacq_mod/scripts/

python main.py --data_name=$4_$2_dw$3_run$1 \
--recog_dict=../../VQ-APC/results/$4_$2_dw$3_run$1_$2/recog_results_dict.pkl
