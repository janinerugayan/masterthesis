# $1 = run number, $2 = wordseg algo, $3 = dur_weight

eval "$(conda shell.bash hook)"

cd ..

bash segmentation.bash $4_$2_dw$3_run$1 $2 $3 100 128 ./logs/apr-19_codesize128_1208-2000epochs.dir/apr-19_codesize128_1208-2000epochs__epoch_2000.model

conda activate spolacq

cd ../spolacq_mod/scripts/

python main.py --data_name=$4_$2_run$1 \
--recog_dict=../../VQ-APC/results/$4_$2_dw$3_run$1_$2/recog_results_dict.pkl
