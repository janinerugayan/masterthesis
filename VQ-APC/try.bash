# cd preprocessed
# rm -rv 16-mar_check*
# cd ../results
# rm -rv 16-mar_check*
# cd ..

eval "$(conda shell.bash hook)"

conda activate vq-apc

python VQ-APC_downstream_try.py --exp_name=18-mar_check \
--sound_file=./wavs/numbers_shuffled.wav \
--pretrained_weights=./logs/mar-13_training_100epochs.dir/mar-13_training_100epochs__epoch_100.model \
--preprocess_path=./preprocessed/ \
--out_path=./results/

conda activate wordseg

python VQ-APC_word_seg.py --wordseg_algorithm=ag \
--phoneseg_interval_dir=./results/18-mar_check/intervals/ \
--output_dir=./results/18-mar_check_ag-wordseg

conda activate vq-apc

python VQ-APC_split_wav.py --preprocessed_wav_path=./preprocessed/18-mar_check/ \
--wordseg_interval_dir=./results/18-mar_check_ag-wordseg/intervals/ \
--output_dir=./results/18-mar_check_ag-wordseg

python VQ-APC_stt_recog.py --wav_path=./results/18-mar_check_ag-wordseg/wavs \
--output_dir=./results/18-mar_check_ag-wordseg
