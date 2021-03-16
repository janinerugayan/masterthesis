cd preprocessed
rm -rv 16-mar_check*
cd ../results
rm -rv 16-mar_check*
cd ..

eval "$(conda shell.bash hook)"

conda activate vq-apc

python VQ-APC_downstream_try.py --exp_name=16-mar_check_phoneseg \
--sound_file=./wavs/numbers_shuffled.wav \
--pretrained_weights=./logs/mar-13_training_100epochs.dir/mar-13_training_100epochs__epoch_100.model \
--preprocess_path=./preprocessed/ \
--out_path=./results/

conda activate wordseg

python VQ-APC_word_seg.py --wordseg_algorithm=tp \
--phoneseg_interval_dir=./results/16-mar_check_phoneseg/intervals/ \
--output_dir=./results/16-mar_check_wordseg_tp

conda activate vq-apc

python VQ-APC_split_wav.py --preprocessed_wav_path=./preprocessed/16-mar_check_phoneseg/ \
--wordseg_interval_dir=./results/16-mar_check_wordseg_tp/intervals/ \
--output_dir=./results/16-mar_check_wordseg_tp

python VQ-APC_stt_recog.py --wav_path=./results/16-mar_check_wordseg_tp/wavs \
--output_dir=./results/16-mar_check_wordseg_tp
