eval "$(conda shell.bash hook)"

conda activate vq-apc
python VQ-APC_downstream_try.py --exp_name=7-mar_phoneseg \
--sound_file=./wavs/combined_sounds_shuffled.wav \
--pretrained_weights=./logs/mar-5_training_run2.dir/mar-5_training_run2__epoch_5.model \
--preprocess_path=./preprocessed/ \
--out_path=./results/

conda activate wordseg

python VQ-APC_word_seg.py --wordseg_algorithm=ag \
--phoneseg_interval_dir=./results/7-mar_phoneseg/intervals/ \
--output_dir=./results/7-mar_wordseg

conda activate vq-apc

python VQ-APC_split_wav.py --preprocessed_wav_path=./preprocessed/7-mar_phoneseg \
--wordseg_interval_dir=./results/7-mar_wordseg/intervals \
--output_dir=./results/7-mar_wordseg
