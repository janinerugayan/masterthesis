eval "$(conda shell.bash hook)"

conda activate vq-apc

python VQ-APC_downstream_try.py --exp_name=10-mar_1utt_phoneseg \
--sound_file=./wavs/numbers_shuffled.wav \
--pretrained_weights=./logs/mar-9_training_10epochs.dir/mar-9_training_10epochs__epoch_10.model \
--preprocess_path=./preprocessed/ \
--out_path=./results/

conda activate wordseg

python VQ-APC_word_seg.py --wordseg_algorithm=ag \
--phoneseg_interval_dir=./results/10-mar_1utt_phoneseg/intervals/ \
--output_dir=./results/10-mar_1utt_wordseg_ag

conda activate vq-apc

python VQ-APC_split_wav.py --preprocessed_wav_path=./preprocessed/10-mar_1utt_phoneseg/ \
--wordseg_interval_dir=./results/10-mar_1utt_wordseg_ag/intervals/ \
--output_dir=./results/10-mar_1utt_wordseg_ag

python VQ-APC_stt_recog.py --wav_path=./results/10-mar_1utt_wordseg_ag/wavs \
--output_dir=./results/10-mar_1utt_wordseg_ag
