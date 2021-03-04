CUDA_VISIBLE_DEVICES=1 python VQ-APC_downstream_try.py --exp_name=4-mar_combined_shuffled \
--sound_file=./wavs/combined_sounds_shuffled.wav \
--pretrained_weights=./logs/mar-3_phoneseg.dir/mar-3_phoneseg__epoch_1.model \
--preprocess_path=./preprocessed/ \
--out_path=./results/
