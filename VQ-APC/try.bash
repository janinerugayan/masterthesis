cd preprocessed
rm -rv 4-mar_combined_shuffled
cd ..
cd results
rm -rv 4-mar_combined_shuffled
cd ..

CUDA_VISIBLE_DEVICES=1 python VQ-APC_downstream_try.py --exp_name=5-mar_phoneseg \
--sound_file=./wavs/combined_sounds_shuffled.wav \
--pretrained_weights=./logs/mar-5_training.dir/mar-5_training__epoch_10.model \
--preprocess_path=./preprocessed/ \
--out_path=./results/
