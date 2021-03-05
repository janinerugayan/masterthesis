# cd preprocessed
# rm -rv 4-mar_combined_shuffled
# cd ..
# cd results
# rm -rv 4-mar_combined_shuffled
# cd ..

# CUDA_VISIBLE_DEVICES=1 python VQ-APC_downstream_try.py --exp_name=5-mar_phoneseg \
# --sound_file=./wavs/combined_sounds_shuffled.wav \
# --pretrained_weights=./logs/mar-5_training_run2.dir/mar-5_training_run2__epoch_5.model \
# --preprocess_path=./preprocessed/ \
# --out_path=./results/

# conda activate wordseg
python VQ-APC_word_seg.py --exp_name=5-mar_wordseg \
--wordseg_algorithm=ag \
--phoneseg_interval_dir=./results/5-mar_phoneseg/intervals \
--output_dir=./results/5-mar_wordseg
