# cd preprocessed
# rm -rv 4-mar_combined_shuffled
# cd ..
# cd results
# rm -rv 4-mar_combined_shuffled
# cd ..

eval "$(conda shell.bash hook)"
# conda activate vq-apc
# # CUDA_VISIBLE_DEVICES=1
# python VQ-APC_downstream_try.py --exp_name=6-mar_phoneseg \
# --sound_file=./wavs/combined_sounds_shuffled.wav \
# --pretrained_weights=./logs/mar-5_training_run2.dir/mar-5_training_run2__epoch_5.model \
# --preprocess_path=./preprocessed/ \
# --out_path=./results/

conda activate wordseg
python VQ-APC_word_seg.py --wordseg_algorithm=ag \
--phoneseg_interval_dir=./results/6-mar_phoneseg/intervals/ \
--output_dir=./results/6-mar_wordseg
