CUDA_VISIBLE_DEVICES=1 python lbg.py --batch_size 1000  \
                      --librispeech_home ../VQ-APC/librispeech_data/preprocessed \
                      --train_partition train-clean-360 \
                      --train_sampling 1. \
                      --val_partition dev-clean \
                      --val_sampling 1. \
                      --codebook_size 512 \
                      --sound_file ./wavs/1utt_numbers_clean.wav \
                      --store_path ./results/ \
                      --exp_name $1
