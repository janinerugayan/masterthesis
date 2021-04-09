CUDA_VISIBLE_DEVICES=1 python train_LBG.py --batch_size 32  \
                      --librispeech_home ../VQ-APC/librispeech_data/preprocessed \
                      --train_partition train-clean-360 \
                      --train_sampling 1. \
                      --val_partition dev-clean \
                      --val_sampling 1. 
                      # --exp_name mar-31_codesize128_1000epochs \
                      # --store_path ./logs
