CUDA_VISIBLE_DEVICES=1 python train_vqapc.py --rnn_num_layers 3 \
                      --rnn_hidden_size 512 \
                      --rnn_dropout 0.1 \
                      --rnn_residual \
                      --codebook_size 128 \
                      --code_dim 512 \
                      --gumbel_temperature 0.5 \
                      --apply_VQ 0 0 1 \
                      --optimizer adam \
                      --batch_size 32  \
                      --learning_rate 0.00001 \
                      --epochs 2000 \
                      --n_future 5 \
                      --librispeech_home ./librispeech_data/preprocessed \
                      --train_partition train-clean-360 \
                      --train_sampling 1. \
                      --val_partition dev-clean \
                      --val_sampling 1. \
                      --exp_name apr-18_codesize128_2000epochs_lr10-5 \
                      --store_path ./logs \
                      --checkpoint_model ./logs/mar-31_codesize128_1000epochs.dir/mar-31_codesize128_1000epochs__epoch_1000.model \
                      --checkpoint_epoch 1000

# default values:
# learning_rate = 0.0001
