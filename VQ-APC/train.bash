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
                      --learning_rate 0.0001 \
                      --epochs 100 \
                      --n_future 5 \
                      --librispeech_home ./librispeech_data/preprocessed \
                      --train_partition train-clean-360 \
                      --train_sampling 1. \
                      --val_partition dev-clean \
                      --val_sampling 1. \
                      --exp_name mar-9_training_10epochs \
                      --store_path ./logs
