# CUDA_VISIBLE_DEVICES=1 python train_vqapc.py --rnn_num_layers 3 \
#                       --rnn_hidden_size 512 \
#                       --rnn_dropout 0.1 \
#                       --rnn_residual \
#                       --codebook_size 128 \
#                       --code_dim 512 \
#                       --gumbel_temperature 0.5 \
#                       --apply_VQ 0 0 1 \
#                       --optimizer adam \
#                       --batch_size 32  \
#                       --learning_rate 0.0001 \
#                       --epochs 1 \
#                       --n_future 5 \
#                       --librispeech_home ./librispeech_data/preprocessed \
#                       --train_partition train-clean-360 \
#                       --train_sampling 1. \
#                       --val_partition dev-clean \
#                       --val_sampling 1. \
#                       --exp_name feb-4_vqextract \
#                       --store_path ./logs

cd proprocessed/
rm -v *.wav *.npy
cd ..

CUDA_VISIBLE_DEVICES=1 python VQ-APC_downstream_try.py --exp_name=combined_sounds_shuffled \
--sound_file=./wavs/combined_sounds_shuffled.wav \
--pretrained_weights=./logs/feb-4_vqextract.dir/feb-4_vqextract__epoch_1.model \
--pretrained_VQ=./logs/feb-4_vqextract.dir/feb-4_vqextract-VQlayers__epoch_1.model
