# --exp_name=$1 --wordseg_algo=$2 --dur_weight=$3 --max_frame=$4

eval "$(conda shell.bash hook)"

conda activate vq-apc

python VQ-APC_downstream_try.py --exp_name=$1 \
--sound_file=./wavs/1utt_numbers.wav \
--pretrained_weights=./logs/mar-30_codesize512_lr10-4_101-1000epochs.dir/mar-30_codesize512_lr10-4_101-1000epochs__epoch_1000.model \
--preprocess_path=./preprocessed/ \
--out_path=./results/ \
--codebook_size=512 \
--dur_weight=$3 \
--max_frame=$4 \

conda activate wordseg

python VQ-APC_word_seg.py --wordseg_algorithm=$2 \
--phoneseg_interval_dir=./results/$1/intervals/ \
--output_dir=./results/$1_$2

conda activate vq-apc

python VQ-APC_split_wav.py --preprocessed_wav_path=./preprocessed/$1/ \
--wordseg_interval_dir=./results/$1_$2/intervals/ \
--output_dir=./results/$1_$2

python VQ-APC_stt_recog.py --wav_path=./results/$1_$2/wavs \
--output_dir=./results/$1_$2
