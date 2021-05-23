
# -----------------------------------------------------------------
#   word segmentation from VQ-seg code of Herman Kamper
# -----------------------------------------------------------------

from pathlib import Path
from tqdm import tqdm
import os
import argparse

from eval_segmentation import get_intervals_from_dir
import wordseg_algorithms

#for debugging
# from IPython import embed



parser = argparse.ArgumentParser()
parser.add_argument('--wordseg_algorithm',   type=str)
parser.add_argument('--phoneseg_interval_dir',   type=str)
parser.add_argument('--output_dir',  type=str)
args = parser.parse_args()

# wordseg algorithm configurations
kwargs = {}
if args.wordseg_algorithm == 'dpseg':
    kwargs = {'nfolds': 1, 'args': "--randseed 1"}
elif args.wordseg_algorithm == 'tp':
    kwargs = {'threshold': "relative"}
elif args.wordseg_algorithm == 'ag':
    # kwargs = {'nruns': 4, 'njobs': 3, 'args': "-n 100"}
    kwargs = {'nruns': 8, 'njobs': 3, 'args': "-n 2000"}  # default according to WordSeg paper nruns: 8, args: "-n 2000"

# Algorithm
segment_func = getattr(wordseg_algorithms, args.wordseg_algorithm)

# Read phone intervals
phoneseg_interval_dict = {}
phoneseg_dir = args.phoneseg_interval_dir
print("Reading: {}".format(phoneseg_dir))
for file in os.listdir(phoneseg_dir):
    if file.endswith('.txt'):
        fn = Path(file).stem
        phoneseg_interval_dict[fn] = []
        file = (Path(phoneseg_dir)/fn).with_suffix(".txt")
        for line in file.read_text().strip().split("\n"):
            if len(line) == 0:
                phoneseg_interval_dict.pop(fn)
                continue
            start, end, label = line.split()
            start = int(start)
            end = int(end)
            phoneseg_interval_dict[fn].append((start, end, label))
        # f = open(phoneseg_dir + file, 'r')
        # for line in f:
        #     if len(line) == 0:
        #         phoneseg_interval_dict.pop(fn)
        #         continue
        #     start, end, label = line.split()
        #     start = int(start)
        #     end = int(end)
        #     phoneseg_interval_dict[fn].append((start, end, label))
        # f.close()

utterances = phoneseg_interval_dict.keys()

# Segmentation
print("Segmenting:")
prepared_text = []
for utt_key in utterances:
    prepared_text.append(
        " ".join([i[2] + "_" for i in phoneseg_interval_dict[utt_key]])
        )

# Write prepared text
output_dir = args.output_dir + '/prepared_text/'
os.makedirs(output_dir)
f = open(output_dir + 'prepared_text.txt', 'w')
f.write(str(prepared_text))
f.close()

word_segmentation = segment_func(prepared_text, **kwargs)
# print(prepared_text[:10])
wordseg_interval_dict = {}
for i_utt, utt_key in tqdm(enumerate(utterances)):
    words_segmented = word_segmentation[i_utt].split(" ")
    word_start = 0
    word_label = ""
    i_word = 0
    wordseg_interval_dict[utt_key] = []
    for (phone_start,
            phone_end, phone_label) in phoneseg_interval_dict[utt_key]:
        word_label += phone_label + "_"
        if words_segmented[i_word] == word_label:
            wordseg_interval_dict[utt_key].append((
                word_start, phone_end, word_label
                ))
            word_label = ""
            word_start = phone_end
            i_word += 1

# Write intervals
output_dir = args.output_dir + '/intervals/'
os.makedirs(output_dir)
print("Writing to: {}".format(output_dir))
for utt_key in wordseg_interval_dict:
    with open(output_dir + utt_key + '_wordseg.txt', 'w') as f:
        for start, end, label in wordseg_interval_dict[utt_key]:
            f.write("{:d} {:d} {}\n".format(start, end, label))
