from pydub import AudioSegment
import argparse
import os
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('--preprocessed_wav_path', type=str)
parser.add_argument('--wordseg_interval_dir', type=str)
parser.add_argument('--output_dir', type=str)
args = parser.parse_args()


# load wordseg intervals
wordseg_interval_dict = {}
wordseg_dir = args.wordseg_interval_dir
print('Reading: {}'.format(wordseg_dir))
for file in os.listdir(wordseg_dir):
    fn = Path(file).stem
    wordseg_interval_dict[fn] = []
    f = open(wordseg_dir + file, 'r')
    for line in f:
        if len(line) == 0:
            wordseg_interval_dict.pop(fn)
            continue
        start, end, label = line.split()
        start = int(start)
        end = int(end)
        wordseg_interval_dict[fn].append((start, end))
    f.close()

utterances = wordseg_interval_dict.keys()

# segment the combined wav file
split_word = '_logmel'
# wav_path = args.preprocessed_wav_path
wav_path = './preprocessed/apr-11_kaldi_1uttnumbers/'  # for reusing same wav segments
export_dir = args.output_dir + '/wavs/'
os.makedirs(export_dir)
for utt_key in utterances:
    wav_name = utt_key.split(split_word)[0]
    wav_original = AudioSegment.from_wav(wav_path + wav_name + '.wav')
    total_len = len(wav_original)
    count = 0
    for (word_start, word_end) in wordseg_interval_dict[utt_key]:
        word_start = word_start * 10
        word_end = word_end * 10
        wav_segment = wav_original[word_start : word_end]
        wav_segment.export(export_dir + wav_name + '_word_' + str(count) + '.wav', format='wav')
        count += 1
