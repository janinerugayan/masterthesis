from pydub import AudioSegment
import argparse
import os
from pathlib import Path


# load wordseg intervals
wordseg_interval_dict = {}
wordseg_file = 'lived_there_for_a_while.txt'
fn = 'lived_there_for_a_while'
wordseg_interval_dict[fn] = []
f = open(wordseg_file, 'r')
for line in f:
    start, end = line.split()
    start = float(start) * 1000
    end = float(end) * 1000
    wordseg_interval_dict[fn].append((start, end))
f.close()

utterances = wordseg_interval_dict.keys()

# wav_path = args.preprocessed_wav_path
wav_path = './s05/s0501a/s0501a.wav'  # for reusing same wav segments
export_dir = './'
for utt_key in utterances:
    wav_name = utt_key
    wav_original = AudioSegment.from_wav(wav_path)
    total_len = len(wav_original)
    count = 0
    for (word_start, word_end) in wordseg_interval_dict[utt_key]:
        print(f'word start: {word_start}')
        print(f'word end: {word_end}')
        wav_segment = wav_original[word_start : word_end]
        wav_segment.export(export_dir + fn + '.wav', format='wav')
        count += 1
