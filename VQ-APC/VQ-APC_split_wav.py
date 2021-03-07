from pydub import AudioSegment
import argparse
import os


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
wav_path = args.preprocessed_wav_path
export_dir = args.output_dir + '/wavs/'
os.makedirs(export_dir)
for utt_key in utterances:
    wav_original = AudioSegment.from_wav(wav_path + utt_key + '.wav')
    total_len = len(wav_original)
    count = 0
    for (word_start, word_end) in wordseg_interval_dict[utt_key]:
        wav_segment = wav_original[word_start : word_end]
        wav_segment.export(export_dir + 'word' + str(count) + '.wav', format='wav')
        count += 1