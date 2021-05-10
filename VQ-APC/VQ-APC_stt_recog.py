import speech_recognition as sr
from os import walk
import argparse, pickle

parser = argparse.ArgumentParser()
parser.add_argument('--wav_path',   type=str)
parser.add_argument('--output_dir', type=str)
args = parser.parse_args()

# read and recognize segmented wav files
wav_path = args.wav_path
output_dir = args.output_dir
__, __, filenames = next(walk(wav_path), (None, None, []))
f = open(output_dir + '/stt_results.txt', 'a')
r = sr.Recognizer()

recog_dict = {}

recog_dict['num_words'] = 0
recog_dict['zero'] = 0
recog_dict['one'] = 0
recog_dict['two'] = 0
recog_dict['three'] = 0
recog_dict['four'] = 0
recog_dict['five'] = 0
recog_dict['six'] = 0
recog_dict['seven'] = 0
recog_dict['eight'] = 0
recog_dict['nine'] = 0

# for action words:
# recog_dict['num_words'] = 0
# recog_dict['up'] = 0
# recog_dict['down'] = 0
# recog_dict['left'] = 0
# recog_dict['right'] = 0
# recog_dict['backward'] = 0
# recog_dict['forward'] = 0


for filename in filenames:

    if '.wav' not in filename:
        print(filename)
        continue

    with sr.AudioFile(wav_path + '/' + filename) as source:
        audio = r.record(source)

    print(f'Google stt recog on: {filename}')

    for j in range(5):
        try:
            recog_result = r.recognize_google(audio)
            f.write(filename + ' ' + recog_result + '\n')
            recog_dict['num_words'] += 1

            if recog_result == "0":
                recog_dict['zero'] += 1
            elif recog_result == "1":
                recog_dict['one'] += 1
            elif recog_result == "2":
                recog_dict['two'] += 1
            elif recog_result == "3":
                recog_dict['three'] += 1
            elif recog_result == "4":
                recog_dict['four'] += 1
            elif recog_result == "5":
                recog_dict['five'] += 1
            elif recog_result == "6":
                recog_dict['six'] += 1
            elif recog_result == "7":
                recog_dict['seven'] += 1
            elif recog_result == "8":
                recog_dict['eight'] += 1
            elif recog_result == "9":
                recog_dict['nine'] += 1

            # for trying with direction action words:
            # if recog_result == "up":
            #     recog_dict['up'] += 1
            # elif recog_result == "down":
            #     recog_dict['down'] += 1
            # elif recog_result == "left":
            #     recog_dict['left'] += 1
            # elif recog_result == "right":
            #     recog_dict['right'] += 1
            # elif recog_result == "forward":
            #     recog_dict['forward'] += 1
            # elif recog_result == "backward":
            #     recog_dict['backward'] += 1

            break

        except sr.UnknownValueError:
            pass

        if 4 == j:
            f.write(filename + ' ' + 'Google Speech Recognition could not understand audio' + '\n')
            recog_dict['num_words'] += 1

f.close()

print(recog_dict)

f = open(output_dir + '/recog_results.txt', 'a')
f.write(str(recog_dict))
f.close()

with open(output_dir + '/recog_results_dict.pkl', 'wb') as handle:
    pickle.dump(recog_dict, handle)
