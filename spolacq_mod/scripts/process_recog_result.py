import os
import argparse, pickle

parser = argparse.ArgumentParser()
parser.add_argument('--data_name',   type=str)
args = parser.parse_args()

recog_result_path = "../exp/segmented_wavs_" + args.data_name + "/stt_results.txt"
exp_path = "../exp/pkls/recog_results_dict.pkl"

res_dict = {}

f = open(recog_result_path, "r")
lines = f.readlines()

res_dict['num_words'] = len(lines)
res_dict['zero'] = 0
res_dict['one'] = 0
res_dict['two'] = 0
res_dict['three'] = 0
res_dict['four'] = 0
res_dict['five'] = 0
res_dict['six'] = 0
res_dict['seven'] = 0
res_dict['eight'] = 0
res_dict['nine'] = 0

for line in lines:
    line_words = line.split(" ")
    if "0" in line_words[1]:
        res_dict['zero'] += 1
    elif "1" in line_words[1]:
        res_dict['one'] += 1
    elif "2" in line_words[1]:
        res_dict['two'] += 1
    elif "3" in line_words[1]:
        res_dict['three'] += 1
    elif "4" in line_words[1]:
        res_dict['four'] += 1
    elif "5" in line_words[1]:
        res_dict['five'] += 1
    elif "6" in line_words[1]:
        res_dict['six'] += 1
    elif "7" in line_words[1]:
        res_dict['seven'] += 1
    elif "8" in line_words[1]:
        res_dict['eight'] += 1
    elif "9" in line_words[1]:
        res_dict['nine'] += 1

print(res_dict)

# added recog result record write to text file
f = open('../exp/process_recog_result.txt', 'a')
f.write(str(res_dict))
f.close()

with open(exp_path, 'wb') as handle:
    pickle.dump(res_dict, handle)
