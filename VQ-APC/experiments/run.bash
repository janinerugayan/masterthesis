# code size 256
# using wordseg-ag
# for i in {1..5}; do
#   bash VQ_spolacq_size256.bash $i ag 36 june-2_size256epoch1600
# done
# using wordseg-tp
# bash VQ_spolacq_size256.bash 1 tp 36 june-2_size256epoch1600


# code size 128
# using wordseg-ag
for i in {1..5}; do
  bash VQ_spolacq_size128.bash $i ag 36 june-2_size128epoch1600
done
# using wordseg-tp
bash VQ_spolacq_size128.bash 1 tp 36 june-2_size128epoch1600



# code size 512
# using wordseg-ag
for i in {1..5}; do
  bash VQ_spolacq_size512.bash $i ag 36 june-2_size512epoch1600
done
# using wordseg-tp
bash VQ_spolacq_size512.bash 1 tp 36 june-2_size512epoch1600
