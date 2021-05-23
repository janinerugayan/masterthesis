# code size 128

# using wordseg-ag
# for i in {1..5}; do
#   bash VQ_spolacq_size128_mod.bash $i ag 25 may-23_size128epoch1000_numshuffledver2
# done

# using wordseg-tp
# bash VQ_spolacq_size128_mod.bash 1 tp 25 may-23_size128epoch1000_numshuffledver2


# code size 256

# using wordseg-ag
for i in {1..5}; do
  bash VQ_spolacq_size256_mod.bash $i ag 25 may-23_size256epoch1000_numshuffledver2
done

# using wordseg-tp
bash VQ_spolacq_size256_mod.bash 1 tp 25 may-23_size256epoch1000_numshuffledver2


# code size 512

# using wordseg-ag
for i in {1..5}; do
  bash VQ_spolacq_size512_mod.bash $i ag 25 may-23_size512epoch1000_numshuffledver2
done

# using wordseg-tp
bash VQ_spolacq_size512_mod.bash 1 tp 25 may-23_size512epoch1000_numshuffledver2
