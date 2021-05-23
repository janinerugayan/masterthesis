# code size 128

# using wordseg-ag
for i in {1..5}; do
  bash VQ_spolacq_size128.bash $i ag 36 may-24_size128epoch1000
done

# using wordseg-tp
bash VQ_spolacq_size128.bash 1 tp 36 may-24_size128epoch1000


# code size 256

# using wordseg-ag
for i in {1..5}; do
  bash VQ_spolacq_size256.bash $i ag 36 may-24_size256epoch1000
done

# using wordseg-tp
bash VQ_spolacq_size256.bash 1 tp 36 may-24_size256epoch1000


# code size 512

# using wordseg-ag
for i in {1..5}; do
  bash VQ_spolacq_size512.bash $i ag 36 may-24_size512epoch1000
done

# using wordseg-tp
bash VQ_spolacq_size512.bash 1 tp 36 may-24_size512epoch1000
