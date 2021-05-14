# code size 128

# using wordseg-ag
for i in {1..5}; do
  bash VQ_spolacq_size128_mod.bash $i ag 25 may-15_size128epoch2000
done

for i in {1..5}; do
  bash VQ_spolacq_size128_mod.bash $i ag 36 may-15_size128epoch2000
done

# using wordseg-tp
for i in {1..5}; do
  bash VQ_spolacq_size128_mod.bash $i tp 25 may-15_size128epoch2000
done

for i in {1..5}; do
  bash VQ_spolacq_size128_mod.bash $i tp 36 may-15_size128epoch2000
done


# code size 512

# using wordseg-ag
for i in {1..5}; do
  bash VQ_spolacq_size512_mod.bash $i ag 25 may-15_size512epoch2000
done

for i in {1..5}; do
  bash VQ_spolacq_size512_mod.bash $i ag 36 may-15_size512epoch2000
done

# using wordseg-tp
for i in {1..5}; do
  bash VQ_spolacq_size512_mod.bash $i tp 25 may-15_size512epoch2000
done

for i in {1..5}; do
  bash VQ_spolacq_size512_mod.bash $i tp 36 may-15_size512epoch2000
done
