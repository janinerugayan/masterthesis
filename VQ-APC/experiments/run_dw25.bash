for i in {1..3}; do
  bash VQ_spolacq_size128_mod.bash $i ag 25 may-14_size128epoch2000
done

for i in {1..3}; do
  bash VQ_spolacq_size512_mod.bash $i ag 25 may-14_size512epoch2000
done
