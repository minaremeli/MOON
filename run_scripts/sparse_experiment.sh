#! /bin/bash

NUM_CLIENTS=100
NUM_ROUNDS=250
LR=0.01

fracs=(0.05 0.1 0.2)

for SEED in {1..3}
do
  for f in "${fracs[@]}"
  do
  command="python main.py --num_clients $NUM_CLIENTS --num_rounds $NUM_ROUNDS --sample_fraction_fit $f --strategy moon --lr $LR --seed $SEED"
  eval "$command"
  done
done

for SEED in {1..3}
do
  for f in "${fracs[@]}"
  do
  command="python main.py --num_clients $NUM_CLIENTS --num_rounds $NUM_ROUNDS --sample_fraction_fit $f --strategy fedAvg --lr $LR --seed $SEED"
  eval "$command"
  done
done
