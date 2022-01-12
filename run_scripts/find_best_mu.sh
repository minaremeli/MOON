#! /bin/bash

NUM_CLIENTS=10
NUM_ROUNDS=100
FRAC=1.0
LR=0.001

mus=(0.1 1.0 3.0 7.0)

for SEED in {0..0}
  do
  for mu in "${mus[@]}"
  do
    command="python main.py --num_clients $NUM_CLIENTS --num_rounds $NUM_ROUNDS --sample_fraction_fit $FRAC --strategy moon --lr $LR --mu $mu --seed $SEED"
    eval "$command"
  done
done
