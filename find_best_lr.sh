#! /bin/bash

NUM_CLIENTS=10
NUM_ROUNDS=100
FRAC=1.0

lrs=(0.01 0.005 0.001 0.0005 0.0001)

for SEED in {0..3}
  do
  for lr in "${lrs[@]}"
  do
    command="python main.py --num_clients $NUM_CLIENTS --num_rounds $NUM_ROUNDS --sample_fraction_fit $FRAC --strategy $1 --lr $lr --seed $SEED"
    eval "$command"
  done
done
