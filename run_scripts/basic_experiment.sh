#! /bin/bash

NUM_CLIENTS=10
NUM_ROUNDS=100
FRAC=1.0
LR=0.001

for SEED in {1..3}
do
  command="python main.py --num_clients $NUM_CLIENTS --num_rounds $NUM_ROUNDS --sample_fraction_fit $FRAC --strategy moon --lr $LR --seed $SEED"
  eval "$command"
done

for SEED in {1..3}
do
  command="python main.py --num_clients $NUM_CLIENTS --num_rounds $NUM_ROUNDS --sample_fraction_fit $FRAC --strategy fedAvg --lr $LR --seed $SEED"
  eval "$command"
done

