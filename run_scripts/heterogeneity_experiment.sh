#! /bin/bash

NUM_CLIENTS=10
NUM_ROUNDS=100
FRAC=1.0
LR=0.001

betas=(0.1 5.0)

for SEED in {1..3}
do
  for b in "${betas[@]}"
  do
  command="python main.py --num_clients $NUM_CLIENTS --num_rounds $NUM_ROUNDS --sample_fraction_fit $FRAC --beta $b --strategy moon --lr $LR --seed $SEED"
  eval "$command"
  done
done

for SEED in {1..3}
do
  for b in "${betas[@]}"
  do
  command="python main.py --num_clients $NUM_CLIENTS --num_rounds $NUM_ROUNDS --sample_fraction_fit $FRAC --beta $b --strategy fedAvg --lr $LR --seed $SEED"
  eval "$command"
  done
done


