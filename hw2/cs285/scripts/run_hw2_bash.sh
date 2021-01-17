#!/usr/bin/env bash

for b in 50000;
do
  for r in 2e-2;
  do
      echo "b=$b, and r=$r"

      python -m cs285.scripts.run_hw2 --env_name HalfCheetah-v2 --ep_len 150 \
        --discount 0.95 -n 100 -l 2 -s 32 -b $b -lr $r -rtg --nn_baseline \
        --exp_name "q4_search_b${b}_lr${r}_rtg_nnbaseline"
  done
done

