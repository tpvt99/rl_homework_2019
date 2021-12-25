#!/usr/bin/env bash

# Experiment 1

#python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \
#-dsa --exp_name q1_sb_no_rtg_dsa

#python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \
#-rtg -dsa --exp_name q1_sb_rtg_dsa

#python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \
#-rtg --exp_name q1_sb_rtg_na

#python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 \
#-dsa --exp_name q1_lb_no_rtg_dsa

#python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 \
#-rtg -dsa --exp_name q1_lb_rtg_dsa

#python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 \
#-rtg --exp_name q1_lb_rtg_na

# -------------------------------------

# Experiment 2
#for b in 500;
#do
#  for r in 5e-2 1e-3 5e-3 1e-4 5e-4;
#  do
#      echo "b=$b, and r=$r"
#      python -m cs285.scripts.run_hw2 --env_name InvertedPendulum-v2 \
#        --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b $b -lr $r -rtg \
#        --exp_name q2_b${b}_lr${r} --video_log_freq -1
#  done
#done

 # ----------------------------------

 # Experiment 4

#for b in 50000;
#do
#  for r in 2e-2;
#  do
#      echo "b=$b, and r=$r"
#
#      python -m cs285.scripts.run_hw2 --env_name HalfCheetah-v2 --ep_len 150 \
#        --discount 0.95 -n 100 -l 2 -s 32 -b $b -lr $r -rtg --nn_baseline \
#        --exp_name "q4_search_b${b}_lr${r}_rtg_nnbaseline"
#  done
#done

python -m cs285.scripts.run_hw2 --env_name HalfCheetah-v2 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b 10000 -lr 0.02 \
--exp_name q4_b10000_r0_02 --video_log_freq -1

python -m cs285.scripts.run_hw2 --env_name HalfCheetah-v2 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b 10000 -lr 0.02 -rtg \
--exp_name q4_b10000_r0_02_rtg --video_log_freq -1

python -m cs285.scripts.run_hw2 --env_name HalfCheetah-v2 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b 10000 -lr 0.02 --nn_baseline \
--exp_name q4_b10000_r0_02_nnbaseline --video_log_freq -1

python -m cs285.scripts.run_hw2 --env_name HalfCheetah-v2 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 --b 10000 -lr 0.02 -rtg --nn_baseline \
--exp_name q4_b10000_r0_02_rtg_nnbaseline --video_log_freq -1
