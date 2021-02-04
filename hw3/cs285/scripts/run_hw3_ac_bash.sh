
python -m cs285.scripts.run_hw3_dqn --env_name LunarLander-v3 --exp_name q1

#python -m cs285.scripts.run_hw3_actor_critic --env_name InvertedPendulum-v2 --ep_len 1000 --discount \
#    0.95 -n 100 -l 2 -s 64 -b 5000 -lr 0.01 --exp_name "q5_1_100" -ntu 1 -ngsptu 100
#
#python -m cs285.scripts.run_hw3_actor_critic --env_name InvertedPendulum-v2 --ep_len 1000 --discount \
#    0.95 -n 100 -l 2 -s 64 -b 5000 -lr 0.01 --exp_name "q5_100_1" -ntu 100 -ngsptu 1

#python -m cs285.scripts.run_hw3_actor_critic --env_name InvertedPendulum-v2 --ep_len 1000 --discount \
#    0.95 -n 100 -l 2 -s 64 -b 5000 -lr 0.01 --exp_name "q5_10_10" -ntu 10 -ngsptu 10

#python -m cs285.scripts.run_hw3_actor_critic --env_name HalfCheetah-v2 --ep_len 150 --discount 0.90 \
#  --scalar_log_freq 1 -n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 --exp_name q5_1_100 -ntu 1 -ngsptu 100
#
#python -m cs285.scripts.run_hw3_actor_critic --env_name HalfCheetah-v2 --ep_len 150 --discount 0.90 \
#  --scalar_log_freq 1 -n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 --exp_name q5_10_10 -ntu 10 -ngsptu 10
#
#python -m cs285.scripts.run_hw3_actor_critic --env_name HalfCheetah-v2 --ep_len 150 --discount 0.90 \
#  --scalar_log_freq 1 -n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 --exp_name q5_100_1 -ntu 100 -ngsptu 1