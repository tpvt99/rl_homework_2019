import gym
env = gym.make('Humanoid-v2')
from cs285.policies.loaded_gaussian_policy import Loaded_Gaussian_Policy
expert_policy = Loaded_Gaussian_Policy('./policies/experts/Humanoid.pkl')
for i_episode in range(20):
    observation = env.reset()
    for t in range(1000):
        env.render()
        action = expert_policy.get_action(observation)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()