import numpy as np


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) == 1 or len(obs.shape) == 3: # 1dim obs or 3 dims obs
            observation = obs[None]
        else:
            observation = obs
        
        ## TODO return the action that maxinmizes the Q-value 
        # at the current observation as the output
        actions = self.critic.qa_values(observation)
        action = np.argmax(actions, axis=-1)

        return action[0]