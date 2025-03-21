import numpy as np

class Agent:
    def __init__(self):
        self.balls=0
        self.score=0
        self.wicketsdown=0
        self.pulls = np.zeros(6, dtype=int)
        self.arm_rewards = np.zeros(6, dtype=int)
        self.arm_wickets = np.zeros(6, dtype=int)
        self.arm_runs=np.zeros(6,dtype=int)
        self.ucb_arms = np.zeros(6, dtype=float)
        self.curr_reward = 0
        self.cum_reward=0
        self.lastaction=0

        pass

    def get_action(self,wicket,runs_scored):
        self.balls+=1
        self.arm_runs[self.lastaction]+=runs_scored
        self.arm_rewards[self.lastaction]+=runs_scored/6

        if self.balls in range(7):
            action=self.balls-1

            self.pulls[action] += 1        
    
        else:
            ucb_arms = ucb_func(self.pulls, self.arm_rewards, self.balls, 6)
            max_ucb = np.amax(ucb_arms)
            indices = np.where(ucb_arms == max_ucb)
            action = np.amax(indices)
            self.pulls[action] += 1

        self.lastaction=action
        return action