import numpy as np 

np.seterr(divide='ignore', invalid='ignore')

def KL(p, q):
	if p == 1:
		return p*np.log(p/q)
	elif p == 0:
		return (1-p)*np.log((1-p)/(1-q))
	else:
		return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))

def solve_q(rhs, p_a):
	if p_a == 1:
		return 1 
	q = np.arange(p_a, 1, 0.01)
	lhs = []
	for el in q:
		lhs.append(KL(p_a, el))
	lhs_array = np.array(lhs)
	lhs_rhs = lhs_array - rhs
	lhs_rhs[lhs_rhs <= 0] = np.inf
	min_index = lhs_rhs.argmin()
	return q[min_index]


def ucb_func(pulls, arm_rewards, time_steps, num_bandits):
	ucb_arms = np.zeros(num_bandits, dtype=float)
	for x in range(0,num_bandits):
		p_a = arm_rewards[x]/pulls[x]
		rhs = (np.log(time_steps) + 3*np.log(np.log(time_steps)))/pulls[x]
		ucb_arms[x] = solve_q(rhs, p_a)
	# print ucb_arms
	return ucb_arms

class Agent:
  def __init__(self):
    self.balls=0
    self.score=0
    self.wicketsdown=0
    self.pulls = np.zeros(6, dtype=int)
    self.arm_rewards = np.zeros(6, dtype=int)
    self.ucb_arms = np.zeros(6, dtype=float)
    self.curr_reward = 0
    self.cum_reward=0

    pass
    
  def get_action(self,wicket,runs_scored):
    self.balls+=1
    self.wicketsdown+=wicket
    
    if self.balls in range(7):
        action=self.balls-1
       
        self.pulls[action] += 1
        self.cum_reward += 1-wicket
        self.arm_rewards[action] += 1-wicket
        return action
    
    else:
        ucb_arms = ucb_func(self.pulls, self.arm_rewards, self.balls, 6)
        max_ucb = np.amax(ucb_arms)
        indices = np.where(ucb_arms == max_ucb)

        action = np.amax(indices)
        self.pulls[action] += 1
        self.cum_reward += 1-wicket
        self.arm_rewards[action] += 1-wicket

    return action
