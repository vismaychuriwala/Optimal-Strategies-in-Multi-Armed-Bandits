import numpy as np 
import numpy.random as random

np.seterr(divide='ignore', invalid='ignore')

def KL(p, q):
	if p == 1:
		return p*np.log(p/q)
	elif p == 0:
		return (1 - p) * np.log((1 - p) / (1 - q))
	else:
		return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

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
		rhs = (np.log(time_steps) + 3 * np.log(np.log(time_steps))) / pulls[x]
		ucb_arms[x] = solve_q(rhs, p_a)
	return ucb_arms



def kl_ucb(num_bandits, bandit_probs, horizon, seed):

	random.seed(seed)

	rewards = np.zeros((num_bandits, horizon), dtype=int)

	for y in range(0,num_bandits): 
		s = np.random.binomial(1, bandit_probs[y], horizon)
		rewards[y, :] = s 

	cum_reward = 0
	curr_arm = 0
	curr_reward = 0
	pulls = np.zeros(num_bandits, dtype=int)
	arm_rewards = np.zeros(num_bandits, dtype=int)
	ucb_arms = np.zeros(num_bandits, dtype=float)

	for x in range(0, min(num_bandits,horizon)):
		curr_arm = x
		curr_reward = rewards[curr_arm, pulls[curr_arm]]
		pulls[curr_arm] += 1
		cum_reward += curr_reward
		arm_rewards[curr_arm] += curr_reward
		
	if horizon > num_bandits:

		for y in range(num_bandits,horizon):
			
			ucb_arms = ucb_func(pulls, arm_rewards, y, num_bandits)
			max_ucb = np.amax(ucb_arms)
			indices = np.where(ucb_arms == max_ucb)
			
			curr_arm = np.amax(indices)
			curr_reward = rewards[curr_arm, pulls[curr_arm]]
			pulls[curr_arm] += 1
			cum_reward += curr_reward
			arm_rewards[curr_arm] += curr_reward
	return cum_reward
