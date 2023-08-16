#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import time
# Students will submit their files with their team-name.py 
# Student have to use the Team as their parent class


# In[170]:


import numpy as np 
import sys 
import numpy.random as random 
import os
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


# In[252]:


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
    self.wicketsdown+=wicket
    
#     print(self.lastaction)
    self.arm_wickets[self.lastaction]+=wicket
    self.arm_runs[self.lastaction]+=runs_scored
#     print("*******",self.lastaction)
    
    reward=np.multiply((1-np.divide(self.arm_wickets,self.pulls)),np.divide(self.arm_runs,self.pulls))/6
#     reward=self.arm_runs
    
    for elem in range(6):
        if np.isnan(reward[elem]):
            reward[elem]=0
#     print(reward)        
    if self.balls in range(7):
        action=self.balls-1
       
        self.pulls[action] += 1
        self.cum_reward += reward[action]
        self.arm_rewards[action] += reward[action]
#         print(self.arm_rewards)
        
    
    else:
#         print("**",self.pulls,reward,self.balls)
        ucb_arms = ucb_func(self.pulls, reward, self.balls, 6)
        max_ucb = np.amax(ucb_arms)
        indices = np.where(ucb_arms == max_ucb)
        
        action = np.amax(indices)
        
#         self.curr_reward = rewards[curr_arm, pulls[curr_arm]]
        self.pulls[action] += 1
        self.cum_reward += reward[action]
        self.arm_rewards[action] += reward[action]
#     action = np.random.randint(0,6)
#     print(self.pulls)
    self.lastaction=action
    return action


# In[255]:


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
            

#         print(self.arm_rewards)
        
    
        else:
#         print("**",self.pulls,reward,self.balls)
            ucb_arms = ucb_func(self.pulls, self.arm_rewards, self.balls, 6)
            max_ucb = np.amax(ucb_arms)
            indices = np.where(ucb_arms == max_ucb)

            action = np.amax(indices)

    #         self.curr_reward = rewards[curr_arm, pulls[curr_arm]]
            self.pulls[action] += 1
        self.lastaction=action
        return action
        
#   
    


# In[245]:


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
            

#         print(self.arm_rewards)
        
    
        else:
            ucb_arms=np.zeros(6)
                    
            for i in range(6):
                ucb_arms[i] = (self.arm_runs[i]/self.pulls[i])+(2*np.log(self.balls)/self.pulls[i])**0.5
            max_ucb = np.amax(ucb_arms)
            indices = np.where(ucb_arms == max_ucb)

            action = np.amax(indices)

    #         self.curr_reward = rewards[curr_arm, pulls[curr_arm]]
            self.pulls[action] += 1
        self.lastaction=action
        return action
        
#   
    


# In[256]:


class Environment:
  def __init__(self,num_balls,agent):
    self.num_balls = num_balls
    self.agent = agent
    self.__run_time = 0
    self.__total_runs = 0
    self.__total_wickets = 0
    self.__runs_scored = 0
    self.__start_time = 0
    self.__end_time = 0
    self.__regret_w = 0
    self.__regret_s = 0
    self.__wicket = 0
    self.__regret_rho = 0
    self.__p_out =np.array([0.001,0.01,0.02,0.03,0.1,0.3])
    self.__p_run =np.array([1,0.9,0.85,0.8,0.75,0.7])
    self.__action_runs_map = np.array([0,1,2,3,4,6])
    self.__s = (1-self.__p_out)*self.__p_run*self.__action_runs_map
    self.__rho = self.__s/self.__p_out


  def __get_action(self):
    self.__start_time      = time. time()
    action          = self.agent.get_action(self.__wicket,self.__runs_scored)
    self.__end_time        = time. time()
    self.__run_time   = self.__run_time + self.__end_time - self.__start_time
    return action


  def __get_outcome(self, action):
    pout = self.__p_out[action]
    prun= self.__p_run[action]
    wicket = np.random.choice(2,1,p=[1-pout,pout])[0]
    runs = 0
    if(wicket==0):
      runs = self.__action_runs_map[action]*np.random.choice(2,1,p=[1-prun,prun])[0]
    return wicket, runs


  def innings(self):
    self.__total_runs = 0
    self.__total_wickets = 0
    self.__runs_scored = 0

    for ball in range(self.num_balls):
      action = self.__get_action()
      self.__wicket, self.__runs_scored   = self.__get_outcome(action)
      self.__total_runs     = self.__total_runs + self.__runs_scored
      self.__total_wickets  = self.__total_wickets + self.__wicket
      self.__regret_w       = self.__regret_w+ (self.__p_out[action]-np.min(self.__p_out))
      self.__regret_s       = self.__regret_s+ (np.max(self.__s) - self.__s[action])
      self.__regret_rho       = self.__regret_rho+ (np.max(self.__rho)-self.__rho[action])
    return self.__regret_w,self.__regret_s,self.__regret_rho, self.__total_runs, self.__total_wickets, self.__run_time


# In[257]:


agent = Agent()
environment = Environment(1000,agent)
regret_w,regret_s,reger_rho,total_runs,total_wickets,run_time = environment.innings()


# In[258]:


for i in range(10):
    agent = Agent()
    environment = Environment(1000,agent)
    regret_w,regret_s,reger_rho,total_runs,total_wickets,run_time = environment.innings()
    print(regret_w,regret_s,reger_rho,total_runs,total_wickets,run_time)


# In[125]:


for elem in np.divide([0,1,2],[0,3,5]):
    if np.isnan(elem):
        print(0)


# In[120]:


0/0


# In[ ]:




