#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import time


# In[ ]:





# In[ ]:


class Agent:
    def __init__(self):
        self.balls=0
        self.score=0
        self.wicketsdown=0
        self.pulls = np.zeros(6, dtype=int)
        self.arm_rewards = np.zeros(6, dtype=int)
        self.arm_runs = np.zeros(6, dtype=int)
        self.arm_wickets = np.zeros(6, dtype=int)
        self.lastaction=0
        self.ucb_arms = np.zeros(6, dtype=float)
        self.curr_reward = 0
        self.cum_reward=0
        self.s=[0,1,2,3,4,6]
        self.count=5

    pass
    
    def get_action(self,wicket,runs_scored):
        self.balls+=1
        self.wicketsdown+=wicket
        self.arm_wickets[self.lastaction]+=wicket
        self.arm_runs[self.lastaction]+=runs_scored
        
        while (len(self.s)>>1):
            if (self.count>=0):
#                 print(self.count)
                action= self.count
                
                self.pulls[action] += 1
                self.balls+=1
                self.count+=-1
                #print(action)
            
                self.lastaction=action
                return action
            else:
#                 print("**")
                delta=0.3
                count=len(self.s)-1
                mu=np.multiply((1-np.divide(self.arm_wickets,self.pulls)),np.divide(self.arm_runs,self.pulls))
                ct=np.divide(self.arm_wickets,self.pulls)
                mc=np.divide(mu,ct)
                for elem in range(len(mc)):
                    if np.isnan(mc[elem]):
                        mc[elem]=0
                    if np.isinf(mc[elem]):
                        mc[elem]=0
                mcmax=max(mc)
                istar=list(np.where(mc == mcmax)[0])[0]
                
                mustar=mu[istar]
                ctstar=ct[istar]
                #print(self.pulls)
                bt=(np.log(4*6/delta)/(2*self.balls))**0.5
#                 print(bt)
                if mustar-2*bt<=0:
                    continue
                n=self.s
                del self.s[istar]
                m=self.s
                self.s=n
                
                for arm in range(len(m)):
                    if (ct[arm]-2*bt<=0):
                        continue
                        
                    elif ((mustar-2*bt)/(ct+2*bt)>(mu[arm]+2*bt)/(ct[arm]-2*bt)):
                        del self.s[arm]
                        del self.arm_wickets[arm]
                        del self.pulls
                        del self.arm_runs
                 
        self.lastaction=self.s
        return self.s
        self.lastaction=action
        return action


# In[ ]:


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


# In[ ]:


agent = Agent()
environment = Environment(1000,agent)
regret_w,regret_s,reger_rho,total_runs,total_wickets,run_time = environment.innings()


# In[ ]:


print(regret_w,regret_s,reger_rho,total_runs,total_wickets,run_time)


# In[35]:





# In[ ]:




