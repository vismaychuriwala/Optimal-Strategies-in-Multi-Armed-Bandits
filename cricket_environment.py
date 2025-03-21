import numpy as np
import time

np.seterr(divide='ignore', invalid='ignore')

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
