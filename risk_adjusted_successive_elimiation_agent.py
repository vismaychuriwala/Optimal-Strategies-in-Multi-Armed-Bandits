import numpy as np

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
        self.s= [0, 1, 2, 3, 4, 6]
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
                self.lastaction=action
                return action
            else:
                delta = 0.3
                count = len(self.s) - 1
                mu = np.multiply((1 - np.divide(self.arm_wickets, self.pulls)),np.divide(self.arm_runs, self.pulls))
                ct = np.divide(self.arm_wickets,self.pulls)
                mc = np.divide(mu,ct)
                for elem in range(len(mc)):
                    if np.isnan(mc[elem]):
                        mc[elem] = 0
                    if np.isinf(mc[elem]):
                        mc[elem] = 0
                mcmax=max(mc)
                istar=list(np.where(mc == mcmax)[0])[0]
                
                mustar=mu[istar]
                ctstar=ct[istar]
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