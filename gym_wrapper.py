
import numpy as np
import gym
import pickle
import random

def standardScale(aa):
    K = 3.29 #99.9%
    #K = 1.96 #95%
    meanV = np.mean(aa)
    stdV = np.std(aa)
    #ss_meanV = 0.00088012709079
    #ss_stdV = 0.0104573489
    #pp_meanV = 11136.855753
    #pp_stdV = 1094.2235127
    x_std = (aa-meanV)/(K*stdV)
    return x_std, meanV, stdV  


with open('/content/Evolutionary-Reinforcement-Learning/envs_repo/train_N.pickle','rb') as f:
    data = pickle.load(file = f)
l = 284 #13:29=> the last one
dataN = len(data)
ss = np.zeros((l,dataN))
pp = np.zeros((l,dataN))
symBe = ''
base = 0

for i in range(dataN):
    sym = (data[i][1])[3:4] #J, K, L
    if sym != symBe:

        #for s in range(l-1):
        for s in range(l):
            #ss[s,i] = (data[i][0].iloc[s,1]/data[i][0].iloc[0,1]-1.)
            ss[s,i] = np.log(data[i][0].iloc[s,1])-np.log(data[i][0].iloc[0,1])
            pp[s,i] = data[i][0].iloc[s,1]
        symBe = sym
        base = data[i][0].iloc[l-1,1]

    else:
        #for s in range(l-1):
        for s in range(l):
            #ss[s,i] = (data[i][0].iloc[s,1]/base-1.)
            ss[s,i] = np.log(data[i][0].iloc[s,1])-np.log(base)
            pp[s,i] = data[i][0].iloc[s,1]
        symBe = sym
        base = data[i][0].iloc[l-1,1]  

ss, _, _ = standardScale(ss)
pp, _, _ = standardScale(pp)
window_size = 4
tradeCost = 47
   
class GymWrapper(gym.Env):

  def __init__(self, env_name, frameskip=None):
      self.state_dim = 6
      self.action_dim = 3
      self.pIndex = ss
      self.price_std = pp
      self.price = data
      self.commission = 47
      self.position = np.array([0.])
      self.inventory = []
      self.d = 0 # day
      self.t = 0 # time
      self.st = 0 # tradin time
      self.done = False

  def is_discrete(self, env):
        try:
            k = env.action_space.n
            return True
        except:
            return False

  def getStateTv(self):
      aa = self.pIndex[self.t-4:self.t , self.d]
      pri_s = np.array([self.price_std[self.t-1, self.d]])
      aa = np.concatenate((aa, pri_s, self.position), axis=0)
      return aa

  def _take_action(self, action):
      reward = 0
      if action == 1:
          if int(self.position[0]) == 0:
              self.position = np.array([1.])
              self.inventory.append(self.price[self.d][0].iloc[self.t-1,1])
              self.st = self.t # tradin time

          if int(self.position[0]) == -1:
              sold_price = self.inventory.pop(0)
              reward = 50*(sold_price - self.price[self.d][0].iloc[self.t-1,1])-2*self.commission
              self.done = True
              self.position = np.array([0.])
      elif action == 2:
          if int(self.position[0]) == 0:
              self.position = np.array([-1.])
              self.inventory.append(self.price[self.d][0].iloc[self.t-1,1])
              self.st = self.t # tradin time

          if int(self.position[0]) == 1:
              bought_price = self.inventory.pop(0)
              reward = 50*(self.price[self.d][0].iloc[self.t-1,1] - bought_price)-2*self.commission
              self.done = True
              self.position = np.array([0.])
       
      return reward, self.position

  def step(self, action):
      reward, self.position = self._take_action(action)
      self.t += 1
      #observation = self.getState()
      observation = self.getStateTv()
      if self.t == 284:
          self.done = True
          if len(self.inventory) > 0:
              if int(self.position[0]) == 1:
                  bought_price = self.inventory.pop(0)
                  reward = 50*(self.price[self.d][0].iloc[self.t-1,1] - bought_price)-2*self.commission
                  #observation[self.state_dim+1] = np.array([0.])
                  observation[5] = np.array([0.])#20220509

              elif int(self.position[0]) == -1:
                  sold_price = self.inventory.pop(0)
                  reward = 50*(sold_price - self.price[self.d][0].iloc[self.t-1,1])-2*self.commission
                  #observation[self.state_dim+1] = np.array([0.])
                  observation[5] = np.array([0.])#20220509
      info = {'env.d':self.d, 'env.t':self.t, 'reward':reward, 'tradein t':self.st}
      #print('observation  ',observation)
      #print('self.done  ',self.done)
      return observation, reward, self.done, info

  def reset(self):
      self.position = np.array([0.])
      self.inventory = []
      dth = random.randint(0, self.pIndex.shape[1]-1)
      self.d = dth # day
      self.t = 4 # time
      self.st = 4 # tradin time
      self.done = False

      return self.getStateTv() 

  def render(self):
      pass

