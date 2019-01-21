import math
def low_reward(next_obs, obs, type_of_action, coordinate):
  reward = 0
  ourside=[20,25]
  enemyside=[52,49]
  #对己方操作
  if type_of_action == 0:
    reward += 100
    dis = math.sqrt((coordinate[0] - ourside[0]) ** 2 + (coordinate[1] - ourside[1]) ** 2)
    reward -= dis * 10
    if reward < -100:
        reward = -100
    reward = float(reward / 100)
    return reward

  #对敌方操作
  if type_of_action == 1:
    reward += 100
    dis = math.sqrt((coordinate[0]-enemyside[0])**2+(coordinate[1]-enemyside[1])**2)
    reward -= dis * 10
    if reward < -100:
      reward = -100
    reward = float(reward / 100)
    return reward

if __name__ == '__main__':

   #测试一下
   next_obs = None
   obs = None
   type_of_action = 0
   print( low_reward(next_obs, obs , type_of_action, [16,17]))