import math
def low_reward(next_obs, obs, coordinate,micro_isdone,macro_type,coord_type):
  reward = 0
  ourside=[20,25]
  enemyside=[52,49]

  build_score_change = next_obs.observation["score_cumulative"][4] - obs.observation["score_cumulative"][4]
  killed_value_units_change = 10 * (next_obs.observation["score_cumulative"][5] - obs.observation["score_cumulative"][5])
  killed_value_structures_change = 10 * (next_obs.observation["score_cumulative"][6] - obs.observation["score_cumulative"][6])

  #坐标类型为minimap
  if coord_type==1:
      #对己方操作
      if macro_type == 0:
        dis = math.sqrt((coordinate[0] - ourside[0]) ** 2 + (coordinate[1] - ourside[1]) ** 2)
        if dis<=5:
            reward = 1000
        else:
            reward += 1000-dis * 100

        if build_score_change > 0:
            if build_score_change == 150:
                reward += 100
            elif build_score_change == 100:
                reward += 50

        if reward > 1000:
            reward = 1000
        if reward < -1000:
            reward = -1000
        reward = float(reward / 1000)
        return reward

      #对敌方操作
      if macro_type == 1:
        dis = math.sqrt((coordinate[0]-enemyside[0])**2+(coordinate[1]-enemyside[1])**2)
        reward += 1000 - dis * 100

        if killed_value_units_change > 0:
            reward += killed_value_units_change
        if killed_value_structures_change > 0:
            reward += killed_value_structures_change

        if reward > 1000:
            reward = 1000
        if reward < -1000:
            reward = -1000
        reward = float(reward / 1000)
        return reward

  # 坐标类型为screen
  else:
      if build_score_change > 0:
          if build_score_change == 150:
              reward += 100
          elif build_score_change == 100:
              reward += 50

      if killed_value_units_change > 0:
          reward += killed_value_units_change
      if killed_value_structures_change > 0:
          reward += killed_value_structures_change

      if reward > 1000:
          reward = 1000
      if reward < -1000:
          reward = -1000
      reward = float(reward / 1000)
      return reward