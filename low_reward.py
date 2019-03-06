import math
import globalvar as GL


def low_reward(next_obs, obs, coordinate, micro_isdone, macro_type, coord_type, ind_thread):
    reward = 0
    # 坐标x方向向下为正，y方向向右为正，左上角是[0, 0]
    ourside = [20, 25]
    enemyside = [44, 39]
    barrack = [20, 35]
    supply = [40, 20]
    build_score_change = next_obs.observation["score_cumulative"][4] - obs.observation["score_cumulative"][4]
    killed_value_units_change = 10 * (
            next_obs.observation["score_cumulative"][5] - obs.observation["score_cumulative"][5])
    killed_value_structures_change = 10 * (
            next_obs.observation["score_cumulative"][6] - obs.observation["score_cumulative"][6])
    army_change = next_obs.observation["player"][5] - obs.observation["player"][5]  # 军队变化
    dir_high = GL.get_value(ind_thread, "dir_high")
    ind_todo = GL.get_value(ind_thread, "ind_micro")

    # if micro_isdone == -1:
    #     reward -= 100
    if not len(obs.observation.last_actions):
        reward -= 100
        # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        # last_actions为[],代表动作函数合法但失败（比如造补给站在available_action_list里，但选的建造坐标在基地的位置上，则造不出来）

    # 针对宏动作设计精准奖励
    # build_supply
    if dir_high == 1:
        dis = math.sqrt((coordinate[0] - supply[0]) ** 2 + (coordinate[1] - supply[1]) ** 2)
        if 2 < dis <= 10:  # 0305
            # reward = 500
            reward += 100 - dis * 10
        elif dis <= 2:
            reward = 0
        else:
            reward -= dis * 10

        if build_score_change == 100:
            reward += 500

        if reward > 1000:
            reward = 1000
        if reward < -1000:
            reward = -1000
        reward = float(reward / 1000)
        # if reward != 0:
        #     print("build_supply_reward: %.4f" % reward)
        return reward

    # build_barrack
    if dir_high == 2:
        dis = math.sqrt((coordinate[0] - barrack[0]) ** 2 + (coordinate[1] - barrack[1]) ** 2)
        if 2 < dis <= 10:  # 0305
            # reward = 500
            reward += 100 - dis * 10
        elif dis <= 2:
            reward = 0
        else:
            reward += 100 - dis * 10

        if build_score_change == 150:
            reward += 500

        if reward > 1000:
            reward = 1000
        if reward < -1000:
            reward = -1000
        reward = float(reward / 1000)
        if reward != 0:
            print("build_barrack_reward: %.4f" % reward)
        return reward

    # 坐标类型为minimap
    if coord_type == 1:
        # 对己方操作
        if macro_type == 0:
            print("Low_minimap to self")
            dis = math.sqrt((coordinate[0] - ourside[0]) ** 2 + (coordinate[1] - ourside[1]) ** 2)
            if dis <= 35:  # 0304, 25*1.4=35
                reward = 500
                reward += 100 - dis * 2
            else:
                reward += 100 - dis * 10

            if build_score_change > 0:
                if build_score_change == 150:
                    reward += 100
                elif build_score_change == 100:
                    reward += 50

            # 军队数量奖惩   yxy
            if army_change > 0:
                reward += 500

            # reward *= 10
            if reward > 1000:
                reward = 1000
            if reward < -1000:
                reward = -1000
            reward = float(reward / 1000)
            if reward != 0:
                print("Low_reward_def is %.4f" % reward)

        # 对敌方操作
        if macro_type == 1:
            print("Low_minimap to enemy")
            dis = math.sqrt((coordinate[0] - enemyside[0]) ** 2 + (coordinate[1] - enemyside[1]) ** 2)
            reward += 100 - dis * 10

            if killed_value_units_change > 0:
                reward += killed_value_units_change
            if killed_value_structures_change > 0:
                reward += killed_value_structures_change

            # reward *= 10
            if reward > 1000:
                reward = 1000
            if reward < -1000:
                reward = -1000
            reward = float(reward / 1000)
            if reward != 0:
                print("Low_reward_atk is %.4f" % reward)
        return reward

    # 坐标类型为screen
    else:
        # 对己方操作
        if macro_type == 0:
            # print("Low_screen to self")
            dis = math.sqrt((coordinate[0] - ourside[0]) ** 2 + (coordinate[1] - ourside[1]) ** 2)
            if 2 < dis <= 20:  # 0305
                # reward = 500
                reward += 200 - dis * 2
            elif dis <= 2:
                reward = 0
            else:
                reward += 100 - dis * 10

            if build_score_change > 0:
                if build_score_change == 150:
                    reward += 100
                elif build_score_change == 100:
                    reward += 50
            # 军队数量奖惩   yxy
            if army_change > 0:
                reward += 500

            if killed_value_units_change > 0:
                reward += killed_value_units_change
            if killed_value_structures_change > 0:
                reward += killed_value_structures_change

            if reward > 1000:
                reward = 1000
            if reward < -1000:
                reward = -1000
            reward = float(reward / 1000)
            if reward != 0:
                print("Low_reward_scr is %.4f" % reward)

        # 对敌方操作
        if macro_type == 1:
            print("Low_screen to enemy_empty")

        return reward
