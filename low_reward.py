import math
import globalvar as GL


def low_reward(next_obs, obs, coordinate, micro_isdone, macro_type, coord_type, ind_thread):
    reward = 0
    # 坐标x方向向右为正，y方向向下为正，左上角是[0, 0]
    base = [19, 22]  # minimap
    enemy = [40, 45]  # minimap
    enemy_2 = [15, 47]  # minimap
    defense = [40, 20]  # minimap
    defense_base = [29, 20]  # minimap
    barrack = [15, 40]  # screen
    supply = [40, 25]  # screen
    build_score_change = next_obs.observation["score_cumulative"][4] - obs.observation["score_cumulative"][4]
    killed_score_units_change = 10 * (
            next_obs.observation["score_cumulative"][5] - obs.observation["score_cumulative"][5])
    killed_score_structures_change = 10 * (
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
        if 1 < dis <= 3:  # 0305
            # reward = 500
            reward += 1000 - dis * 300
        elif dis <= 1:
            reward = 0
        else:
            reward -= dis * 300

        if build_score_change == 100:
            reward += 300

        if reward > 1000:
            reward = 1000
        if reward < -1000:
            reward = -1000
        reward = float(reward / 1000)
        return reward

    # build_barrack
    if dir_high == 2:
        dis = math.sqrt((coordinate[0] - barrack[0]) ** 2 + (coordinate[1] - barrack[1]) ** 2)
        if 2 < dis <= 5:  # 0305
            # reward = 500
            reward += 1000 - dis * 200
        elif dis <= 2:
            reward = 0
        else:
            reward += 1000 - dis * 300

        if build_score_change == 150:
            reward += 500

        if reward > 1000:
            reward = 1000
        if reward < -1000:
            reward = -1000
        reward = float(reward / 1000)
        if reward > 0.5:
            print("build_barrack_reward: %.4f" % reward)
        return reward

    # 坐标类型为minimap
    if coord_type == 1:
        # 对己方操作
        if macro_type == 0:
            dis = math.sqrt((coordinate[0] - base[0]) ** 2 + (coordinate[1] - base[1]) ** 2)
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
            if reward > 0.5:
                print("Low_reward_def is %.4f" % reward)

        # 对敌方操作
        if macro_type == 1:
            dis_atk = math.sqrt((coordinate[0] - enemy[0]) ** 2 + (coordinate[1] - enemy[1]) ** 2)
            dis_atk_2 = math.sqrt((coordinate[0] - enemy_2[0]) ** 2 + (coordinate[1] - enemy_2[1]) ** 2)
            dis_def = math.sqrt((coordinate[0] - defense[0]) ** 2 + (coordinate[1] - defense[1]) ** 2)
            dis_def_base = math.sqrt((coordinate[0] - defense_base[0]) ** 2 + (coordinate[1] - defense_base[1]) ** 2)
            dis = min(dis_atk, dis_atk_2, dis_def, dis_def_base)
            reward += 500 - dis * 40

            if killed_score_units_change > 0:
                reward += 10 * killed_score_units_change
            if killed_score_structures_change > 0:
                reward += 10 * killed_score_structures_change

            if reward > 1000:
                reward = 1000
            if reward < -1000:
                reward = -1000
            reward = float(reward / 1000)
            if reward > 0.5:
                print("Low_reward_atk is %.4f" % reward)
        return reward

    # 坐标类型为screen
    else:
        # 对己方操作
        if macro_type == 0:
            dis = math.sqrt((coordinate[0] - base[0]) ** 2 + (coordinate[1] - base[1]) ** 2)
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

            if killed_score_units_change > 0:
                reward += killed_score_units_change
            if killed_score_structures_change > 0:
                reward += killed_score_structures_change

            if reward > 1000:
                reward = 1000
            if reward < -1000:
                reward = -1000
            reward = float(reward / 1000)
            if reward > 0.5:
                print("Low_reward_scr is %.4f" % reward)

        # 对敌方操作
        if macro_type == 1:
            print("Low_screen to enemy_empty")

        return reward
