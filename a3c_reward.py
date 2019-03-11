from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import globalvar as gl


def a3c_reward(ind_thread, next_obs, obs, micro_isdone, coordinate, macro_id, macro_type, coord_type):
    reward = 0.0
    minerals = next_obs.observation.player.minerals
    last_minerals = obs.observation.player.minerals
    minerals_change = minerals - last_minerals  # 矿改变量
    idle_worker = next_obs.observation.player.idle_worker_count  # 空闲工人
    # vespene = next_obs.observation.player.vespene  # 气矿
    # last_vespene = obs.observation.player.vespene
    # vespene_change = last_vespene - vespene  # 气矿改变量
    army_change = next_obs.observation["player"][5] - obs.observation["player"][5]  # 军队变化
    worker_change = next_obs.observation["player"][6] - obs.observation["player"][6]  # 农民变化
    food_remain = next_obs.observation["player"][4] - next_obs.observation["player"][3]  # 剩余人口
    step = gl.get_value(ind_thread, "num_steps")  # 当前的步数，1秒2.8步，50步约为18s
    # 坐标x方向向右为正，y方向向下为正，左上角是[0, 0]
    base = [20, 25]  # minimap
    enemy = [44, 39]  # minimap
    enemy_2 = [20, 39]  # minimap
    defense = [40, 25]  # minimap
    defense_base = [25, 25]  # minimap
    barrack = [15, 35]  # screen
    supply = [40, 25]  # screen
    build_score_change = next_obs.observation["score_cumulative"][4] - obs.observation["score_cumulative"][4]
    killed_score_units_change = 10 * (
            next_obs.observation["score_cumulative"][5] - obs.observation["score_cumulative"][5])
    killed_score_structures_change = 10 * (
            next_obs.observation["score_cumulative"][6] - obs.observation["score_cumulative"][6])
    army_change = next_obs.observation["player"][5] - obs.observation["player"][5]  # 军队变化
    dir_high = gl.get_value(ind_thread, "dir_high")
    # ind_todo = gl.get_value(ind_thread, "ind_micro")
    # 190125改写各项系数，yxy

    # 动作执行成功或失败：micro_is_done出现-1的情况，就说明宏动作失败了（出现微动作id不在available_action_list里的情况）
    # 同时，切记！ micro_is_done为1时，不要给正奖励，因为这不能说明宏动作是成功了还是失败了
    if micro_isdone == -1:
        reward -= 500

    # 矿变化相关reward
    # 190125改写，yxy
    if minerals >= 500 and minerals_change >= 0:
        reward -= 50
    # if minerals >= 200 and minerals < 500 and minerals_change>=0:
    #     reward += 20-0.5*(minerals-200)
    if 500 > minerals >= 200 and minerals_change >= 0:
        reward += 20 - 0.2 * (minerals - 200)
    if 0 <= minerals < 200 and minerals_change > 0:
        reward += minerals / 10

    # 闲置农民惩罚
    if idle_worker > 0:
        reward -= 20 * idle_worker

    # 剩余人口奖惩
    if 0 < food_remain < 2:  # food_remain可能为负数
        reward -= 10 * food_remain
    if food_remain <= 0:
        reward -= 500

    # 农民数量奖惩   yxy
    worker_count = next_obs.observation["player"][6]
    if worker_change > 0 and worker_count <= 22:
        reward += 10
    if worker_change > 0 and worker_count > 22:
        reward -= 400

    # 军队数量奖惩   yxy
    army_count = next_obs.observation["player"][5]
    if army_count > 0:
        reward += 30 * army_count
    if step >= 500 and army_count == 0:
        reward -= 300
    if army_change > 0:
        reward += 50

    # 建造得分计算，补给站是100，兵营是150，指挥中心是400
    # 补给站数目
    supply_num = gl.get_value(ind_thread, "supply_num")
    if build_score_change == 100:
        supply_num += 1
        gl.set_value(ind_thread, "supply_num", supply_num)
    if build_score_change == -100:
        supply_num -= 1
        gl.set_value(ind_thread, "supply_num", supply_num)
    # 兵营数目
    barrack_num = gl.get_value(ind_thread, "barrack_num")
    if build_score_change == 150:
        barrack_num += 1
        gl.set_value(ind_thread, "barrack_num", barrack_num)
    if build_score_change == -150:
        barrack_num -= 1
        gl.set_value(ind_thread, "barrack_num", barrack_num)

    if step >= 300 and supply_num == 0:
        reward -= 500
    if step >= 500 and barrack_num == 0:
        reward -= 400
    if step >= 50 and obs.observation.player.food_workers <= 12:
        reward -= 100
    if build_score_change > 0:
        reward += 3 * build_score_change

    # # build units score  重复了，先不用
    # total_value_units_change = next_obs.observation["score_cumulative"][3] - obs.observation["score_cumulative"][3]
    # if total_value_units_change > 0:
    #     reward += total_value_units_change

    # kill units score
    if killed_score_units_change > 0:
        reward += killed_score_units_change
    # kill structure score
    if killed_score_structures_change > 0:
        reward += killed_score_structures_change

    # 生存时间与胜利条件判断
    if step > 1000:
        step_r = step - 1000
        reward += (step_r / 10)
    if step >= 3000 and obs.observation.player.food_workers >= 12 and obs.observation.score_cumulative.total_value_structures >= 400:
        # 存活18分钟，人口大于12，基地没被打爆
        reward += 500

    if not len(obs.observation.last_actions):
        reward -= 100
        # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        # last_actions为[],代表动作函数合法但失败（比如造补给站在available_action_list里，但选的建造坐标在基地的位置上，则造不出来）
    if coordinate[0] != -99 and coordinate[1] != -99:    # 调用了spatial
        # 针对宏动作设计精准奖励
        # build_supply
        if macro_id == 1:
            dis = math.sqrt((coordinate[0] - supply[0]) ** 2 + (coordinate[1] - supply[1]) ** 2)
            if 1 < dis <= 5:  # 0305
                # reward = 500
                reward += 100 - dis * 20
            elif dis <= 1:
                reward = 0
            else:
                reward -= dis * 100
    
            if build_score_change == 100:
                reward += 300
    
        # build_barrack
        if macro_id == 2:
            dis = math.sqrt((coordinate[0] - barrack[0]) ** 2 + (coordinate[1] - barrack[1]) ** 2)
            if 2 < dis <= 5:  # 0305
                # reward = 500
                reward += 100 - dis * 20
            elif dis <= 2:
                reward = 0
            else:
                reward += 100 - dis * 10
    
            if build_score_change == 150:
                reward += 500
    
            if reward != 0:
                print("build_barrack_reward: %.4f" % reward)
    
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
                if reward != 0:
                    print("Low_reward_def is %.4f" % reward)
    
            # 对敌方操作
            if macro_type == 1:
                dis_atk = math.sqrt((coordinate[0] - enemy[0]) ** 2 + (coordinate[1] - enemy[1]) ** 2)
                dis_atk_2 = math.sqrt((coordinate[0] - enemy_2[0]) ** 2 + (coordinate[1] - enemy_2[1]) ** 2)
                dis_def = math.sqrt((coordinate[0] - defense[0]) ** 2 + (coordinate[1] - defense[1]) ** 2)
                dis_def_base = math.sqrt((coordinate[0] - defense_base[0]) ** 2 + (coordinate[1] - defense_base[1]) ** 2)
                dis = min(dis_atk, dis_atk_2, dis_def, dis_def_base)
                reward += 200 - dis * 5
    
                if killed_score_units_change > 0:
                    reward += 10 * killed_score_units_change
                if killed_score_structures_change > 0:
                    reward += 10 * killed_score_structures_change
    
                if reward != 0:
                    print("Low_reward_atk is %.4f" % reward)
    
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
    
                if reward != 0:
                    print("Low_reward_scr is %.4f" % reward)
    
            # 对敌方操作
            if macro_type == 1:
                print("Low_screen to enemy_empty")

    # 归一化到正负1之间
    reward = float(reward / 2000)
    if reward >= 1.0:
        reward = 1.0
    if reward <= -1.0:
        reward = -1.0

    # print("Thread%d high_reward is %.4f" % (ind_thread, reward))
    return reward
