from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import agents.globalvar as gl
import numpy as np

def high_reward(ind_thread, next_obs, obs, action, micro_isdone):

    reward = 0
    minerals = next_obs.observation.player.minerals
    last_minerals = obs.observation.player.minerals
    minerals_change = minerals - last_minerals  # 矿改变量
    idle_worker = next_obs.observation.player.idle_worker_count  # 空闲工人
    vespene = next_obs.observation.player.vespene  # 气矿
    last_vespene = obs.observation.player.vespene
    vespene_change = last_vespene - vespene  # 气矿改变量

    army_change = next_obs.observation["player"][5]-obs.observation["player"][5]       #军队变化
    worker_change = next_obs.observation["player"][6]-obs.observation["player"][6]     #农民变化
    food_remain = next_obs.observation["player"][4] - next_obs.observation["player"][3] #剩余人口

    step = gl.get_value(ind_thread, "num_frames") #当前的步数，是UPDATE_GLOBAL_ITER的倍数，这里是50的倍数。1秒2.8步，50步约为18s

    # print("step is ",step)
    # print("workernumber ", obs.observation.player.food_workers)
    # print("num_frames is",gl.get_value("num_frames"))

    # 动作执行成功或失败：micro_is_done出现-1的情况，就说明宏动作失败了（出现微动作id不在available_action_list里的情况）
    # 同时，切记！ micro_is_done为1时，不要给正奖励，因为这不能说明宏动作是成功了还是失败了
    if micro_isdone == -1:
      reward -= 666

    if action.function is 140 or 143 or 144 or 168:  # 如果是取消类动作
        reward -= 10
    if not len(obs.observation.last_actions) and action.function != 0:  # 如果没有有效操作 该操作待议
        reward -= 10

    #矿变化相关reward
    if minerals >= 500 and minerals_change>=0:
        reward -= 50
    if minerals >= 400 and minerals < 500 and minerals_change>=0:
        reward -= 0.5*(minerals-400)
    if 0 <= minerals < 200 and minerals_change > 0:
        reward += 1


    #闲置农民惩罚
    if idle_worker > 0:
        reward -= 2 * idle_worker
        # print("idle_workers are :%d"%idle_worker)

    #剩余人口惩罚
    if food_remain < 2:    # food_remain可能为负数
        reward -= 5 * np.abs(food_remain)
        if food_remain ==0 :
            reward -= 50

    #军队数量增加奖励
    if army_change>0:
        reward += 50

    #农民数量增加奖励
    worker_army = next_obs.observation["player"][6]
    if worker_change>0 and  worker_army <=20:
        reward += 10

    #没有军队惩罚
    food_army = next_obs.observation["player"][5]
    if step >= 500 and food_army == 0:
        reward -= 20


    # 建造得分计算，补给站是100，兵营是150，指挥中心是400
    build_score_change = next_obs.observation["score_cumulative"][4] - obs.observation["score_cumulative"][4]

    #补给站数目
    supply_num = gl.get_value(ind_thread, "supply_num")
    if build_score_change == 100:
        supply_num += 1
        gl.set_value(ind_thread, "supply_num", supply_num)
    if build_score_change == -100:
        supply_num -= 1
        gl.set_value(ind_thread, "supply_num", supply_num)
    # print("thread%d supply_num is %d" % (ind_thread, supply_num))

    # #补给站编号（补给站建造起来方便，调试时用补给站代替兵营调试）
    # if build_score_change == 100 and supplydepot_number != 0:
    #     supplydepot_x= "supplydepot_" + str(supplydepot_number)
    #     gl.set_value(supplydepot_x, "(x,y)")
    #     location =  gl.get_value(supplydepot_x, "(x,y)")
    #     print("####################",supplydepot_x, "is",location,"#####################")

    #兵营数目
    barrack_num = gl.get_value(ind_thread, "barrack_num")
    if build_score_change == 150:
        barrack_num += 1
        gl.set_value(ind_thread, "barrack_num", barrack_num)
    if build_score_change == -150:
        barrack_num -= 1
        gl.set_value(ind_thread, "barrack_num", barrack_num)
        # print("thread%d barrack_num is %d" % (ind_thread, barrack_num))

    #兵营编号
    if  build_score_change == 150 and barrack_num != 0:
        gl.add_value_list(ind_thread, "brrack_location", [0, 0]) # 暂时用"(x，y)"代替坐标

    if  step >= 300 and  supply_num == 0:
        reward -= 50

    if  step >= 500 and  barrack_num == 0:
        reward -= 50

    if step >= 50 and obs.observation.player.food_workers <= 12:
        reward -= 10


    if build_score_change > 0:
        if build_score_change == 150:
            reward += 50
        elif build_score_change == 100:
            reward += 30
            # print("buile_score_change is %d"%build_score_change)


    # # build units score  重复了，先不用
    # total_value_units_change = next_obs.observation["score_cumulative"][3] - obs.observation["score_cumulative"][3]
    # if total_value_units_change > 0:
    #     reward += total_value_units_change



    # kill units score
    killed_value_units_change = 10 * (next_obs.observation["score_cumulative"][5] - obs.observation["score_cumulative"][5])
    if killed_value_units_change > 0:
        reward += killed_value_units_change
    # kill structure score
    killed_value_structures_change = 10*(next_obs.observation["score_cumulative"][6] - obs.observation["score_cumulative"][6])
    if killed_value_structures_change > 0:
        reward += killed_value_structures_change

    # 生存时间与胜利条件判断
    if step >1000:
        step_r = step - 1000
        reward += (step_r / 100)
    if step >= 3000 and  obs.observation.player.food_workers>=12 and obs.observation.score_cumulative.total_value_structures >=400:
     #存活18分钟，人口大于12，基地没被打爆
       reward = 1000

    #归一化到正负1之间
    reward = float(reward / 1000)
    if reward >= 1.0:
        reward = 1.0
    if reward <= -1.0:
        reward = -1.0

    print("thread%d reward is %.4f" % (ind_thread, reward))
    return reward