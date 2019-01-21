from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import agents.globalvar as gl

def high_reward(ind_thread, next_obs, obs, action, micro_isdone):

    reward = 0
    minerals = next_obs.observation.player.minerals
    last_minerals = obs.observation.player.minerals
    minerals_change = minerals - last_minerals  # 矿改变量
    idle_worker = next_obs.observation.player.idle_worker_count  # 空闲工人
    vespene = next_obs.observation.player.vespene  # 气矿
    last_vespene = obs.observation.player.vespene
    vespene_change = last_vespene - vespene  # 气矿改变量
    step = gl.get_value(ind_thread, "num_frames") #当前的步数，是UPDATE_GLOBAL_ITER的倍数，这里是50的倍数。1秒2.8步，50步约为18s

    # print("step is ",step)
    # print("workernumber ", obs.observation.player.food_workers)
    # print("num_frames is",gl.get_value("num_frames"))

    # 动作执行成功
    if micro_isdone == -1:
      reward -= 100
    # 动作执行失败
    if micro_isdone == 1:
      reward += 100

    if step> 50 and obs.observation.player.food_workers <= 12:
       reward -= 10

    if action.function is 140 or 143 or 144 or 168:  # 如果是取消类动作
        reward -= 10
    if not len(obs.observation.last_actions):  # 如果没有有效操作 该操作待议
        reward -= 10

    if minerals >= 400:
        reward -= 1
        if minerals_change < 0:
            reward += 10
    if 200 <= minerals < 400:
        if minerals_change < 0:
            reward += 10
    if 0 <= minerals < 200:
        if minerals_change < 0:
            reward += 10
        else:
            reward += 1

    if idle_worker > 0:
        reward -= 2 * idle_worker
        # print("idle_workers are :%d"%idle_worker)

    # 补给站是100，兵营是150，指挥中心是400
    build_score_change = next_obs.observation["score_cumulative"][4] - obs.observation["score_cumulative"][4]
    if build_score_change > 0:
        if build_score_change == 150:
            reward += 100
        elif build_score_change == 100:
            reward += 50
    #print("buile_score_change is %d"%build_score_change)

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


    # build units score
    total_value_units_change = next_obs.observation["score_cumulative"][3] - obs.observation["score_cumulative"][3]
    if total_value_units_change > 0:
        reward += total_value_units_change
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