from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import globalvar as gl


def high_reward(ind_thread, next_obs, obs, action, micro_isdone):
    reward = 0.0
    minerals = next_obs.observation.player.minerals
    last_minerals = obs.observation.player.minerals
    minerals_change = minerals - last_minerals  # 矿改变量
    idle_worker = next_obs.observation.player.idle_worker_count  # 空闲工人
    vespene = next_obs.observation.player.vespene  # 气矿
    last_vespene = obs.observation.player.vespene
    vespene_change = last_vespene - vespene  # 气矿改变量

    army_change = next_obs.observation["player"][5] - obs.observation["player"][5]  # 军队变化
    worker_change = next_obs.observation["player"][6] - obs.observation["player"][6]  # 农民变化
    food_remain = next_obs.observation["player"][4] - next_obs.observation["player"][3]  # 剩余人口

    step = gl.get_value(ind_thread, "num_steps")  # 当前的步数，1秒2.8步，50步约为18s
    # 190125改写各项系数，yxy

    # 动作执行成功或失败：micro_is_done出现-1的情况，就说明宏动作失败了（出现微动作id不在available_action_list里的情况）
    # 同时，切记！ micro_is_done为1时，不要给正奖励，因为这不能说明宏动作是成功了还是失败了
    if micro_isdone == -1:
        reward -= 50

    if action.function is 140 or 143 or 144 or 168:  # 如果是取消类动作
        reward -= 10
    if not len(obs.observation.last_actions) and action.function != 1:  # 如果没有有效操作 该操作待议
        reward -= 10

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
        # print("idle_workers are :%d"%idle_worker)

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
    build_score_change = next_obs.observation["score_cumulative"][4] - obs.observation["score_cumulative"][4]

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
    # # 兵营编号----在run_thread存储
    # if build_score_change == 150 and barrack_num > 0:
    #     gl.add_value_list(ind_thread, "barrack_location", [0, 0])  # 暂时用"(x，y)"代替坐标

    if step >= 300 and supply_num == 0:
        reward -= 500

    if step >= 500 and barrack_num == 0:
        reward -= 400

    if step >= 50 and obs.observation.player.food_workers <= 12:
        reward -= 10

    if build_score_change > 0:
        reward += build_score_change

    # # build units score  重复了，先不用
    # total_value_units_change = next_obs.observation["score_cumulative"][3] - obs.observation["score_cumulative"][3]
    # if total_value_units_change > 0:
    #     reward += total_value_units_change

    # kill units score
    killed_score_units_change = 10 * (
            next_obs.observation["score_cumulative"][5] - obs.observation["score_cumulative"][5])
    if killed_score_units_change > 0:
        reward += killed_score_units_change
    # kill structure score
    killed_score_structures_change = 10 * (
            next_obs.observation["score_cumulative"][6] - obs.observation["score_cumulative"][6])
    if killed_score_structures_change > 0:
        reward += killed_score_structures_change

    # 生存时间与胜利条件判断
    if step > 1000:
        step_r = step - 1000
        reward += (step_r / 10)
    if step >= 3000 and obs.observation.player.food_workers >= 12 and obs.observation.score_cumulative.total_value_structures >= 400:
        # 存活18分钟，人口大于12，基地没被打爆
        reward += 500

    # 归一化到正负1之间
    reward = float(reward / 1000)
    if reward >= 1.0:
        reward = 1.0
    if reward <= -1.0:
        reward = -1.0

    # print("Thread%d high_reward is %.4f" % (ind_thread, reward))
    return reward
