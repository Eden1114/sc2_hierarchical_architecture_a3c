from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pysc2.env import sc2_env
from pysc2.env import available_actions_printer
from pysc2.lib import actions
import time
import numpy as np
from macro_actions import action_micro
import globalvar as GL

list_actions, _ = GL.get_list()


# agents是列表，里面有一个agent（A3CAgent对象），env是SC2Env对象经过处理后的变量，max_frames是回合内最多进行的step数
# 190305：将main的run_thread与原有的run_loop放到同一个文件内。
# run_thread.py是main与agent的衔接，run_thread负责执行thread，run_loop控制episode与step进程，控制agent和环境的交互，控制数据保存
def run_loop(agents, env, max_steps, ind_thread):
    start_time = time.time()
    try:
        while True:  # 底下发生的是一个episode内的过程
            # 每局游戏需要用的全局变量清空
            GL.episode_init(ind_thread)
            num_steps = 0  # 计算回合里的step数
            num_call_step_low = 0  # 计算回合内调用下层网络的次数
            timesteps = env.reset()
            # reset函数返回TimeStep四元组（sc2_env.py 512行），包含的信息有4种，在知乎上PySC2详解的文章里有介绍
            for a in agents:  # 为后面的for agent, timestep in zip(agents, timesteps) 快速遍历提供方便，所以写成单一元素的列表
                a.reset()

            while True:  # 底下发生的是回合内一个step的过程
                ind_last = GL.get_value(ind_thread, "ind_micro")
                num_steps += 1
                GL.set_value(ind_thread, "num_steps", num_steps)
                last_timesteps = timesteps
                # DHN add:
                if ind_last == -1 or ind_last == -99 or ind_last == 666:
                    # ind_last == -99 (表示宏动作里的微动作执行失败)
                    # ind_last == 666 (表示宏动作成功执行完毕）:
                    dir_high = [agent.step_high(timestep) for agent, timestep in
                                zip(agents, timesteps)]  # dir_high是 要执行的宏动作id（从0开始）
                    GL.set_value(ind_thread, "dir_high", dir_high[0])
                    GL.set_value(ind_thread, "ind_micro", 0)
                    ind_todo = GL.get_value(ind_thread, "ind_micro")
                else:
                    temp = GL.get_value(ind_thread, "ind_micro")
                    GL.set_value(ind_thread, "ind_micro", temp + 1)
                    ind_todo = GL.get_value(ind_thread, "ind_micro")

                dir_high = GL.get_value(ind_thread, "dir_high")
                action, call_step_low, act_id, macro_type, coord_type = action_micro(ind_thread, dir_high,
                                                                                     ind_todo)
                # 关键一步，调用了macro_action.action_micro计算出选择的action和其他参数。
                # 如果call_step_low为False，则act_id没用了，直接使用上行中的action
                # 如果其为True，则进入以下的模块，action没用了，act_id被使用来计算新的action
                if call_step_low:
                    num_call_step_low += 1
                    GL.set_value(ind_thread, "act_id_micro", ind_todo)
                    target_pack = [agent.step_low(ind_thread, timestep, dir_high, ind_todo) for agent, timestep in
                                   zip(agents, timesteps)]
                    target_0 = target_pack[0][0]
                    target_1 = target_pack[0][1]
                    if dir_high == 2 and ind_todo == 3:  # 代表当前要执行的动作是宏动作“build_barrack”中的序号3微动作：造兵营（动作函数42）
                        GL.set_value(ind_thread, "barrack_location_NotSure", [target_1, target_0])
                        print("Thread ", ind_thread, end=" ")
                        print("Barrack_location_NotSure: ", [target_1, target_0])
                    act_args = []
                    for arg in actions.FUNCTIONS[act_id].args:  # actions是pysc2.lib中的文件 根据act_id获取其可使用的参数，并添加到args中去
                        if arg.name in ('screen', 'minimap', 'screen2'):
                            act_args.append([target_1, target_0])
                        else:
                            act_args.append([0])  # TODO: Be careful
                        action = [actions.FunctionCall(act_id, act_args)]

                # 校验宏动作：
                flag_success = True
                if list_actions[dir_high][ind_todo] not in last_timesteps[0].observation['available_actions']:
                    GL.set_value(ind_thread, "ind_micro", -99)  # 表示宏动作里的微动作执行失败
                    action = [actions.FunctionCall(function=0, arguments=[])]  # 执行no_op
                    flag_success = False
                if flag_success:
                    GL.add_value_list(ind_thread, "micro_isdone", 1)
                else:
                    GL.add_value_list(ind_thread, "micro_isdone", -1)
                # 当ind_todo是最后一个需要执行的动作，且执行成功时，将ind_done[ind_thread]设为666（即宏动作成功执行完毕）
                if ind_todo == len(list_actions[dir_high]) - 1 and flag_success:
                    GL.set_value(ind_thread, "ind_micro", 666)  # 表示宏动作执行到了最后一步微动作且执行成功
                timesteps = env.step(action)  # env环境的step函数根据动作计算出下一个timesteps
                # 动作函数合法但失败（比如造补给站在available_action_list里，但选的建造坐标在基地的位置上，则造不出来），则将ind_micro置为-99，表示“宏动作执行失败”
                if not len(timesteps[0].observation.last_actions) and action[0].function != 1:
                    GL.set_value(ind_thread, "ind_micro", -99)
                # 代表当前要执行的动作是宏动作“build_barrack”中的序号3微动作：造兵营（动作函数42）,【且执行成功（ind_micro != 99）】（也有可能没成功，比如造了一半农民走了or造了一半被敌方拆了）
                if dir_high == 2 and ind_todo == 3 and GL.get_value(ind_thread, "ind_micro") != -99:
                    barrack_location = GL.get_value(ind_thread, "barrack_location_NotSure")
                    GL.add_value_list(ind_thread, "barrack_location", barrack_location)

                # Only for a single player!
                is_done = (num_steps >= max_steps) or timesteps[0].last()
                # timesteps[0]是timesteps的第一个变量step_type（状态类型），last()为True即到了末状态
                yield [last_timesteps[0], action[0], timesteps[0]], \
                      is_done, num_steps, call_step_low, num_call_step_low, macro_type, coord_type
                # yield适用于函数返回内容较多，占用内存量很大的情况。可以看成返回了一个列表（实际不是）
                # 详解见http://www.runoob.com/w3cnote/python-yield-used-analysis.html
                if is_done:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        elapsed_time = time.time() - start_time
        print("Took %.3f seconds" % elapsed_time)


def run_thread(agent, map_name, visualize, ind_thread, FLAGS, LOCK):  # A3CAgent对象，地图名（字符串），是否可视化feature map（布尔值）
    players = [sc2_env.Agent(sc2_env.Race[FLAGS.agent_race])]
    players.append(sc2_env.Bot(sc2_env.Race[FLAGS.agent2_race],
                               sc2_env.Difficulty[FLAGS.difficulty]))  # 2018/08/03: 2018/08/03: Simple64枪兵互拼新加代码
    agent_interface = sc2_env.parse_agent_interface_format(  # 增加一个player的代码应该在run_thread开头这里修改
        feature_screen=FLAGS.screen_resolution,
        feature_minimap=FLAGS.minimap_resolution)
    if FLAGS.training:
        PARALLEL = FLAGS.parallel  # PARALLEL 指定开几个线程（几个游戏窗口在跑星际2）
        MAX_AGENT_STEPS = FLAGS.max_agent_steps  # 回合里agent的最大步数
    else:
        PARALLEL = 1
        MAX_AGENT_STEPS = 1e5  # 回合里agent的最大步数
    SNAPSHOT = FLAGS.snapshot_path + FLAGS.map + '/' + FLAGS.net

    with sc2_env.SC2Env(
            map_name=map_name,
            players=players,
            step_mul=8,
            agent_interface_format=agent_interface,
            visualize=visualize) as env:
        env = available_actions_printer.AvailableActionsPrinter(env)

        # Only for a single player!
        counter = GL.get_episode_counter()  # global_episode_counter
        # 后缀1供下层网络更新时使用， 后缀2供上层网络更新时使用
        iswin = -99
        replay_buffer_1 = []
        dir_high_buffer_1 = []
        replay_buffer_2 = []
        dir_high_buffer_2 = []
        num_call_step_low = 0

        # 下行中的run_loop是个生成器，for循环每次进入到run_loop里，得到yield后返回，继续进行循环体里的语句，for循环再次进入run_loop后从run_loop的yield的下一条语句开始执行，执行到yield再次返回，继续执行循环体语句...
        for recorder, is_done, num_steps, call_step_low, num_call_step_low, macro_type, coord_type in run_loop(
                [agent], env, MAX_AGENT_STEPS, ind_thread):  # 将agent对象存入[]再作为参数传递进run_loop生成器里，recorder是一个三元列表
            if FLAGS.training:  # 这里是if FLAGS.training，但后面并没有if not FLAGS.training。即若是非训练模式（restore了以前的网络参数），则不再进行网络参数的更新
                # 默认填充replay_buffer，直到update后清空
                if call_step_low:
                    replay_buffer_1.append(recorder)
                    dir_high_buffer_1.append(GL.get_value(ind_thread, "dir_high"))
                replay_buffer_2.append(recorder)
                dir_high_buffer_2.append([GL.get_value(ind_thread, "dir_high")])

                if is_done:  # 若为训练模式，最终状态，首先锁线程
                    with LOCK:  # 使用线程锁（跟java类似，应用于不同线程会调用相同资源的情况）
                        counter = GL.get_episode_counter()
                        counter += 1
                        GL.set_episode_counter(counter)
                iswin = replay_buffer_2[-1][-1].reward

                # 更新下层网络
                if call_step_low:
                    learning_rate_a_low = FLAGS.learning_rate * (
                            1 - 0.9 * counter / FLAGS.max_episodes)  # 根据当前进行完的回合数量修改学习速率（减小）
                    learning_rate_c_low = FLAGS.learning_rate * (
                            1 - 0.9 * counter / FLAGS.max_episodes)  # 根据当前进行完的回合数量修改学习速率（减小）
                    agent.update_low(ind_thread, replay_buffer_1, dir_high_buffer_1, FLAGS.discount,
                                     learning_rate_a_low, learning_rate_c_low, counter, macro_type, coord_type)
                    replay_buffer_1 = []
                    dir_high_buffer_1 = []

                # 更新上层网络
                ind_last = GL.get_value(ind_thread, "ind_micro")
                if ind_last == -99 or ind_last == 666:
                    learning_rate_a_high = FLAGS.learning_rate * (
                            1 - 0.9 * counter / FLAGS.max_episodes)  # 根据当前进行完的回合数量修改学习速率（减小）
                    learning_rate_c_high = FLAGS.learning_rate * (
                            1 - 0.9 * counter / FLAGS.max_episodes)  # 根据当前进行完的回合数量修改学习速率（减小）
                    agent.update_high(ind_thread, replay_buffer_2, dir_high_buffer_2, FLAGS.discount,
                                      learning_rate_a_high, learning_rate_c_high, counter)
                    replay_buffer_2 = []
                    dir_high_buffer_2 = []

                if is_done:  # 最终状态，后续处理，存储数据
                    print("Episode_counter: ", counter)
                    print("obs.reward_isWin:", iswin)
                    GL.add_value_list(ind_thread, "victory_or_defeat", iswin)
                    GL.add_value_list(ind_thread, "reward_high_list",
                                      GL.get_value(ind_thread, "sum_high_reward") / num_steps)
                    GL.add_value_list(ind_thread, "reward_low_list",
                                      GL.get_value(ind_thread, "sum_low_reward") / num_call_step_low)
                    # global_episode是FLAGS.snapshot_step的倍数+1，或指定回合数
                    # 存单个episode的reward变化，存储网络参数（tf.train.Saver().save(),见a3c_agent），存全局numpy以备急停
                    if (counter % FLAGS.snapshot_step == 1) or (counter in FLAGS.quicksave_step_list):
                        agent.save_model(SNAPSHOT, counter)
                        for i in range(PARALLEL):
                            np.save(
                                "./DataForAnalysis/low_reward_of_episode" + str(counter) + "thread" + str(i) + ".npy",
                                GL.get_value(i, "low_reward_of_episode"))
                            np.save(
                                "./DataForAnalysis/high_reward_of_episode" + str(counter) + "thread" + str(i) + ".npy",
                                GL.get_value(i, "high_reward_of_episode"))
                            np.save(
                                "./DataForAnalysis/low_reward_list_thread" + str(i) + "episode" + str(counter) + ".npy",
                                GL.get_value(i, "reward_low_list"))
                            np.save("./DataForAnalysis/high_reward_list_thread" + str(i) + "episode" + str(
                                counter) + ".npy",
                                    GL.get_value(i, "reward_high_list"))
                            np.save("./DataForAnalysis/victory_or_defeat_thread" + str(i) + "episode" + str(
                                counter) + ".npy",
                                    GL.get_value(i, "victory_or_defeat"))
                    if counter >= FLAGS.max_episodes:  # 超过设定的最大训练回合数后，退出循环（等于线程结束）
                        break

            elif is_done:  # 非训练模式下，回合结束的话，根据观测环境的结果在屏幕上打印出得分（is_done成立时的操作对比：
                obs = recorder[-1].observation  # 训练模式下Counter/counter会加一，会更新网络参数，有可能存储网络参数，不打印得分；
                score = obs["score_cumulative"][0]  # 非训练模式下不更新Counter/counter,不更新不存储网络参数，打印得分
                print('Your score is ' + str(score) + '!')

        if FLAGS.save_replay:  # 若设置为True，保存该线程的游戏回放
            env.save_replay(agent.name)
