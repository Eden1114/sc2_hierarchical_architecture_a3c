from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from pysc2.lib import actions
from agents.macro_actions import action_micro
import agents.globalvar as GL

list_actions, _ = GL.get_list()

# 该函数的作用是使agent和环境进行交互（根据环境选动作，有了动作后更新环境）
def run_loop(agents, env, max_frames, ind_thread):  # agents是列表，里面有一个agent（A3CAgent对象），env是SC2Env对象经过处理后的变量，max_frames是回合内最多进行的step数
  """A run loop to have agents and an environment interact."""
  start_time = time.time()

  try:
    while True:   # 底下发生的是一个回合内的过程
      GL.set_value(ind_thread, "ind_micro", -1)
      num_frames = 0  # 计算回合里的step数
      GL.set_value(ind_thread, "supply_num",0)  #每局游戏需要用的全局变量清空
      GL.set_value(ind_thread, "barrack_num", 0)
      GL.set_value(ind_thread, "brrack_location", [])
      GL.set_value(ind_thread, "sum_high_reward", 0)
      GL.set_value(ind_thread, "sum_low_reward", 0)
      timesteps = env.reset()
      # reset函数返回TimeStep四元组（sc2_env.py 512行），包含的信息有4种，在知乎上PySC2详解的文章里有介绍

      for a in agents:  # 疑问：明明agents里只有一个agent对象，为啥要写成循环？或者说为啥一开始设计的参数要是agents(列表),而不是直接agent对象
        a.reset()
      while True:   # 底下发生的是回合内一步的过程
        ind_last = GL.get_value(ind_thread, "ind_micro")
        num_frames += 1
        GL.set_value(ind_thread,"num_frames",num_frames )
        last_timesteps = timesteps
        # actions = [agent.step(timestep) for agent, timestep in zip(agents, timesteps)]      # 关键一步，调用了agent对象的step方法计算出选择的action。

        #DHN add:
        # print("ind_last", ind_last)
        if ind_last == -1 or ind_last == -99 or ind_last == 666:
                          # ind_last == -99 (表示宏动作里的微动作执行失败)
                          # ind_last == 666 (表示宏动作成功执行完毕）:
          dir_high = [agent.step_high(timestep) for agent, timestep in zip(agents, timesteps)]  # dir_high是 要执行的宏动作id（从0开始）
          GL.set_value(ind_thread, "dir_high", dir_high[0])
          GL.set_value(ind_thread, "ind_micro", 0)
          ind_todo = GL.get_value(ind_thread, "ind_micro")
        else:
          temp = GL.get_value(ind_thread, "ind_micro")
          GL.set_value(ind_thread, "ind_micro", temp+1)
          ind_todo = GL.get_value(ind_thread, "ind_micro")

        dir_high = GL.get_value(ind_thread, "dir_high")
        action, call_step_low, act_id, macro_type, coord_type = action_micro(dir_high, ind_todo)
        # 如果call_step_low为False，则act_id没用了，直接使用上行中的action
        # 如果其为True，则进入以下的模块，action没用了，act_id被使用来计算新的action

        if call_step_low == True:
          GL.set_value(ind_thread, "act_id_micro", ind_todo)
          target_pack = [agent.step_low(ind_thread, timestep, dir_high, ind_todo) for agent, timestep in zip(agents, timesteps)]
          # target_pack = [agent.step_low(ind_thread, timestep) for agent, timestep in zip(agents, timesteps)]
          target_0 = target_pack[0][0]
          target_1 = target_pack[0][1]
          act_args = []
          for arg in actions.FUNCTIONS[act_id].args:  # actions是pysc2.lib中的文件 根据act_id获取其可使用的参数，并添加到args中去
            if arg.name in ('screen', 'minimap', 'screen2'):
              act_args.append([target_1, target_0])
            else:
              act_args.append([0])  # TODO: Be careful
            action = [actions.FunctionCall(act_id, act_args)]

        # 校验：
        flag_success = True
        if list_actions[dir_high][ind_todo] not in last_timesteps[0].observation['available_actions']:
          GL.set_value(ind_thread, "ind_micro", -99)  # 表示宏动作里的微动作执行失败
          action = [actions.FunctionCall(function=0, arguments=[])] # 执行no_op
          flag_success = False

        if flag_success:
          GL.add_value_list(ind_thread, "micro_isdone", 1)
        else:
          GL.add_value_list(ind_thread, "micro_isdone", -1)

        # 当ind_todo是最后一个需要执行的动作，且执行成功时，将ind_done[ind_thread]设为666（即宏动作成功执行完毕）
        if ind_todo == len(list_actions[dir_high])-1 and flag_success:
          GL.set_value(ind_thread, "ind_micro", 666)  # 表示宏动作执行到了最后一步微动作且执行成功

        timesteps = env.step(action)   # env环境的step函数根据动作计算出下一个timesteps

        # 如下模块表示：动作函数合法但失败（比如造补给站在available_action_list里，但选的建造坐标在基地的位置上，则造不出来）
        # 则将ind_micro置为-99，表示“宏动作执行失败”
        if not len(timesteps[0].observation.last_actions) and action[0].function != 0:
          GL.set_value(ind_thread, "ind_micro", -99)

        # Only for a single player!
        is_done = (num_frames >= max_frames) or timesteps[0].last()   # timesteps[0]是timesteps的第一个变量step_type（状态类型），last()为True即到了末状态
        yield [last_timesteps[0], action[0], timesteps[0]], is_done, num_frames, call_step_low, macro_type, coord_type
        # yield适用于函数返回内容较多，占用内存量很大的情况。可以看成返回了一个列表（实际不是）
        # 详解见http://www.runoob.com/w3cnote/python-yield-used-analysis.html
        if is_done:
          break
  except KeyboardInterrupt:
    pass
  finally:
    elapsed_time = time.time() - start_time
    print("Took %.3f seconds" % elapsed_time)
