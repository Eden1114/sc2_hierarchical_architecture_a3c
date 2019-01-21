from pysc2.lib import actions
import agents.globalvar as GL

list_actions, _ = GL.get_list()

def action_micro(target_0, target_1, dir_high, ind_todo):

    macro_type = 0 #宏动作类型：0防御，1进攻
    coord_type = 0 #坐标类型： 0 screen，1 minimap
    call_step_low = 1 #是否调用low_net，给update用
    print("macro_actions", dir_high, ind_todo)
    # act_id = list_actions[dir_high][ind_todo]
    act_id = list_actions[dir_high][ind_todo]

    act_args = []
    for arg in actions.FUNCTIONS[act_id].args:  # actions是pysc2.lib中的文件 根据act_id获取其可使用的参数，并添加到args中去
      if arg.name in ('screen', 'minimap', 'screen2'):
        act_args.append([target_1, target_0])
      else:
        act_args.append([0])  # TODO: Be careful
    action = [actions.FunctionCall(act_id, act_args)]

    # train_scv里固定写好坐标的动作
    if dir_high == 0 and ind_todo == 0:
        action = [actions.FunctionCall(function=1, arguments=[[20, 25]])]  # 移动镜头到标准位置
        call_step_low = 0
    elif dir_high == 0 and ind_todo == 1:
        action = [actions.FunctionCall(function=2, arguments=[[0], [30, 24]])]  # 选中基地
        call_step_low = 0
    elif dir_high == 0 and ind_todo == 3:
        action = [actions.FunctionCall(function=3, arguments=[[0], [5, 5], [30, 25]])]  # 框中一些scv
        call_step_low = 0

    # build_supply里固定写好坐标的动作
    elif dir_high == 1 and ind_todo == 0:
        action = [actions.FunctionCall(function=1, arguments=[[20, 25]])]  # 移动镜头到标准位置
        call_step_low = 0
    elif dir_high == 1 and ind_todo == 1:
        action = [actions.FunctionCall(function=3, arguments=[[0], [5, 5], [30, 25]])]  # 框中一些scv
        call_step_low = 0
    # elif dir_high == 1 and ind_todo == 4:
    #     action = [actions.FunctionCall(function=1, arguments=[[20, 25]])]  # 移动镜头到标准位置
    #     call_step_low = 0
    elif dir_high == 1 and ind_todo == 4:
        action = [actions.FunctionCall(function=3, arguments=[[0], [5, 5], [30, 25]])]  # 框中一些scv
        call_step_low = 0

    # build_barrack里固定写好坐标的动作
    elif dir_high == 2 and ind_todo == 0:
        action = [actions.FunctionCall(function=1, arguments=[[20, 25]])]  # 移动镜头到标准位置
        call_step_low = 0
    elif dir_high == 2 and ind_todo == 1:
        action = [actions.FunctionCall(function=3, arguments=[[0], [5, 5], [30, 25]])]  # 框中一些scv
        call_step_low = 0
    # elif dir_high == 2 and ind_todo == 4:
    #     action = [actions.FunctionCall(function=1, arguments=[[20, 25]])]  # 移动镜头到标准位置
    #     call_step_low = 0
    elif dir_high == 2 and ind_todo == 4:
        action = [actions.FunctionCall(function=3, arguments=[[0], [5, 5], [30, 25]])]  # 框中一些scv
        call_step_low = 0

    # train_marine里固定写好坐标的动作
    elif dir_high == 3 and ind_todo == 0:
        action = [actions.FunctionCall(function=1, arguments=[[20, 25]])]  # 移动镜头到标准位置
        call_step_low = 0
    # elif dir_high == 3 and ind_todo == 1:
    #     action = [actions.FunctionCall(function=2, arguments=[[0], [32, 32]])]  # 选中spatial_action算出来的“绝对位置”（希望是兵营）
    #     call_step_low = 0
    # elif dir_high == 3 and ind_todo == 3:
    #     action = [actions.FunctionCall(function=1, arguments=[[20, 25]])]  # 移动镜头到标准位置
    #     call_step_low = 0
    elif dir_high == 3 and ind_todo == 3:
        action = [actions.FunctionCall(function=3, arguments=[[0], [5, 5], [30, 25]])]  # 框中一些scv
        call_step_low = 0

    #all_army_attack改macro_type和coord_type
    elif dir_high == 5:
        macro_type = 1
        coord_type = 1

    print("action = ", action)
    # return action
    return action, call_step_low,macro_type,coord_type