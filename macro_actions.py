from pysc2.lib import actions
import globalvar as GL
import random

list_actions, _ = GL.get_list()


def action_micro(ind_thread, dir_high, ind_todo):
    macro_type = 0  # 宏动作类型：0防御，1进攻
    coord_type = 0  # 坐标类型： 0 screen，1 minimap
    call_step_low = False
    act_id = list_actions[dir_high][ind_todo]
    # print("macro_actions", dir_high, ind_todo)

    # train_scv里固定写好坐标的动作
    if dir_high == 0 and ind_todo == 0:
        action = [actions.FunctionCall(function=1, arguments=[[20, 25]])]  # 移动镜头到标准位置
    elif dir_high == 0 and ind_todo == 1:
        action = [actions.FunctionCall(function=2, arguments=[[0], [30, 24]])]  # 选中基地
    elif dir_high == 0 and ind_todo == 3:
        action = [actions.FunctionCall(function=3, arguments=[[0], [5, 5], [30, 25]])]  # 框中一些scv

    # build_supply里固定写好坐标的动作
    elif dir_high == 1 and ind_todo == 0:
        action = [actions.FunctionCall(function=1, arguments=[[20, 25]])]  # 移动镜头到标准位置
    elif dir_high == 1 and ind_todo == 1:
        action = [actions.FunctionCall(function=3, arguments=[[0], [5, 5], [30, 25]])]  # 框中一些scv
    # elif dir_high == 1 and ind_todo == 4:
    #     action = [actions.FunctionCall(function=1, arguments=[[20, 25]])]  # 移动镜头到标准位置
    elif dir_high == 1 and ind_todo == 4:
        action = [actions.FunctionCall(function=3, arguments=[[0], [5, 5], [30, 25]])]  # 框中一些scv

    # build_barrack里固定写好坐标的动作
    elif dir_high == 2 and ind_todo == 0:
        action = [actions.FunctionCall(function=1, arguments=[[20, 25]])]  # 移动镜头到标准位置
    elif dir_high == 2 and ind_todo == 1:
        action = [actions.FunctionCall(function=3, arguments=[[0], [5, 5], [30, 25]])]  # 框中一些scv
    # elif dir_high == 2 and ind_todo == 4:
    #     action = [actions.FunctionCall(function=1, arguments=[[20, 25]])]  # 移动镜头到标准位置
    elif dir_high == 2 and ind_todo == 4:
        action = [actions.FunctionCall(function=3, arguments=[[0], [5, 5], [30, 25]])]  # 框中一些scv

    # train_marine里固定写好坐标的动作
    elif dir_high == 3 and ind_todo == 0:
        action = [actions.FunctionCall(function=1, arguments=[[20, 25]])]  # 移动镜头到标准位置
    # elif dir_high == 3 and ind_todo == 1:
    #     action = [actions.FunctionCall(function=2, arguments=[[0], [32, 32]])]  # 选中spatial_action算出来的“绝对位置”（希望是兵营）
    elif dir_high == 3 and ind_todo == 1:
        temp = GL.get_value(ind_thread, "barrack_location")  # 可能有[0,0]
        tem = []  # 去除[0,0]，实际决策的列表空间
        for test in temp:
            if test == [0, 0]:
                continue
            else:
                tem.append(test)
        if len(tem) >= 1:
            LocationToApply = random.sample(tem, 1)[0]  # 随机选取其中的一个坐标
        else:
            LocationToApply = [random.randint(0, 63), random.randint(0, 63)]
        action = [actions.FunctionCall(function=2, arguments=[[0],
                                                              LocationToApply])]  # 随机选择全局变量barrack_location中存储的一个坐标（希望是之前成功建好的&目前健在的兵营）
        print("Thread ", ind_thread, end=" ")
        print("Train_marine: ", action, end=" ")
        print("Barrack_location: ", tem)
    # elif dir_high == 3 and ind_todo == 3:
    #     action = [actions.FunctionCall(function=1, arguments=[[20, 25]])]  # 移动镜头到标准位置
    elif dir_high == 3 and ind_todo == 3:
        action = [actions.FunctionCall(function=3, arguments=[[0], [5, 5], [30, 25]])]  # 框中一些scv

    else:
        if dir_high == 5:
            # 更改宏动作类型和坐标类型
            macro_type = 1
            coord_type = 1
        act_args = []
        for arg in actions.FUNCTIONS[act_id].args:  # actions是pysc2.lib中的文件 根据act_id获取其可使用的参数，并添加到args中去
            if arg.name in ('screen', 'minimap', 'screen2'):
                call_step_low = True
            else:
                act_args.append([0])  # TODO: Be careful
                action = [actions.FunctionCall(act_id, act_args)]

    # print("action = ", action)
    # return action
    return action, call_step_low, act_id, macro_type, coord_type  # 如果call_step_low不为1，则返回的action不会被采用；如果call_step_low为1，则返回的action会被采用
