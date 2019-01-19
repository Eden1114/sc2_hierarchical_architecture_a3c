def global_init(index):
    global ind_done
    ind_done = []
    for i in range(index):
        ind_done.append([-1, -1])  # 第一个数表示当前线程的dir_hig, 第二个数表示该线程宏动作(dir_hig)中“被选择后要执行的”微动作的序号

    global global_var_dict
    global_var_dict = []

    for i in range(index):
        dict = {"num_frames": -9999, "ind_done": -9999, "supply_num": -9999, "barrack_num": -9999, "brrack_location": [],
                "micro_isdone":[]}
        # micro_isdone 成功是1，失败是-1
        global_var_dict.append(dict)

def set_value(index, name, value):
    global_var_dict[index][name] = value

def get_value(index, name):
    try:
        return global_var_dict[index][name]
    except KeyError:
        print("取全局变量时输入的参数有误！")

def add_value_list(index, name, value):
    # global_var_dict[index]["micro_isdone"].append(value)
    global_var_dict[index][name].append(value)

def Set_value(index, value):
    ind_done[index][1] = value

def set_value_dir_high(index, value):
    ind_done[index][0] = value

def Get_value(index):
    return ind_done[index][1]

def get_value_dir_high(index):
    return ind_done[index][0]

def get_list():
    train_scv = [1, 2, 490, 3, 5]
    build_supply = [1, 3, 5, 91, 3, 5]
    build_barrack = [1, 3, 5, 42, 3, 5]
    train_marine = [1, 2, 477, 3, 5]
    idle_worker = [6, 273]
    all_army_attack = [7, 13]
    list_actions = {0: train_scv, 1: build_supply, 2: build_barrack, 3: train_marine, 4: idle_worker, 5: all_army_attack}
    return list_actions, len(list_actions)
