def global_init(index):
    global global_var_dict
    global_var_dict = []

    for i in range(index):
        dict = {"num_frames": -9999, "ind_micro": -9999, "act_id_micro": -9999, "dir_high": -9999,
                "supply_num": -9999, "barrack_num": -9999, "barrack_location": [],
                "micro_isdone": [],
                "reward_low_list": [], "reward_high_list": [],
                "sum_high_reward": -9999, "sum_low_reward": -9999,
                "high_reward_of_episode": [], "low_reward_of_episode": [],
                "victory_or_defeat": [], "barrack_location_NotSure": [-99, -99]}
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
    global_var_dict[index][name].append(value)


def get_list():
    train_scv = [1, 2, 490, 3, 5]
    build_supply = [1, 3, 5, 91, 3, 5]
    build_barrack = [1, 3, 5, 42, 3, 5]
    train_marine = [1, 2, 477, 3, 5]
    # idle_worker = [6, 273]
    idle_worker = [6, 269]
    all_army_attack = [7, 13]
    list_actions = {0: train_scv, 1: build_supply, 2: build_barrack, 3: train_marine, 4: idle_worker,
                    5: all_army_attack}
    return list_actions, len(list_actions)
