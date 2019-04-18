from pysc2.env import sc2_env
from absl import flags


def config_init():
    # 每次运行时设置：training，continuation，max_episodes，snapshot_step， render，parallel
    FLAGS = flags.FLAGS  # 定义超参数
    flags.DEFINE_bool("training", True, "Whether to train agents.")
    # flags.DEFINE_bool("training", False, "Whether to train agents.")    # 非训练模式，用于演示
    flags.DEFINE_bool("continuation", False, "Continuously training.")  # 重新训练，演示时应为False
    # flags.DEFINE_bool("continuation", True, "Continuously training.")    # 继续训练
    flags.DEFINE_float("learning_rate", 5e-4, "Learning rate for training.")
    flags.DEFINE_float("discount", 0.99, "Discount rate for future rewards.")
    flags.DEFINE_integer("max_episodes", int(1000), "Total episodes for training.")  # 训练的最大回合episode数
    flags.DEFINE_integer("snapshot_step", int(50), "Step for snapshot.")  # 存储snapshot快照和numpy数据的iter
    flags.DEFINE_list("quicksave_step_list", [10, 20, 30, 40, 60, 70, 80, 90], "Additional data-saving step list ")
    flags.DEFINE_string("snapshot_path", "./snapshot/", "Path for snapshot.")
    flags.DEFINE_string("log_path", "./log/", "Path for log.")
    # 这里的Device每个机器运行的时候都不一样，依据配置设定
    flags.DEFINE_string("device", "0", "Device for training.")
    flags.DEFINE_string("map", "Simple64", "Name of a map to use.")  # 2018/08/03: Simple64枪兵互拼新加代码
    flags.DEFINE_bool("render", False, "Whether to render with pygame.")
    flags.DEFINE_integer("screen_resolution", 64, "Resolution for screen feature layers.")
    flags.DEFINE_integer("minimap_resolution", 64, "Resolution for minimap feature layers.")
    flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")  # APM参数，step_mul为8相当于APM180左右
    flags.DEFINE_string("agent", "a3c_agent.A3CAgent", "Which agent to run.")
    # flags.DEFINE_string("net", "fcn", "atari or fcn.")
    flags.DEFINE_string("net", "hierarchical", "network architecture for logging")
    flags.DEFINE_string("agent_race", 'terran', "Agent's race.")
    # 2018/08/03: Simple64枪兵互拼新加代码
    flags.DEFINE_string("agent2", "Bot", "Second agent, either Bot or agent class.")
    flags.DEFINE_enum("agent2_race", "terran", sc2_env.Race._member_names_,  # pylint: disable=protected-access
                      "Agent 2's race.")
    flags.DEFINE_enum("difficulty", "very_easy", sc2_env.Difficulty._member_names_,  # pylint: disable=protected-access
                      "If agent2 is a built-in Bot, it's strength.")
    flags.DEFINE_integer("max_agent_steps", 4000, "Total agent steps.")  # 这里的step指的是回合episode里的那个step
    # 3000step = 18min游戏时间??   5000 = 21min??
    flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
    flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
    # 线程数的最佳值是4 @ 1080ti单卡 + i7 6700
    flags.DEFINE_integer("parallel", 4, "How many instances to run in parallel.")
    flags.DEFINE_bool("save_replay", False, "Whether to save a replay at the end.")
    return FLAGS
