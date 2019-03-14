import matplotlib.pyplot as plt
import numpy as np

"""
Python-Matplotlib 9 颜色和样式



1 颜色
　　八种内建默认颜色缩写
　　b: blue
　　g: green
　　r: red
　　c: cyan
　　m: magenta
　　y: yellow
　　k: black
　　w: white
　　其他颜色表示方法
　　　　灰色阴影
　　　　html 十六进制
　　　　RGB 元组

2 点、线的样式
　　23种点状态，注意不同点形状默认使用不同颜色
　　4种线形
　　　　- 实线
　　　　-- 虚线
　　　　-. 点划线
　　　　: 点线

3 式样字符串
　　可以将颜色，点型，线型写成一个字符串
　　　　cx--
　　　　mo:
　　　　kp-
"""

# tmp0 = np.load('./DataForAnalysis/victory_or_defeat_thread_0.npy')
# tmp1 = np.load('./DataForAnalysis/victory_or_defeat_thread_1.npy')
# tmp2 = np.load('./DataForAnalysis/victory_or_defeat_thread_2.npy')
# tmp3 = np.load('./DataForAnalysis/victory_or_defeat_thread_3.npy')
# tmp0 = np.load('./DataForAnalysis/victory_or_defeat_thread0episode1401.npy')
# tmp1 = np.load('./DataForAnalysis/victory_or_defeat_thread1episode1401.npy')
# tmp2 = np.load('./DataForAnalysis/victory_or_defeat_thread2episode1401.npy')
# tmp3 = np.load('./DataForAnalysis/victory_or_defeat_thread3episode1401.npy')
# tmp4 = np.load('./DataForAnalysis/victory_or_defeat_thread4.npy')
# tmp5 = np.load('./DataForAnalysis/victory_or_defeat_thread5.npy')
# tmp6 = np.load('./DataForAnalysis/victory_or_defeat_thread6.npy')
# tmp7 = np.load('./DataForAnalysis/victory_or_defeat_thread7.npy')

# tmp0 = np.load('./DataForAnalysis/high_reward_list_thread0.npy')
# tmp1 = np.load('./DataForAnalysis/high_reward_of_episode1thread0.npy')
# tmp2 = np.load('./DataForAnalysis/high_reward_of_episode51thread0.npy')
# tmp3 = np.load('./DataForAnalysis/high_reward_of_episode101thread0.npy')
# tmp4 = np.load('./DataForAnalysis/high_reward_of_episode151thread0.npy')
# tmp0 = np.load('./DataForAnalysis/low_reward_list_thread0.npy')
# tmp1 = np.load('./DataForAnalysis/low_reward_of_episode1thread0.npy')
# tmp2 = np.load('./DataForAnalysis/low_reward_of_episode51thread0.npy')
# tmp3 = np.load('./DataForAnalysis/low_reward_of_episode101thread0.npy')
# tmp4 = np.load('./DataForAnalysis/low_reward_of_episode151thread0.npy')
# tmp0 = np.load('./DataForAnalysis/high_reward_list_thread0.npy')
# tmp1 = np.load('./DataForAnalysis/high_reward_list_thread1.npy')
# tmp2 = np.load('./DataForAnalysis/high_reward_list_thread2.npy')
# tmp3 = np.load('./DataForAnalysis/high_reward_list_thread3.npy')
# tmp0 = np.load('./DataForAnalysis/high_reward_list_thread0episode1.npy')
# tmp1 = np.load('./DataForAnalysis/high_reward_list_thread1episode1.npy')
# tmp2 = np.load('./DataForAnalysis/high_reward_list_thread2episode1.npy')
# tmp3 = np.load('./DataForAnalysis/high_reward_list_thread3episode1.npy')
# tmp4 = np.load('./DataForAnalysis/high_reward_list_thread4.npy')
# tmp0 = np.load('./DataForAnalysis/low_reward_list_thread0.npy')
# tmp1 = np.load('./DataForAnalysis/low_reward_list_thread1.npy')
# tmp2 = np.load('./DataForAnalysis/low_reward_list_thread2.npy')
# tmp3 = np.load('./DataForAnalysis/low_reward_list_thread3.npy')
# tmp4 = np.load('./DataForAnalysis/low_reward_list_thread4.npy')

# tmp0 = np.load('./DataForAnalysis/high_reward_of_episode151thread0.npy')
# tmp1 = np.load('./DataForAnalysis/high_reward_of_episode201thread0.npy')
# tmp2 = np.load('./DataForAnalysis/high_reward_of_episode251thread0.npy')
# tmp3 = np.load('./DataForAnalysis/high_reward_of_episode301thread0.npy')
# tmp4 = np.load('./DataForAnalysis/high_reward_of_episode351thread0.npy')
# tmp0 = np.load('./DataForAnalysis/high_reward_of_episode151thread0.npy')
# tmp1 = np.load('./DataForAnalysis/high_reward_of_episode151thread1.npy')
# tmp2 = np.load('./DataForAnalysis/high_reward_of_episode151thread2.npy')
# tmp3 = np.load('./DataForAnalysis/high_reward_of_episode151thread3.npy')
# tmp4 = np.load('./DataForAnalysis/high_reward_of_episode151thread4.npy')

tmp0 = np.load('./DataForAnalysis/victory_or_defeat_thread_4.npy')
# tmp1 = np.load('./DataForAnalysis/reward_list_thread_4.npy')
# tmp2 = np.load('./DataForAnalysis/episode_score_list_thread_4.npy')

print(len(tmp0))
# print(tmp1)
# print(tmp2)
# print(tmp3)
# print(tmp4)
# print(tmp5)
# print(tmp6)
# print(tmp7)

plt.plot(np.arange(len(tmp0)), tmp0, 'b-')
# plt.plot(np.arange(len(tmp1)), tmp1, 'r-')
# plt.plot(np.arange(len(tmp2)), tmp2, 'g-')
# plt.plot(np.arange(len(tmp3)), tmp3, 'y-')
# plt.plot(np.arange(len(tmp4)), tmp4, 'm-')
# plt.plot(np.arange(len(tmp3)), tmp3, 'r-')
# plt.plot(np.arange(len(tmp50)), tmp50, 'g-')
# plt.plot(np.arange(len(tmp49)), tmp49, 'y-')
# plt.plot(np.arange(len(tmp44)), tmp44, 'm-')
# plt.plot(np.arange(len(tmp48)), tmp48, 'c-')
# plt.plot(np.arange(len(tmp39)), tmp39, 'k-')
# plt.plot(np.arange(len(tmp40)), tmp37, 'r-')
# plt.plot(np.arange(len(tmp41)), tmp38, 'g-')
# plt.plot(np.arange(len(tmp3)) + 1006, tmp3, 'b-')
plt.show()
