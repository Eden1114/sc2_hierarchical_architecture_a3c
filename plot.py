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

tmp0 = np.load('./DataForAnalysis/victory_or_defeat_parallel0.npy')
tmp1 = np.load('./DataForAnalysis/victory_or_defeat_parallel1.npy')
tmp2 = np.load('./DataForAnalysis/victory_or_defeat_parallel2.npy')
tmp3 = np.load('./DataForAnalysis/victory_or_defeat_parallel3.npy')
# tmp4 = np.load('./DataForAnalysis/victory_or_defeat_parallel4.npy')
# tmp5 = np.load('./DataForAnalysis/victory_or_defeat_parallel5.npy')
# tmp6 = np.load('./DataForAnalysis/victory_or_defeat_parallel6.npy')
# tmp7 = np.load('./DataForAnalysis/victory_or_defeat_parallel7.npy')

# tmp0 = np.load('./DataForAnalysis/high_reward_list_parallel0.npy')
# tmp1 = np.load('./DataForAnalysis/high_reward_of_episode1parallel0.npy')
# tmp2 = np.load('./DataForAnalysis/high_reward_of_episode51parallel0.npy')
# tmp3 = np.load('./DataForAnalysis/high_reward_of_episode101parallel0.npy')
# tmp4 = np.load('./DataForAnalysis/high_reward_of_episode151parallel0.npy')
# tmp0 = np.load('./DataForAnalysis/low_reward_list_parallel0.npy')
# tmp1 = np.load('./DataForAnalysis/low_reward_of_episode1parallel0.npy')
# tmp2 = np.load('./DataForAnalysis/low_reward_of_episode51parallel0.npy')
# tmp3 = np.load('./DataForAnalysis/low_reward_of_episode101parallel0.npy')
# tmp4 = np.load('./DataForAnalysis/low_reward_of_episode151parallel0.npy')
# tmp0 = np.load('./DataForAnalysis/high_reward_list_parallel0.npy')
# tmp1 = np.load('./DataForAnalysis/high_reward_list_parallel1.npy')
# tmp2 = np.load('./DataForAnalysis/high_reward_list_parallel2.npy')
# tmp3 = np.load('./DataForAnalysis/high_reward_list_parallel3.npy')
# tmp4 = np.load('./DataForAnalysis/high_reward_list_parallel4.npy')
# tmp0 = np.load('./DataForAnalysis/low_reward_list_parallel0.npy')
# tmp1 = np.load('./DataForAnalysis/low_reward_list_parallel1.npy')
# tmp2 = np.load('./DataForAnalysis/low_reward_list_parallel2.npy')
# tmp3 = np.load('./DataForAnalysis/low_reward_list_parallel3.npy')
# tmp4 = np.load('./DataForAnalysis/low_reward_list_parallel4.npy')

# tmp0 = np.load('./DataForAnalysis/high_reward_of_episode151parallel0.npy')
# tmp1 = np.load('./DataForAnalysis/high_reward_of_episode201parallel0.npy')
# tmp2 = np.load('./DataForAnalysis/high_reward_of_episode251parallel0.npy')
# tmp3 = np.load('./DataForAnalysis/high_reward_of_episode301parallel0.npy')
# tmp4 = np.load('./DataForAnalysis/high_reward_of_episode351parallel0.npy')
# tmp0 = np.load('./DataForAnalysis/high_reward_of_episode151parallel0.npy')
# tmp1 = np.load('./DataForAnalysis/high_reward_of_episode151parallel1.npy')
# tmp2 = np.load('./DataForAnalysis/high_reward_of_episode151parallel2.npy')
# tmp3 = np.load('./DataForAnalysis/high_reward_of_episode151parallel3.npy')
# tmp4 = np.load('./DataForAnalysis/high_reward_of_episode151parallel4.npy')
# print(tmp0)
# print(tmp1)
# print(tmp2)
# print(tmp3)
# print(tmp4)
# print(tmp5)
# print(tmp6)
# print(tmp7)

plt.plot(np.arange(len(tmp0)), tmp0, 'b-')
plt.plot(np.arange(len(tmp1)), tmp1, 'r-')
plt.plot(np.arange(len(tmp2)), tmp2, 'g-')
plt.plot(np.arange(len(tmp3)), tmp3, 'y-')
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
