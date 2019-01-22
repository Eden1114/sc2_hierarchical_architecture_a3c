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


tmp1 = np.load('./reward_list/rewards.npy')


plt.plot(np.arange(len(tmp1)), tmp1, 'b-')
# plt.plot(np.arange(len(tmp1)), tmp1, 'r-')
# plt.plot(np.arange(len(tmp50)), tmp50, 'g-')
# plt.plot(np.arange(len(tmp49)), tmp49, 'y-')
# plt.plot(np.arange(len(tmp44)), tmp44, 'm-')
# plt.plot(np.arange(len(tmp48)), tmp48, 'c-')
# plt.plot(np.arange(len(tmp39)), tmp39, 'k-')
# plt.plot(np.arange(len(tmp40)), tmp37, 'r-')
# plt.plot(np.arange(len(tmp41)), tmp38, 'g-')
# plt.plot(np.arange(len(tmp3)) + 1006, tmp3, 'b-')
plt.show()

