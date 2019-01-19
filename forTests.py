from __future__ import division
import numpy as np
import random
#foo = ['a', 'b', 'c', 'd', 'e']

# foo = np.array([1,2,3,4,5])
# filter = np.array([0, 5, 2])
# print( np.argmax(filter) )
# print(random.randint(0,5))
# print(np.arange(0, 10)[ np.newaxis, :])

# a = np.array([[0.2,0.8],[0.4,0.6]])
# b = np.array([[0,1],[1,0]])
# print(a*b)
#
# print(np.argmax([9, 6, 7]))
#
# train_scv = [1, 2, 490]
# list_actions = {0:train_scv}
#
# print( random.randint(0,0) )
#
# x = [(20, 15)]
# print(x[0][0])
# print(x[0][1])
#
# for i in range(3 - 1):
#     print(i)
#
# from pysc2.lib import actions
# print( actions.FUNCTIONS[2].args )
#
# list_actions = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1}
# print ( "list_actions", len(list_actions) )
#
# g = []
# dict = {"num_frames": -95559, "ind_done": -9999, "supply_num": -9999, "barrack_num": -9999, "brrack_location": []}
# g.append(dict)
#
# ddd = [8, 8, 0]
#
# print("ddd = ", )
#
# print("thread%d supply_num is %d" % (5,7) )


import random

for i in range(100000000000):
    t = random.randint(0,262143)
    a = t//4096
    b = t%4096
    x_s = a//8
    y_s = a%8
    x_m = b//64
    y_m = b%64
    if x_s >7 or y_s >7 or x_m >63 or y_m >63:
        print('x_s', x_s)
        print('y_s', y_s)
        print('x_m', x_m)
        print('y_m', y_m)