from os import O_SEQUENTIAL
import numpy as np 

def act(x):
    return 0 if x < 0.5 else 1 

def go(house,rock,attr):
    x = np.array([house,rock,attr])
    w11 = [0.3,0.3,0]
    w12 = [0.4,-0.5,1]
    weight1 = np.array([w11,w12])
    weight2 = np.array([-1,1])

    sum_hidden = np.dot(weight1,x)
    print('Neiro sum hidden layers  :' + str(sum_hidden))

    out_hidden = np.array([act(x) for x in sum_hidden])
    print("Value in exit neiros hidden layers:" + str(out_hidden))

    sum_end = np.dot(weight2,out_hidden)
    y = act(sum_end)
    print('Выходное значение HC:'+ str(y))

    return y 

house = 1
rock = 1 
attr = 1

res = go(house,rock,attr)

if res == 1:
    print('I love you')
else:
    print('I will call you')
