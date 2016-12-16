import numpy as np
from numpy import array, vstack, shape, dot, ones, transpose, sqrt

linelist = list()
with open('data.txt') as data:
    for line in data:
        linelist.append(list(map(float, line.split())))
    # linelist: [sample 1, sample 2,...,sample m]
    x = array(linelist[0])
    x.fill(0)
    # x: the only usage of x is to make concatenate possible
    for sample in linelist:
        a = array(sample)
        x = vstack((x,a))
    # x = matrix[0; sample 1; sample2; ...; sample m] (507x14)

x = x[1:]
# delete the first row, whose values are all 0. x:(506x14)
column = shape(x)[1]-1
# colunm: the last colunm in x (13)

# TODO: implement
max = x.max(axis=0)
min = x.min(axis=0)
ran = max-min
ran[column]=1
avg = x.sum(axis=0)/shape(x)[0]
avg[3]=0
avg[column]=0
x = (x-avg)/ran
# normalize the data

avgd = x.sum(axis=0)/shape(x)[0]
xd = (x-avgd)**2
d = sqrt(xd.sum(axis=0)/shape(x)[0])
up = avgd+3*d
up[3]=2
lb = avgd-3*d
lb[3]=-1
for i in range(shape(x)[1]-1):
    for j in range(shape(x)[0]):
        if x[j,i]<lb[i] or x[j,i]>up[i]:
            x[j]=0
x = x[~np.all(x == 0, axis=1)]
# delete derivated values

x = transpose(x)
# transpose. After that: (14,***)
row = shape(x)[0]-1
y = x[row]
# y: the last row extracted, which is the house price.
x = x[:row]
# delete the last row from x. x: (13,***)
z = ones(shape(x)[1])
# z: a new row whose value are all 1. z: (1x***)
x = vstack((z,x))
# add z into the first row of x. x: (14x***)
theta = dot(dot(np.linalg.inv(dot(x,transpose(x))),x),y)
# solve theta by normal equation.
print(theta)

# And then we calculate r^2
rup = 0
for i in range(0,shape(x)[1]):
    rup += (y[i]-dot(theta,x[:,i]))**2

rblw =0
avgy = (np.sum(y))/shape(y)[0]
for j in range(0,shape(x)[1]):
    rblw +=(y[i]-avgy)**2

r= 1-(rup/rblw)
print(r)
