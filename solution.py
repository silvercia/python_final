import numpy as np
from numpy import array, vstack, shape, dot, ones, transpose, sqrt, delete


linelist = list()
with open('data.txt') as data:
    for line in data:
        linelist.append(list(map(float, line.split())))
    # linelist: [sample 1, sample 2,...,sample m]
    x = array(linelist[0])
    x.fill(0)
    # x1 = array([linelist[1]])
    # x1.fill(0)
    # x2 = array([linelist[2]])
    # x2.fill(0)
    # x: the only usage of x is to make concatenate possible
    for sample in linelist:
        a = array(sample)
        x = vstack((x,a))
    # x = matrix[0; sample 1; sample2; ...; sample m] (507x14)
    # for sample in linelist:
    #     if sample[3] ==0.0:
    #         b = array([sample])
    #         x1 = vstack((x1,b))
    #     else:
    #         c = array([sample])
    #         x2 = vstack((x2,c))


x = x[1:]
# x1 = x1[1:]
# x2 = x2[1:]
# x1 = delete(x1,3,1)
# x2 = delete(x2,3,1)
# delete the first row, whose values are all 0. x:(506x14)
column = shape(x)[1]-1
# column1 = shape(x1)[1]-1
# column2 = shape(x2)[1]-1
# colunm: the last colunm in x (13)
y = x[:,column]
# y1 = x1[:,column1]
# y2 = x2[:,column2]
# y: the last row extracted, which is the house price.

# TODO: implement
max = x.max(axis=0)
min = x.min(axis=0)
avg = x.sum(axis=0)/shape(x)[0]
avg[3]=0
x = (x-avg)/(max-min)
avgd = x.sum(axis=0)/shape(x)[0]
xd = (x-avgd)**2
d = sqrt(xd.sum(axis=0)/shape(x)[0])
up = avgd+3*d
up[3]=2
lb = avgd-3*d
lb[3]=-1
for i in range(shape(x)[1]):
    for j in range(shape(x)[0]):
        if x[j,i]<lb[i] or x[j,i]>up[i]:
            x[j]=0
x = x[~np.all(x == 0, axis=1)]
# x1 = (x1-avg)/(max-min)
# x2 = (x2-avg)/(max-min)
# normalize the data

x = transpose(x)
# x1 = transpose(x1)
# x2 = transpose(x2)
# transpose. After that: (14x506)
row = shape(x)[0]-1
# row1 = shape(x1)[0]-1
# row2 = shape(x2)[0]-1
x = x[:row]
# x1 = x1[:row1]
# x2 = x2[:row2]
# delete the last row from x. x: (13x506)
z = ones(shape(x)[1])
# z1 = ones(shape(x1)[1])
# z2 = ones(shape(x2)[1])
# z: a new row whose value are all 1. z: (1x506)
x = vstack((z,x))
# x1 = vstack((z1,x1))
# x2 = vstack((z2,x2))
# x0 = delete(x,4,0)
# CHAS = transpose(np.ones(shape(x0))*x[4])
# add z into the first row of x. x: (14x506)
theta = dot(dot(np.linalg.inv(dot(x,transpose(x))),x),y)
# theta1 = dot(dot(np.linalg.inv(dot(x1,transpose(x1))),x1),y1)
# theta2 = dot(dot(np.linalg.inv(dot(x2,transpose(x2))),x2),y2)
# y0 = dot(CHAS*(theta2-theta1)+theta1,x0)
# y0 = np.diag(y0)
# theta0 = dot(dot(np.linalg.inv(dot(x,transpose(x))),x),y0)
# solve theta by normal equation.
print(theta)
# print(theta1)
# print(theta2)
# print(theta0)

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


# rup1 = 0
# for i in range(0,shape(x1)[1]):
#     rup1 += (y1[i]-dot(theta1,x1[:,i]))**2
#
# rblw1 =0
# avgy1 = (np.sum(y1))/shape(y1)[0]
# for j in range(0,shape(x1)[1]):
#     rblw1 +=(y1[i]-avgy1)**2
#
# r1= 1-rup1/rblw1
# print(r1)
#
#
# rup2 = 0
# for i in range(0,shape(x2)[1]):
#     rup2 += (y2[i]-dot(theta2,x2[:,i]))**2
#
# rblw2 =0
# avgy2 = (np.sum(y2))/shape(y2)[0]
# for j in range(0,shape(x2)[1]):
#     rblw2 +=(y2[i]-avgy2)**2
#
# r2= 1-rup2/rblw2
# print(r2)

