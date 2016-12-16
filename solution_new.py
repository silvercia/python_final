import numpy as np
from numpy import array, vstack, shape, dot, ones, transpose, sqrt

## Begin: read the data into a matrix
linelist = list()
with open('data.txt') as data:
    for line in data:
        linelist.append(list(map(float, line.split())))
    # linelist: [sample 1, sample 2,...,sample m]
    x = array(linelist[0])
    x.fill(0)
    # x: the only usage of x is to make vstack work
    for sample in linelist:
        a = array(sample)
        x = vstack((x,a))
    # x = matrix[0;
        #        sample 1;
        #        sample 2;
        #          ...;
        #        sample m] (507,14)
x0 = x[1:]   # original data
x = x[1:]
# delete the first row, whose values are all 0, so that only the real data from 'data.txt' is left. x:(506,14)
## End: read the data into a matrix


## Begin: normalizing the data (seems no use...)
# column = shape(x)[1]-1
# # column: the last column in x, which is 13.
# max = x.max(axis=0)
# min = x.min(axis=0)
# ran = max-min
# avg = x.sum(axis=0)/shape(x)[0]
# # calculate the range and average of every feature.
# avg[3]=0
# # for the categorical feature, we have to maintain its value, which is either 0 or 1.
# avg[column]=0
# ran[column]=1
# # the last column is the price of houses, which cannot be changed.
# # so we do this to maintain its value.
# x = (x-avg)/ran
# x0 = (x0-avg)/ran
## Eed: normalizing the data (seems no use...)

## Begin: deleting the noises (seems no use... descreases R instead)
# avgd = x.sum(axis=0)/shape(x)[0]
# xd = (x-avgd)**2
# d = sqrt(xd.sum(axis=0)/shape(x)[0])
# # calculate the standard deviation of each feature.
# up = avgd+3*d
# up[3]=2
# lb = avgd-3*d
# lb[3]=-1
# # calculate the bound for noise (noise: data beyond 3-sigma).
# for i in range(shape(x)[1]-1):
#     for j in range(shape(x)[0]):
#         if x[j,i]<lb[i] or x[j,i]>up[i]:
#             x[j]=0
# x = x[~np.all(x == 0, axis=1)]
# delete the noises.
## Bnd: deleting the noises (seems no use... descreases R instead)

## Begin: find our ys and do something on x to make calculating weights and R-square possible.
x = transpose(x)
# transpose. After that: (14,***)
row = shape(x)[0]-1
# row: the last row of x.
y = x[row]
# y: the last row of x, which is the house price.
x = x[:row]
# delete the last row from x. x: (13,***)
z = ones(shape(x)[1])
x = vstack((z,x))
# add a new row whose values are all 1, into the first row of x. x: (14,***).
# this makes calculating weights possible.

# and then we do the similar things for x0.
x0 = transpose(x0)
# transpose. After that: (14,506)
row0 = shape(x0)[0]-1
# row: the last row of x0.
y0 = x0[row0]
# y0: the last row of x0, which is the house price for all samples.
x0 = x0[:row0]
# delete the last row from x0. x0: (13,506)
z0 = ones(shape(x0)[1])
x0 = vstack((z0,x0))
# add a new row whose values are all 1, into the first row of x0. x0: (14,506).
# this makes calculating R-square possible.
## End: find our ys and do something on x to make calculating weights and R-square possible.

## Begin: calculate weights
theta = dot(dot(np.linalg.inv(dot(x,transpose(x))),x),y)
# solve theta by normal equation.
print(theta)
## End: calculate weights

## Begin: evaluating our weights by R-square.
r_numerator = 0
for i in range(0,shape(x0)[1]):
    r_numerator += (y0[i]-dot(theta,x0[:,i]))**2

r_denominator =0
avgy = (np.sum(y0))/shape(y0)[0]
for j in range(0,shape(x0)[1]):
    r_denominator +=(y0[i]-avgy)**2

r=1-(r_numerator/r_denominator)
print(r)
## End: evaluating our weights by R-square.
