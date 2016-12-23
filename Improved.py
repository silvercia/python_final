import numpy as np
import math
from numpy import array, vstack, shape, dot, ones, transpose, sqrt, sign


class Linear_Regression(object):
   '''
   Construct the Linear model from the sample.
   '''

   def data_into_matrix(self, data_path):
       '''
       :param data_path: get the data for training from the data path.
       Turn the data into two matrix: one for calculating the weights, and one for calculating R-square.
       '''
       linelist = list()
       with open(data_path) as data:
           for line in data:
               linelist.append(list(map(float, line.split())))
       # linelist: [sample 1, sample 2,...,sample m]
           self.x = array(linelist[0])
           self.x.fill(0)
           self.data_param = array(linelist[1])
           self.data_param.fill(0)
       # x: the only usage of x is to make vstack work
           for sample in linelist:
               a = array(sample)
               self.x = vstack((self.x,a))
           for sample in linelist:
               b = array(sample)
               self.data_param = vstack((self.data_param,b))
       # x = matrix[0;
       #        sample 1;
       #        sample 2;
       #          ...;
       #        sample m] (507,14)

       self.x = self.x[1:]
       self.data_param = self.data_param[1:]   # original data
       # delete the first row, whose values are all 0, so that only the real data from 'data.txt' is left. x:(506,14)
       self.y = self.x[:,shape(self.x)[1]-1]
       self.data_price = self.data_param[:,shape(self.data_param)[1]-1]   # original data
       # extract the price
       self.x = self.x[:,:shape(self.x)[1]-1]
       self.data_param = self.data_param[:,:shape(self.data_param)[1]-1]  # original data
       # delete the price

   def improve(self):
       '''
       Normalise the data and Transform LSTAT into square root.
       '''
       max = self.x.max(axis=0)
       min = self.x.min(axis=0)
       ran = max-min
       avg = self.x.sum(axis=0)/shape(self.x)[0]
       # calculate the range and average of every feature.
       avg[3]=0
       # for the categorical feature, we have to maintain its value, which is either 0 or 1.
       self.x0 = (self.x-avg)/ran
       # x0: normalized x.
       sqrx = sqrt(np.abs(self.x)) * sign(self.x)
       squx = (self.x**2) * sign(self.x)
       self.x[:,7:10] = sqrx[:,7:10]
       self.x[:,11:13] = sqrx[:,11:13]
       self.x[:,0:6] = squx[:,0:6]
       sqrx0 = sqrt(np.abs(self.x0)) * sign(self.x0)
       squx0 = (self.x0**2) * sign(self.x0)
       self.x0[:,7:10]= sqrx0[:,7:10]
       self.x0[:,11:13]= sqrx0[:,11:13]
       self.x0[:,0:6] = squx0[:,0:6]
       sqrdata=sqrt(np.abs(self.data_param)) * sign(self.data_param)
       squdata = (self.data_param**2) * sign(self.data_param)
       self.data_param[:,7:10] = sqrdata[:,7:10]
       self.data_param[:,11:13] = sqrdata[:,11:13]
       self.data_param[:,0:6] = squdata[:,0:6]
       # transform LSTAT, B, TAX, RAD, and DIS into the square root. This can improve R-square.
       # transform CRIM, ZN, INDUS, NOX, RM into square.

   def get_parameters(self):
       '''
       Calculate the weights.
       '''
       self.x = transpose(self.x)
       self.x0 = transpose(self.x0)
       # transpose. After that: (13,506)
       z = ones(shape(self.x)[1])
       self.x = vstack((z,self.x))
       self.x0 = vstack((z,self.x0))
       # add a new row whose values are all 1, into the first row of x and x0. (14,506).
       # this makes calculating weights possible.
       # and then we do the similar things for the original data so that we can calculate R-square.
       self.data_param = transpose(self.data_param)
       # transpose. After that: (14,506)
       z0 = ones(shape(self.data_param)[1])
       self.data_param = vstack((z0,self.data_param))
       # add a new row whose values are all 1, into the first row of data_param. data_param:(14,506).
       # this makes calculating R-square possible.
       parameter = dot(self.y,dot(transpose(self.x0),np.linalg.inv(dot(self.x0,transpose(self.x0)))))
       # calculate the parameter by normal equation.
       y0 =dot(parameter,self.x0)
       self.parameter = dot(y0,dot(transpose(self.x),np.linalg.inv(dot(self.x,transpose(self.x)))))
       # calculate the parameter for non-normalized data.

   def get_R_square(self):
       '''
       Calculate R-square.
       '''
       r_numerator = 0
       for i in range(0,shape(self.data_param)[1]):
           r_numerator += (self.data_price[i]-dot(self.parameter,self.data_param[:,i]))**2
       # calculate the numerator.
       r_denominator =0
       avgy = (np.sum(self.data_price))/shape(self.data_price)[0]
       for j in range(0,shape(self.data_param)[1]):
           r_denominator +=(self.data_price[i]-avgy)**2
       # calculate the denominator.
       self.evaluation=1-(r_numerator/r_denominator)
       # there is R-square.


class Prediction(object):
    '''
    Predict the price based on previous results.
    '''

    def __init__(self, crim=0, zn=0, induc=0, chas=0, nox=0, rm=0, age=0, dis=0, rad=0, tax=0, ptratio=0, b=0, lstat=0):
        '''
        All the features.
        '''
        self.crim = crim
        self.zn = zn
        self.induc = induc
        self.chas = chas
        self.nox= nox
        self.rm = rm
        self.age = age
        self.dis = dis
        self.rad = rad
        self.tax = tax
        self.ptratio = ptratio
        self.b = b
        self.lstat = lstat

    def predict(self,parameter):
        '''
        :param parameter: the result calculated from previous.
        :param ran: the range for a feature got from the previous class by training on sample datas.
        :param avg: the average for a feature got from the previous class by training on sample datas.
        '''
        list = [1,self.crim, self.zn, self.induc, self.chas, self.nox, self.rm, self.age, self.dis, self.rad, self.tax, self.ptratio, self.b, self.lstat]
        for i in [8,9,10,12,13]:
            list[i] = math.sqrt(list[i])
        for i in range(1,6):
            list[i] = list[i]**2
        param = array(list)
        self.price = (dot(parameter,param))*1000
        # estimate the price.

def main(data_path,crim=0, zn=0, induc=0, chas=0, nox=0, rm=0, age=0, dis=0, rad=0, tax=0, ptratio=0, b=0, lstat=0):
    a = Linear_Regression()
    a.data_into_matrix(data_path)
    a.improve()
    a.get_parameters()
    a.get_R_square()
    print('The linear regression is: \n'
          'estimated price = '+str(a.parameter[0])+str(a.parameter[1])+'CRIM**2+'+str(a.parameter[2])+'ZN**2+'+str(a.parameter[3])+'INDUS**2+'+str(a.parameter[4])+'CHAS'+str(a.parameter[5])+'NOX**2+'+str(a.parameter[6])+'RM**2+'+str(a.parameter[7])+'AGE'+str(a.parameter[8])+'DIS**0.5+'+str(a.parameter[9])+'RAD**0.5'+str(a.parameter[10])+'TAX**0.5'+str(a.parameter[11])+'PTRATTO+'+str(a.parameter[12])+'B**0.5'+str(a.parameter[13])+'LSTAT**0.5')
    print('R-square is: '+ str(a.evaluation))
    b = Prediction(crim, zn, induc, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat)
    b.predict(a.parameter)
    # if you want a precise estimated price, use this line:
    # print('Estimated price is: $'+str(b.price)+'.')

if __name__ == "__main__":
    main('data.txt')

