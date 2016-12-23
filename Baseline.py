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

   def get_parameters(self):
       '''
       Calculate the weights.
       '''
       self.x = transpose(self.x)
       # transpose. After that: (13,506)
       z = ones(shape(self.x)[1])
       self.x = vstack((z,self.x))
       # add a new row whose values are all 1, into the first row of x. (14,506).
       # this makes calculating weights possible.
       # and then we do the similar things for the original data so that we can calculate R-square.
       self.data_param = transpose(self.data_param)
       # transpose. After that: (14,506)
       z0 = ones(shape(self.data_param)[1])
       self.data_param = vstack((z0,self.data_param))
       # add a new row whose values are all 1, into the first row of data_param. data_param:(14,506).
       # this makes calculating R-square possible.
       self.parameter = dot(self.y,dot(transpose(self.x),np.linalg.inv(dot(self.x,transpose(self.x)))))
       # calculate the parameter by normal equation.

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
        Estimate the price.
        '''
        list = [1,self.crim, self.zn, self.induc, self.chas, self.nox, self.rm, self.age, self.dis, self.rad, self.tax, self.ptratio, self.b, self.lstat]
        param = array(list)
        self.price = (dot(parameter,param))*1000
        # estimate the price.

def main(data_path,crim=0, zn=0, induc=0, chas=0, nox=0, rm=0, age=0, dis=0, rad=0, tax=0, ptratio=0, b=0, lstat=0):
    a = Linear_Regression()
    a.data_into_matrix(data_path)
    a.get_parameters()
    a.get_R_square()
    print('The linear regression is: \n'
          'estimated price = '+str(a.parameter[0])+str(a.parameter[1])+'CRIM+'+str(a.parameter[2])+'ZN+'+str(a.parameter[3])+'INDUS+'+str(a.parameter[4])+'CHAS'+str(a.parameter[5])+'NOX+'+str(a.parameter[6])+'RM+'+str(a.parameter[7])+'AGE'+str(a.parameter[8])+'DIS+'+str(a.parameter[9])+'RAD'+str(a.parameter[10])+'TAX'+str(a.parameter[11])+'PTRATTO+'+str(a.parameter[12])+'B'+str(a.parameter[13])+'LSTAT**0.5')
    print('R-square is: '+ str(a.evaluation))
    b = Prediction(crim, zn, induc, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat)
    b.predict(a.parameter)
    # if you want a precise estimated price, use this line:
    # print('Estimated price is: $'+str(b.price)+'.')

if __name__ == "__main__":
    main('data.txt')

