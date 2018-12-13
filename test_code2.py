import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from scipy import optimize
import pylab as py
from scipy.stats import norm
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from scipy.stats import norm
import matplotlib.mlab as mlab
from sklearn.model_selection import StratifiedShuffleSplit   
from sklearn import linear_model
from numpy import sin, cos
import numpy as np
import scipy.integrate as integrate
import random

# for each experiment value of l1,l2,m1,m2 and th1,th2,w1,w2 are same so explicitely add these features after training.


'''
this code is for generation x2 and y2 and input set seperately and testing it in 
test_code.py file. where inputs are obtained by run this file used to test the SVR model in 
test_code.py file.
'''

# th1 and th2 are the initial angles (degrees)
# w10 and w20 are the initial angular velocities (degrees per second)

th1 = 120.0
w1 = 0.0
th2 = -10.0
w2 = 0.0

# create a time array from 0..100 sampled at 0.05 second steps
dt = 0.1
t = np.arange(0.0, 100, dt)

inputs = []

outputs = []

out_x = []

out_y = []

file = open("data/outputs.txt","w+")
input_file = open("data/inputs.txt","w+")

l1 = open("data/L1.txt" , "w+")
l2 = open("data/L2.txt" , "w+")


m1 = open("data/M1.txt" , "w+")
m2 = open("data/M2.txt" , "w+")

theta1 = open("data/th1.txt" , "w+")
theta2 = open("data/th2.txt" , "w+")

omega1 = open("data/w1.txt" , "w+")
omega2 = open("data/w2.txt" , "w+")

time_slot = open("data/time_slots.txt" , "w+")

xx2 = open("data/x2.txt","w+")
yy2 = open("data/y2.txt","w+")

  

G = 9.8  # acceleration due to gravity, in m/s^2
L1 = 1.0 #random.uniform(0.0 , 2.0)   length of pendulum 1 in m
L2 = 1.0 #random.uniform(0.0 , 2.0)   length of pendulum 2 in m
M1 = 1.0  # mass of pendulum 1 in kg
M2 = 1.0 # mass of pendulum 2 in kg

def derivs(state, t):

  dydx = np.zeros_like(state)
  dydx[0] = state[1]

  del_ = state[2] - state[0]
  den1 = (M1 + M2)*L1 - M2*L1*cos(del_)*cos(del_)
  dydx[1] = (M2*L1*state[1]*state[1]*sin(del_)*cos(del_) +
               M2*G*sin(state[2])*cos(del_) +
               M2*L2*state[3]*state[3]*sin(del_) -
               (M1 + M2)*G*sin(state[0]))/den1

  dydx[2] = state[3]

  den2 = (L2/L1)*den1
  dydx[3] = (-M2*L2*state[3]*state[3]*sin(del_)*cos(del_) +
               (M1 + M2)*G*sin(state[0])*cos(del_) -
               (M1 + M2)*L1*state[1]*state[1]*sin(del_) -
               (M1 + M2)*G*sin(state[2]))/den2

  return dydx

  # initial state
state = np.radians([th1, w1, th2, w2])

  # integrate your ODE using scipy.integrate.
y = integrate.odeint(derivs, state, t)

exp = []

out = []

x1 = L1*sin(y[:, 0])
y1 = -L1*cos(y[:, 0])

x2 = L2*sin(y[:, 2]) + x1
y2 = -L2*cos(y[:, 2]) + y1

for i in range(len(t)):
    inputs.append([L1 , L2 , M1 , M2 , th1 , th2 , w1 , w2 , t[i]])
    l1.write(str(L1))
    l1.write("\n")
    l2.write(str(L2))
    l2.write("\n")
    m1.write(str(M1))
    m1.write("\n")
    m2.write(str(M2))
    m2.write("\n")
    theta1.write(str(th1))
    theta1.write("\n")
    theta2.write(str(th2))
    theta2.write("\n")
    omega1.write(str(w1))
    omega1.write("\n")
    omega2.write(str(w1))
    omega2.write("\n")
    time_slot.write(str(t[i]))
    time_slot.write("\n")


for i in range(len(x2)): 
    outputs.append([x2[i] , y2[i]])
    out_x.append(x2[i])
    out_y.append(y2[i])
   

  #outputs.append(out)

print("len(x2)" , len(x2))
print("len(y2)" , len(y2))


for i in range(len(inputs)):
  input_file.write(str(inputs[i]))
  input_file.write("\n")

for i in range(len(outputs)):
  file.write(str(outputs[i]))
  file.write("\n")


for i in range(len(out_x)):
  xx2.write(str(out_x[i]))
  xx2.write("\n")

for i in range(len(out_y)):
  yy2.write(str(out_y[i]))
  yy2.write("\n")

file.close()
input_file.close()
xx2.close()
yy2.close()


#kernel ridge regression

#

clf1 = KernelRidge(alpha=1.0 ,coef0=1, degree=3, gamma=None, kernel='rbf',kernel_params=None) #SVR(C=1.0, epsilon=0.2)  Ridge(alpha=1.0) or linear_model.LinearRegression()
'''for i in range(len(inputs)):
    clf1.fit (inputs[i], outputs[i])
    print('kernel_ridge regression_coefficients',clf1.coef_)
    print('length of coef',len(clf1.coef_))
'''
clf1.fit(inputs, outputs)
#print('kernel_ridge regression_coefficients',clf1.dual_coef_)
#print('length of coef',len(clf1.dual_coef_))

'''
t=[1, 1, 0.9263597553793446, 0.4447075849449722, 120.0, -10.0, 0.0, 0.0, 0.02]
y_pred25=clf1.predict(t) #prediction function Ridge Regression
print('predicted output is : ',y_pred25)

print("Calculated output is : ",outputs[0])

print("error for eastimating X is ",((abs(y_pred25[0][0])-abs(outputs[0][0])))*100)
print("error for eastimating Y is ",((abs(y_pred25[0][1])-abs(outputs[0][1])))*100)
'''



