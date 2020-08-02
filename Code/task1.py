import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Read the dataset
data = pd.read_csv('task1_dataset.csv') #reads the csv file to the program and assigns it to array data
data_task1 = data.sort_values(by = ['x'])  #sorts the data by x and assigns to new array data_task1

x = data_task1['x'].values #gets all x values from the dataset and imports under variable x
y = data_task1['y'].values #gets all y values from the dataset and imports under variable y

#the train and test variables will be used to calculate the train_set data and test_set data
train = int(len(x)*0.7) #Works out 70% of the dataset
test = int(len(x)*0.3) #Works out 30% of the dataset

#x dataset
x_train = np.empty(train) #Creates a numpy array at the size of 70% of the dataset
x_test = np.empty(test) #Creates a numpy array at the size of 30% of the dataset

#For loop that will add the 70% of the x values into the training dataset
for i in range(0, train):
    x_train[i] = x[i] 

#For loop that will add the rest of the 30% of x values into the test dataset
for i in range(0,test):
    x_test[i] = x[train + i]

#y dataset
y_train = np.empty(train) #Creates a numpy array at the size of 70% of the dataset
y_test = np.empty(test) #Creates a numpy array at the size of 30% of the dataset

#For loop that will add the 70% of the y values into the training dataset
for i in range(0, train):
    y_train[i] = y[i]

#Forr loop that will ad the rest of the 30% of y values into the test dataset    
for i in range(0,test):
    y_test[i] = y[train + i] 
    
#This funciton will do feature expansion up to a certain degree given data set x
def getPolynomialDataMatrix(x, degree):
    X = np.ones(x.shape)
    for i in range(1, degree + 1):
        X = np.column_stack((X, x ** i))
    return X #returns the data matrix

#This function will compute the optimal beta value given input data x and output data y and desired degree of polynomial
def pol_regression(x, y, degree):
    if degree > 0: 
        X = getPolynomialDataMatrix(x, degree) #uses the data matrix function
        
        XX = X.transpose().dot(X)
        w = np.linalg.solve(XX, X.transpose().dot(y))
    else:
        w = sum(x) / len(x) #the else statement allows me to do polynomial regression if the degree is 0
    return w #returns the weight used for polynomial regression

def eval_pol_regression(w, x, y, degree): #function that will evaluate the polynomial regression given on a certain degree
    w = pol_regression(x, y, degree) #uses the pol_regression function to get a parameter
    x_test = getPolynomialDataMatrix(x,degree) #creates our x_test with the use of the data matrix function
    y_test = x_test.dot(w) #creates a predicted y based on our x_test
    
    error = y - y_test #works out the error using actual y - predict y
    sqr_error = np.square(error) #calculate the square of all the errors
    
    mse = sqr_error.sum()/sqr_error.size # calculates the mean squared error
    rmse = np.sqrt(mse) #square roots the mean squared error to give the root mean squared error
    
    return rmse #returns the root mean squared error

#Degree 0
w0 = pol_regression(x, y, 0) #uses prol_regression function to find the weights at degree 0
Xtest0 = getPolynomialDataMatrix(x,0) #uses the data matrix function to create a matrix at the degree of 0
ytest0 = Xtest0.dot(w0) 
plt.plot(x, y, 'bo') #plots the x and y data points to a scatter plot
plt.plot(x, ytest0, 'r') #Illustrated with a red line
plt.show() #draws the poly regression for degree 0

#1st degree
w1 = pol_regression(x, y, 1) #uses pol_regression function to find the weights at degree 1
Xtest1 = getPolynomialDataMatrix(x, 1) #uses the data matrix function to create a matrix at the degree of 1
ytest1 = Xtest1.dot(w1)
plt.plot(x, y, 'bo') #plots the x and y data points to a scatter plot
plt.plot(x, ytest1, 'c') #Illustrated with a cyan line
plt.show() # draws the poly rergression for degree 1

#2nd degree
w2 = pol_regression(x, y, 2) #uses pol_regression function to find the weights at degree 2
Xtest2 = getPolynomialDataMatrix(x, 2) # uses the data matrix function to create a matrix at the degree of 2
ytest2 = Xtest2.dot(w2)
plt.plot(x, y, 'bo') # plots the x and y data poitns to a scatter plot
plt.plot(x, ytest2, 'g') #Illustrated with a green line
plt.show() # draws the poly regression for degree 2

#3rd degree
w3 = pol_regression(x, y, 3) #uses pol_regression function to find the weights at degree 3
Xtest3 = getPolynomialDataMatrix(x, 3) # uses the data matrix function to create a matrix at the degree of 3
ytest3 = Xtest3.dot(w3)
plt.plot(x, y, 'bo') #plots hte x and y data points to a scatter plot
plt.plot(x, ytest3, 'm') #Illustrated with a magenta line
plt.show() # draws the poly regression for degree 3

#5th degree
w5 = pol_regression(x, y, 5) # uses pol_regression function to find the weights at degree 5
Xtest5 = getPolynomialDataMatrix(x, 5) #uses the data matrix function to create a matrix at the degree of 5
ytest5 = Xtest5.dot(w5)
plt.plot(x, y, 'bo') # plots the x and y data points to a scatter plot
plt.plot(x, ytest5, 'b') #Illustrated with a blue line
plt.show() #draws the poly regression for degree 5

#10th degree
w10 = pol_regression(x, y, 10) #uses pol_regression function to find the weights at degree 10
Xtest10 = getPolynomialDataMatrix(x, 10) # uses the data matrix function to create a matrix at the degree of 10
ytest10 = Xtest10.dot(w10)
plt.plot(x, y, 'bo') # plots the x and y data points to a scatter plot
plt.plot(x, ytest10, 'y') #Illustrated with a yellow line
plt.show() # draws the poly regression for degree 10

#evaluate train model
array_train = [] #creates an array to store the train rmse values
array_train.append([0, eval_pol_regression(w0, x_train, y_train, 0)]) #zero degree rmse plot
array_train.append([1, eval_pol_regression(w1, x_train, y_train, 1)]) #1st degree rmse plot
array_train.append([2, eval_pol_regression(w2, x_train, y_train, 2)]) #2nd degree rmse plot
array_train.append([3, eval_pol_regression(w3, x_train, y_train, 3)]) #3rd degree rmse plot
array_train.append([5, eval_pol_regression(w5, x_train, y_train, 5)]) #5th degree rmse plot
array_train.append([10, eval_pol_regression(w10, x_train, y_train, 10)]) #10th degree rmse plot

x = [] #x values for the array_train array
y = [] #y values for the array_train array

for i in range(len(array_train)): #for loop that will add all the x values to the x array
    x.append(array_train[i][0])

for i in range(len(array_train)): #for loop that will add all the y values to the y array
    y.append(array_train[i][1])
    
plt.plot(x, y, 'b') #plots the rmse based on the degree
plt.plot#plots the graph

#evaluate the test model
array_test = [] #creates an array to store the test rmse values
array_test.append([0, eval_pol_regression(w0, x_test, y_test, 0)]) #zero degree rmse plot
array_test.append([1, eval_pol_regression(w1, x_test, y_test, 1)]) #1st degree rmse plot
array_test.append([2, eval_pol_regression(w2, x_test, y_test, 2)]) #2nd degree rmse plot
array_test.append([3, eval_pol_regression(w3, x_test, y_test, 3)]) #3rd degree rmse plot
array_test.append([5, eval_pol_regression(w5, x_test, y_test, 5)]) #5th degree rmse plot
array_test.append([10, eval_pol_regression(w10, x_test, y_test, 10)]) #10th degree rmse plot

x = [] #x values for the array_test array
y = [] #y values for the array_test array

for i in range(len(array_test)): #for loop that will add all the x values to the x array
    x.append(array_test[i][0])

for i in range(len(array_test)): #for loop that will add all the y values to the y array
    y.append(array_test[i][1])
    
plt.plot(x, y, 'g') #plots the rmse based on the degree
plt.plot#plots the graph