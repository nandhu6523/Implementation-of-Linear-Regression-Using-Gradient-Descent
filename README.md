# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program.

2. import numpy as np.

3. Give the header to the data.

4. Find the profit of population.

5. Plot the required graph for both for Gradient Descent Graph and Prediction Graph.

6. End the program.


## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Nandhini S
RegisterNumber:  212222220028

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("ex1.txt",header=None)
data

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of city (10,000s)")
plt.ylabel("Profit ($10,000")
plt.title("Profit Predication")

def computeCost(x,y,theta):
  m=len(y)
  h=x.dot(theta)
  square_err=(h-y)**2
  return 1/(2*m)*np.sum(square_err)
  
data_n=data.values
m=data_n[:,0].size
x=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(x,y,theta)

def gradientDescent(x,y,theta,alpha,num_iters):
  m=len(y)
  J_history=[]
  for i in range(num_iters):
    predictions=x.dot(theta)
    error=np.dot(x.transpose(),(predictions-y))
    descent=alpha*1/m*error
    theta-=descent
    J_history.append(computeCost(x,y,theta))
  return theta,J_history

theta,J_history = gradientDescent(x,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000s")
plt.title("Profit Prediction")

def predict(x,theta):
    predictions=np.dot(theta.transpose(),x)
    return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))
*/
```

## Output:
DATASET:



![Screenshot 2023-09-25 204338](https://github.com/nandhu6523/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/123856724/2fd6f408-5a7b-44eb-8f1f-d9e0fe0e5f90)

Compute cost value:
  
  ![Screenshot 2023-10-29 131904](https://github.com/nandhu6523/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/123856724/01e4f249-2785-475f-b8c2-59fe2d6efacd)

h(x) Value:
    ![Screenshot 2023-10-29 131916](https://github.com/nandhu6523/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/123856724/c1c442d3-f095-45bd-aa84-1d027a6ef41b)


Plt.profitprediction:

![Screenshot 2023-09-25 204412](https://github.com/nandhu6523/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/123856724/795267cd-1600-410f-a4aa-e295eac2993a)

Cost function using Gradient Descent:

![Screenshot 2023-09-25 204428](https://github.com/nandhu6523/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/123856724/b809bd14-141a-4c07-b91d-e41cfc4f24c5)

Profit Prediction:

![Screenshot 2023-09-25 205927](https://github.com/nandhu6523/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/123856724/bcf3cc71-0588-4389-aca4-69ed0a2b132e)

predict1:

![Screenshot 2023-09-25 204517](https://github.com/nandhu6523/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/123856724/6031c451-2ad2-41fb-a836-17570e753a8c)

predict2:

![Screenshot 2023-09-25 204526](https://github.com/nandhu6523/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/123856724/87465221-332f-4ba0-87f9-db34877a54f0)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
