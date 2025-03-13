# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:


Program to implement the simple linear regression model for predicting the marks scored.
Developed by: THAMIZH KUMARAN S
RegisterNumber: 212223240166



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()

df.tail()

x=df.iloc[:,:-1].values
x

y=df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

y_pred

y_test

mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)

plt.scatter(x_train,y_train,color="orange")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours vs Scores (Traning Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color="purple")
plt.plot(x_test,y_pred,color="yellow")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()



## Output:


 HEAD VALUES
 
![Screenshot 2025-03-10 162345](https://github.com/user-attachments/assets/348de10d-d8ce-4e95-968d-f0979153290d)

TAIL VALUES
 
![Screenshot 2025-03-10 162353](https://github.com/user-attachments/assets/08418405-7389-46fb-9442-c918b4fe2372)

HOURS VALUES

![Screenshot 2025-03-10 162400](https://github.com/user-attachments/assets/77fc763d-cb6f-48d8-8b41-7b87c8973f62)

SCORES VALUES

![Screenshot 2025-03-10 162407](https://github.com/user-attachments/assets/aec0fd18-8704-47e2-aa09-beb047bf297b)

Y_PREDICTION

![Screenshot 2025-03-10 162415](https://github.com/user-attachments/assets/84f6bf97-a6cf-41cf-b0ad-b8caf50e4d25)

Y_TEST

![Screenshot 2025-03-10 162421](https://github.com/user-attachments/assets/274e8f0d-663a-420e-81ab-4b35dd9e6368)

RESULT OF MSE,MAE,RMSE

![Screenshot 2025-03-10 162427](https://github.com/user-attachments/assets/22227a84-e191-464a-9451-f652fc423459)

TRAINING SET

![Screenshot 2025-03-10 162437](https://github.com/user-attachments/assets/4a310545-5b64-4665-9fee-9fcf7784dff7)

TEST SET

![Screenshot 2025-03-10 162447](https://github.com/user-attachments/assets/9df15eb3-2083-4b1a-833e-61b6fa733463)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
