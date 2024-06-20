# Importing important libraries and modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix

# Loading data
df = pd.read_csv('C:\\Users\\shiwa\\Downloads\\Iris Flower - Iris.csv')

# Two columns have same values so deleting one
df=df.drop(columns="Id")

# Deviding input and output basically
x=df.iloc[:,:4] # Independent Variables
y=df.iloc[:,4] # Dependent variables

# Splitting the data for training and testing
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)

# Creating an model
model=LogisticRegression()

# Training the model
model.fit(x_train,y_train)

# Testing the model
y_pred=model.predict(x_test)

# Checking its confusion matrix
print(confusion_matrix(y_test,y_pred))

# accuracy
accuracy=accuracy_score(y_test,y_pred)*100

print("Accuracy of the model is {:.2f}".format(accuracy))

# RESULT
"""
Output: 

 [13  0  0]
 [ 0 15  1]
 [ 0  0  9]

Accuracy of the model is 97.37

"""
