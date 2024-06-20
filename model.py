# Importing important libraries and modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
from my_iris_model import my_predict

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

print("\n--Model using LogisticRegression--\n")
# Checking its confusion matrix
print(confusion_matrix(y_test,y_pred))

# accuracy
accuracy=accuracy_score(y_test,y_pred)*100

print("Accuracy of the model is {:.2f}\n\n".format(accuracy))

# RESULT
"""
Output: 

 [13  0  0]
 [ 0 15  1]
 [ 0  0  9]

Accuracy of the model is 97.37

"""

print("--My Model--\n")

# Prediction Testing of my model
my_y_pred = my_predict(x)

# Checking its confusion matrix
my_cm = confusion_matrix(y,my_y_pred)
print("Confusion Matrix of my iris model\n",my_cm)

# accuracy
my_accuracy = accuracy_score(y,my_y_pred)*100

print("Accuracy of my iris model is {:.2f}\n\n".format(my_accuracy))


# RESULT
"""
--My Model--

Confusion Matrix of my iris model

 [50  0  0]
 [ 0 48  2]
 [ 0  4 46]

Accuracy of my iris model is 96.00

"""
