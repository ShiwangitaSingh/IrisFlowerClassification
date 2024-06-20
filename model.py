import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix

df = pd.read_csv('C:\\Users\\shiwa\\Downloads\\Iris Flower - Iris.csv')

df=df.drop(columns="Id")

x=df.iloc[:,:4] # Independent Variables
y=df.iloc[:,4] # Dependent variables

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)

model=LogisticRegression()

model.fit(x_train,y_train)

y_pred=model.predict(x_test)

print(confusion_matrix(y_test,y_pred))

accuracy=accuracy_score(y_test,y_pred)*100

print("Accuracy of the model is {:.2f}".format(accuracy))

"""
Output: 

 [13  0  0]
 [ 0 15  1]
 [ 0  0  9]

Accuracy of the model is 97.37

"""
