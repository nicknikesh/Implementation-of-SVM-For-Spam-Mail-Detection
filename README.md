# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Preprocess Data: Clean emails and convert text to numeric features (TF-IDF or BoW).
2. Split Dataset: Divide data into training and testing sets.
3. Train Model: Use SVM classifier (linear/RBF kernel) on training data.
4. Evaluate Model: Test performance using accuracy, precision, recall, and F1-score.

## Program:

Program to implement the SVM For Spam Mail Detection.

Developed by: NIKESH KUMAR C

RegisterNumber: 212223040132

```
import chardet
file="spam.csv"
with open(file,'rb') as rawdata:
    result=chardet.detect(rawdata.read(100000))
result
import pandas as pd
data=pd.read_csv('spam.csv',encoding='Windows-1252')
data.head()
data.isnull().sum()
data.info()
x=data["v2"].values
y=data["v1"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
x_train
x_test
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
x_train
x_test
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
print("NAME: NIKESHKUMAR C")
print("REG NO:212223040132")
cr=metrics.classification_report(y_test,y_pred)
print("Classification report:")
print(cr)
cm=metrics.confusion_matrix(y_test,y_pred)
print("Confusion Matrix")
print(cm)
```

## Output:
<img width="609" height="533" alt="Screenshot 2025-11-17 142555" src="https://github.com/user-attachments/assets/f1da50c2-d129-4141-9801-c2a2feffc95b" />



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
