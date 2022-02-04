from random import random
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os,ssl,time

if(not os.environ.get('PYTHONSTTPSVERIFY','')and getattr(ssl,'_create_unverified_context',None)):
    ssl._create_default_https_context=ssl._create_unverified_context

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
print(pd.Series(y).value_counts())
classes = ['A', 'B', 'C','D', 'E','F', 'G', 'H', 'I', 'J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
nclasses = len(classes)

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=9,train_size=7500,test_size=25000)
X_train_scaled=X_train/255.0
X_test_scaled=X_test/255.0

clf=LogisticRegression(solver='saga',multi_class='multinomial').fit(X_train_scaled,y_train)
ypred=clf.predict(X_test_scaled)
accuracy=accuracy_score(y_test,ypred)
print(accuracy)

cap=cv2.VideoCapture(0)
#while(True):
 #   try:
        
 #       ret,frame=cap.read()
 #       gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        

