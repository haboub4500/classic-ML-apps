from numpy.random.mtrand import f
import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

st.set_option('deprecation.showPyplotGlobalUse', False)


st.title("Classical Classification Applications")

st.write(""" 
# Explore different classifiers
  Which one is the best?
""")

dataset_name = st.sidebar.selectbox("select your dataset :" , ("Iris", "Breast Cancer", "Wine"))
classifier_name = st.sidebar.selectbox("select your classifier :" , ("KNN", "SVM", "Random forest"))

def get_dataset(dataset_name):
  if dataset_name == "Iris" : 
    data = datasets.load_iris()
  elif dataset_name == "Wine" : 
    data = datasets.load_wine()
  else  :
    data = datasets.load_breast_cancer()
    
  x=data.data
  y=data.target

  return  x,y


x,y = get_dataset(dataset_name)
st.write("shape of dataset of the ",dataset_name,"dataset : ", x.shape)
st.write("number of classes of the ",dataset_name,"dataset : ", len(np.unique(y)))

def add_parameter_ui(clf_name):
  params = dict()
  if clf_name == "KNN":
    K = st.sidebar.slider("K" , 1, 15)
    params["K"]=K
  elif clf_name =="SVM":
    C = st.sidebar.slider("C" , 0.01, 10.0)
    params["C"]=C
  else :
    max_depth = st.sidebar.slider(" Max depth " , 2 , 15)
    nb_estimators = st.sidebar.slider(" number of estimators" ,1 , 100)
    params["max depth"]=max_depth
    params["number of estimators"]=nb_estimators
  return params
params =add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
  if clf_name == "KNN":
    clf = KNeighborsClassifier(n_neighbors=params["K"])
  elif clf_name =="SVM":
    clf = SVC(C=params["C"])
  else :
    clf = RandomForestClassifier(n_estimators=params["number of estimators"], max_depth=params["max depth"], random_state=69)
  return clf
clf = get_classifier(classifier_name, params)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2 , random_state=69)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)  

acc = accuracy_score(y_test , y_pred)
st.write(f"classifier name : {classifier_name}")
st.write(f"classifier accuracy : {acc}")

pca = PCA(2)
x_projected = pca.fit_transform(x)

x1 = x_projected[:,0]
x2 = x_projected[:,1]


fig = plt.figure()
plt.scatter(x1,x2, c=y, alpha=0.8, cmap = "viridis")
plt.xlabel("Pricipal component 1")
plt.ylabel("Pricipal component 2")
plt.colorbar()

st.pyplot()

