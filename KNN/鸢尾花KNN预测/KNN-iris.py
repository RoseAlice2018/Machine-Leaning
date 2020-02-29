'''1.数据读入'''
from sklearn import datasets
iris_dataset=datasets.load_iris()
print("Keys of iris_dataset:\n{}".format(iris_dataset.keys()))
'''2.数据分析'''
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)
iris_dataframe=pd.DataFrame(X_train,columns=iris_dataset.feature_names)
grr=pd.plotting.scatter_matrix(iris_dataframe,c=y_train,figsize=(15,15),marker='o',hist_kwds={'bins':20},s=60,alpha=.8)
'''3.调库使用KNN模型'''
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
X_new=np.array([[5,2.9,1,0.2]])
print("X_new.shape:{}".format(X_new.shape))
prediction=knn.predict(X_new)
print("Prediction:{}".format(prediction))
print("Predicted target name:{}".format(iris_dataset['target_names'][prediction]))
