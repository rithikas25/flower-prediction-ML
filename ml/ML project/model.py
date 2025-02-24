#import all the dependencies
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv('iris.csv')
print(df.head())
print('-------------------------------------------------------------')
print(df.tail())


x =df[["Sepal_Length","Sepal_Width","Petal_Length","Petal_Width"]]
Y = df["Class"]

X_train,X_test,Y_train,Y_test = train_test_split(x,Y, test_size=0.3, random_state=50)


sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test) 

#model selection
classifier = RandomForestClassifier()

#fit the model
classifier.fit(X_train,Y_train)

#make pickle file of our model
pickle.dump(classifier,open("model.pkl", "wb"))
