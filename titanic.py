#module
import csv
import pandas as pd
from sklearn import model_selection
import seaborn as sns
from sklearn.ensemble import  RandomForestClassifier 
from sklearn.metrics import accuracy_score
#Read training and test data
df_train = pd.read_csv('TRAIN.CSV')
df_test = pd.read_csv('TEST.CSV')
#Display part of training data
df_train.head()
#Check of the null value of training data
df_train.isnull().sum()
#Check of the null value of test data
df_test.isnull().sum()
#Change string to integer
df_train['Sex'] =  df_train['Sex'].map({'female': 0, 'male': 1})
df_test['Sex'] =  df_test['Sex'].map({'female': 0, 'male': 1})


df_train['Embarked'] =  df_train['Embarked'].map({'Q': 1, 'C': 2,'S':3})
df_test['Embarked'] =  df_test['Embarked'].map({'Q': 1, 'C': 2,'S':3})
#Change null value to integer

df_train = pd.DataFrame(df_train)
df_test  = pd.DataFrame(df_test)

df_train['Age'] =  df_train['Age'].fillna(-1)
df_test['Age'] = df_test['Age'].fillna(-1)

df_train['Cabin'] = df_train['Cabin'].fillna(0)
df_test['Cabin'] = df_test['Cabin'].fillna(0)


df_train['Embarked'] = df_train['Embarked'].fillna(0)

df_test['Fare'] = df_test['Fare'].fillna(0)

#Drop unused columns

df_train = df_train.drop(columns='Name')
df_test = df_test.drop(columns='Name')

df_train = df_train.drop(columns='Ticket')
df_test = df_test.drop(columns='Ticket')

df_train = df_train.drop(columns='Cabin')
df_test = df_test.drop(columns='Cabin')

#Check training data
df_train.head()

#Display training data correlation
sns.heatmap(df_train.corr(),cmap="summer_r")

#Split the objective variable of training data and its explanatory variables
titanic_target = df_train['Survived']

titanic_data = df_train.drop(columns='Survived')

#Split  training data as  training data and test data
x_train,x_test,y_train,y_test = model_selection.train_test_split(titanic_data,titanic_target,test_size =0.2)
                
#Create the model and fit it
model = RandomForestClassifier( n_estimators= 2000,max_depth= 20)
model.fit(x_train,y_train)


y_pred = model.predict(x_test)

accuracy_score(y_test,y_pred)

#Read the data for submission
evaluate_data = pd.read_csv('SUBMISSON.CSV')

evaluate_test = evaluate_data['Survived']


#Predict  the data for submission
test_y_pred = model.predict(df_test)
accuracy_score(evaluate_test,test_y_pred)

#Format the data for submission
submit_data = pd.DataFrame({'PassengerId': evaluate_data['PassengerId'], 'Survived': pd.Series(test_y_pred)})

#Display the data for submission
submit_data.head()

#Change the data for submission to csv file
submit_data.to_csv('submission.csv',index=False)
