# Ex-07-Feature-Selection
# AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file

# PROGRAM:
```
DEVELOPED BY: Vishnupriya R
REGISTER NO : 212222110054
```
```
#importing library
import pandas as pd
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# data loading
data = pd.read_csv('/content/titanic_dataset.csv')
data
data.tail()
data.isnull().sum()
data.describe()

#now, we are checking start with a pairplot, and check for missing values
sns.heatmap(data.isnull(),cbar=False)

#Data Cleaning and Data Drop Process
data['Fare'] = data['Fare'].fillna(data['Fare'].dropna().median())
data['Age'] = data['Age'].fillna(data['Age'].dropna().median())

# Change to categoric column to numeric
data.loc[data['Sex']=='male','Sex']=0
data.loc[data['Sex']=='female','Sex']=1

# instead of nan values
data['Embarked']=data['Embarked'].fillna('S')

# Change to categoric column to numeric
data.loc[data['Embarked']=='S','Embarked']=0
data.loc[data['Embarked']=='C','Embarked']=1
data.loc[data['Embarked']=='Q','Embarked']=2

#Drop unnecessary columns
drop_elements = ['Name','Cabin','Ticket']
data = data.drop(drop_elements, axis=1)

data.head(11)

#heatmap for train dataset
f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

# Now, data is clean and read to a analyze
sns.heatmap(data.isnull(),cbar=False)

# how many people survived or not... %60 percent died %40 percent survived
fig = plt.figure(figsize=(18,6))
data.Survived.value_counts(normalize=True).plot(kind='bar',alpha=0.5)
plt.show()

#Age with survived
plt.scatter(data.Survived, data.Age, alpha=0.1)
plt.title("Age with Survived")
plt.show()

#Count the pessenger class
fig = plt.figure(figsize=(18,6))
data.Pclass.value_counts(normalize=True).plot(kind='bar',alpha=0.5)
plt.show()

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X = data.drop("Survived",axis=1)
y = data["Survived"]

mdlsel = SelectKBest(chi2, k=5)
mdlsel.fit(X,y)
ix = mdlsel.get_support()
data2 = pd.DataFrame(mdlsel.transform(X), columns = X.columns.values[ix]) # en iyi leri aldi... 7 tane...

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

target = data['Survived'].values
data_features_names = ['Pclass','Sex','SibSp','Parch','Fare','Embarked','Age']
features = data[data_features_names].values

#Build test and training test
X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=0.3,random_state=42)

my_forest = RandomForestClassifier(max_depth=5, min_samples_split=10, n_estimators=500, random_state=5,criterion = 'entropy')


my_forest_ = my_forest.fit(X_train,y_train)
target_predict=my_forest_.predict(X_test)

print("Random forest score: ",accuracy_score(y_test,target_predict))

from sklearn.metrics import mean_squared_error, r2_score
print ("MSE    :",mean_squared_error(y_test,target_predict))
print ("R2     :",r2_score(y_test,target_predict))
```
# OUTPUT:
## data.tail():
![237874262-02f8f580-3780-40e1-b804-191fa125e0bf](https://github.com/vishnupriyaramesh17/ODD2023-Datascience-Ex-07/assets/119393589/b1727105-9215-419d-bd92-d769346cd1e0)

## Null Values:
![237874308-05a88478-7db3-468a-aa00-74749b3c1557](https://github.com/vishnupriyaramesh17/ODD2023-Datascience-Ex-07/assets/119393589/5dbd9cd7-402f-4d68-8e15-b71ace4ea314)

## Describe:
![237874317-2698167c-c887-497a-943a-804a8b452078](https://github.com/vishnupriyaramesh17/ODD2023-Datascience-Ex-07/assets/119393589/e0d29d98-1b6f-4244-8732-00921b3da518)

## missing values:
![237874335-861220c1-de5f-4cca-8e88-9476f8f1b4cf](https://github.com/vishnupriyaramesh17/ODD2023-Datascience-Ex-07/assets/119393589/7726beeb-e42e-42a1-9acc-722a769c2f54)

## Data after cleaning:
![237874341-049d0d1e-69ba-4b80-88ac-0dcf12405eeb](https://github.com/vishnupriyaramesh17/ODD2023-Datascience-Ex-07/assets/119393589/26e48f3a-0b55-4d19-bdda-40feca02beaa)

## Data on Heatmap:
![237874382-661435db-813b-4217-96a8-74b5793e8bc5](https://github.com/vishnupriyaramesh17/ODD2023-Datascience-Ex-07/assets/119393589/e2a9a1b5-d570-41b7-8bbd-3e034ac70968)

## Report of (people survived & Died):
![237874393-e155a502-3bc5-4475-96d9-9f38eef992f2](https://github.com/vishnupriyaramesh17/ODD2023-Datascience-Ex-07/assets/119393589/279b9252-70b7-4504-bfe1-8dae16f51678)

## Cleaned Null values:
![237874408-8cd2b168-8efa-4502-a68f-809d2359bbc1](https://github.com/vishnupriyaramesh17/ODD2023-Datascience-Ex-07/assets/119393589/3ffc798b-d597-4a31-a215-939ad802becb)

## Report of Survived People's Age:
![237874416-f3a33d25-64f3-4e25-8e76-b6b1d786a89e](https://github.com/vishnupriyaramesh17/ODD2023-Datascience-Ex-07/assets/119393589/056002d4-4443-4d04-aca0-d20af6ada35f)

## Report of pessengers:
![237874442-20adfe2d-3dbe-452d-8918-cf8324591ec1](https://github.com/vishnupriyaramesh17/ODD2023-Datascience-Ex-07/assets/119393589/9c8c981b-5de1-436e-9013-2824bd2d5791)

## Report:
![237874468-97844d01-95aa-4637-9a18-128e313fd08c](https://github.com/vishnupriyaramesh17/ODD2023-Datascience-Ex-07/assets/119393589/3513e089-5217-4e2b-823c-2a029585945a)
![237874474-0c49c827-f57d-4831-ab03-6aa867e1d5a7](https://github.com/vishnupriyaramesh17/ODD2023-Datascience-Ex-07/assets/119393589/91d62021-5ef9-4f31-a72c-e3e7e7f7f78e)

# RESULT:
Thus, Sucessfully performed the various feature selection techniques on a given dataset.









