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

# CODE AND OUPUT:
```
NAME : KRISHNARAJ D
REG NO : 212222230070
```


```
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import chi2

df=pd.read_csv("/content/titanic_dataset.csv")

df.columns
```
![281754647-f12b92dd-2b99-4112-b503-f6621e188376](https://github.com/KRISHNARAJ-D/ODD2023-Datascience-Ex-07/assets/119559695/83c80f20-b327-45c3-8306-6008000af1f2)

```
df.shape
```
![281754731-8bff8a64-a204-406d-b94c-d79a2d9d11f0](https://github.com/KRISHNARAJ-D/ODD2023-Datascience-Ex-07/assets/119559695/253266b0-1735-4397-b988-e1668a3e10ac)

```
x=df.drop("Survived",1)
y=df['Survived']
```
![281755039-3d3995cb-3574-48e1-98fd-abb65decf8c4](https://github.com/KRISHNARAJ-D/ODD2023-Datascience-Ex-07/assets/119559695/67ab8d98-8b17-49fb-9fe1-5c153f6d0cda)


```
df1=df.drop(["Name","Sex","Ticket","Cabin","Embarked"],axis=1)
df1.columns
```
![281755143-777a3891-8d67-4730-b88f-a1b302559dc5](https://github.com/KRISHNARAJ-D/ODD2023-Datascience-Ex-07/assets/119559695/99b82b0f-0c4d-4f13-8904-85f6a86cc063)

```
df1['Age'].isnull().sum()
```
![281755206-941c441a-571b-4f50-a2ca-609aca3035df](https://github.com/KRISHNARAJ-D/ODD2023-Datascience-Ex-07/assets/119559695/554d0db6-b432-4193-b8f1-e7528e7003fc)

```
df1['Age'].fillna(method='ffill')
```
![281755266-a57b5018-113f-4b6d-ad67-a8cf4036b400](https://github.com/KRISHNARAJ-D/ODD2023-Datascience-Ex-07/assets/119559695/7be18f50-0550-4088-9f48-342c9e263090)

```
df1['Age']=df1['Age'].fillna(method='ffill')
df1['Age'].isnull().sum()
```
![281755324-0384839c-ffd0-455f-8c0d-366a5877592e](https://github.com/KRISHNARAJ-D/ODD2023-Datascience-Ex-07/assets/119559695/5e7559d9-0784-432f-b04f-f43f40d6d962)


```
feature=SelectKBest(mutual_info_classif,k=3)
df1.columns
```
![281755427-8d83bcfe-7ef0-4977-a254-8ef27aad0faf](https://github.com/KRISHNARAJ-D/ODD2023-Datascience-Ex-07/assets/119559695/c598b166-e3f9-4980-a67a-4b192c9d48f9)


```
cols=df1.columns.tolist()
cols[-1],cols[1]=cols[1],cols[-1]
df1.columns
```
![281755485-8b7b083c-a7f1-48f0-ae16-882b0b463814](https://github.com/KRISHNARAJ-D/ODD2023-Datascience-Ex-07/assets/119559695/143cd9b3-78cc-4e0c-bad8-08da69ee1341)


```
x=df1.iloc[:,0:6]
y=df1.iloc[:,6]

x.columns
```
![281755556-13d45295-4f3b-4186-8dad-40f25412f2f6](https://github.com/KRISHNARAJ-D/ODD2023-Datascience-Ex-07/assets/119559695/bc0ff672-9aa0-4dff-92e7-f239c8b78c26)


```
y=y.to_frame()
y.columns
```
![281755602-dda29eb8-f135-483c-acc2-2551e996326c](https://github.com/KRISHNARAJ-D/ODD2023-Datascience-Ex-07/assets/119559695/fb218b35-7e52-45ae-a104-fb697a862aee)


```
from sklearn.feature_selection import SelectKBest

data=pd.read_csv("/content/titanic_dataset.csv")
data=data.dropna()
x=data.drop(['Survived','Name','Ticket'],axis=1)
y=data['Survived']
x
```
![281755728-5104376f-476a-4307-92a1-a17b3ad8a502](https://github.com/KRISHNARAJ-D/ODD2023-Datascience-Ex-07/assets/119559695/a3b692d5-fc06-42da-b046-41b4598866cc)

```
data["Sex"]=data["Sex"].astype("category")
data["Cabin"]=data["Cabin"].astype("category")
data[ "Embarked" ]=data ["Embarked"] .astype ("category")

data["Sex"]=data["Sex"].cat.codes
data["Cabin"]=data["Cabin"].cat.codes
data[ "Embarked" ]=data ["Embarked"] .cat.codes
data
```
![281755982-7c4b2301-0b65-4474-a5a7-3d52cc0b82c6](https://github.com/KRISHNARAJ-D/ODD2023-Datascience-Ex-07/assets/119559695/ee2a4dab-4fbe-4985-8646-9ea8975b1dab)


```
k=5
selector = SelectKBest(score_func=chi2,k=k)
x_new = selector.fit_transform(x,y)

selected_feature_indices = selector.get_support(indices=True)

selected_feature_indices = selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]
print("Selected Features: ")
print(selected_features)
```

![281756031-ed5377b2-0d14-4e6b-a9d7-6f4f158ba179](https://github.com/KRISHNARAJ-D/ODD2023-Datascience-Ex-07/assets/119559695/601172b0-ae20-4ab3-b8a5-d38786c660cf)

```
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
sfm = SelectFromModel(model, threshold='mean')
sfm.fit(x,y)
```
![281756106-e09fe730-1ead-4544-8b5e-55eeb1389fb0](https://github.com/KRISHNARAJ-D/ODD2023-Datascience-Ex-07/assets/119559695/ee094931-92a7-4448-b2b5-1a92100f1e6c)


```
selected_feature = x.columns[sfm.get_support()]

print("Selected Features:")
print(selected_feature)
```
![281756168-68a7feb8-53e3-4e98-9173-7e091cee5233](https://github.com/KRISHNARAJ-D/ODD2023-Datascience-Ex-07/assets/119559695/751c0bde-da68-43f6-af25-13c23937c9f1)

```
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

model = LogisticRegression()
num_features_to_remove =2
rfe = RFE(model, n_features_to_select=(len(x.columns) - num_features_to_remove))
rfe.fit(x,y)
```
![281756231-510b30f9-2b30-4893-ac51-5e24a054d5fd](https://github.com/KRISHNARAJ-D/ODD2023-Datascience-Ex-07/assets/119559695/b30f43bb-2c50-47f6-816c-ebb19dfcfde0)

```
selected_features = x.columns[rfe.support_]
print("Selected Features:")
print(selected_feature)
```
![281756276-65ba56f6-29ee-4a05-bdbf-8107b905fb09](https://github.com/KRISHNARAJ-D/ODD2023-Datascience-Ex-07/assets/119559695/4437b84d-243a-4bc6-a39e-3ddc039c58bf)

```
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x,y)
```

![281756313-40660088-e662-4b1b-9080-1c0aaadcfb27](https://github.com/KRISHNARAJ-D/ODD2023-Datascience-Ex-07/assets/119559695/e4997a83-345a-45bb-857d-501a8818eb48)

```
feature_importances = model.feature_importances_
threshold = 0.15
selected_features = x.columns[feature_importances > threshold]

print("Selected Features:")
print(selected_feature)
```
![281756423-fbc287cf-44b1-404c-96b2-eb3b162649a5](https://github.com/KRISHNARAJ-D/ODD2023-Datascience-Ex-07/assets/119559695/23b5ce98-c61d-4790-adc9-90aa41b729ae)



# RESULT :
Thus, the various feature selection techniques have been performed on a given dataset successfully.
