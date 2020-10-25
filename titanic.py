import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 12)

train = pd.read_csv('E:/.../train.csv')

train.drop('PassengerId', axis=1, inplace=True)

sns.jointplot(x='Age', y='Survived', data=train)
plt.show()

sns.countplot(x='Survived', data=train, hue='Sex')
plt.show()

sns.boxplot(x='Pclass', y='Age', data=train)
plt.show()

print(train['Age'].isnull().sum())


def fillnulls(cols):
    ages = cols[0]
    classes = cols[1]
    if pd.isna(ages):
        if classes == 1:
            return 37
        elif classes == 2:
            return 28
        else:
            return 24
    else:
        return ages


train['Age'] = train[['Age', 'Pclass']].apply(fillnulls, axis=1)
print(train[['Age', 'Name']])

train.loc[train['Sex'] == 'male', 'Sex'] = 1
train.loc[train['Sex'] == 'female', 'Sex'] = 0

train.drop('Cabin', axis=1, inplace=True)
train.dropna(inplace=True)

sns.pairplot(train)
plt.show()

sns.heatmap(train.isnull())
plt.show()

sex = pd.get_dummies(train['Sex'], drop_first=True)
embarked = pd.get_dummies(train['Embarked'], drop_first=True)
train = pd.concat([train, sex, embarked], axis=1)

x = train[['Pclass', 'Age', 'SibSp', 'Parch', 'male', 'Q', 'S']]
y = train['Survived']

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(xtrain, ytrain)
predicts = logmodel.predict(xtest)

print(predicts)

from sklearn.metrics import classification_report
print(classification_report(ytest, predicts))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(ytest, predicts))

print(train.info())
print(train.head())

