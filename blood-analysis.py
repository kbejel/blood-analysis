import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

#import necessary data as panda dataframe
train = pd.read_csv('blood-train.csv')
test = pd.read_csv('blood-test.csv')

#rename the unnamed column
train.rename(columns={"Unnamed: 0" : "Donor_id"}, inplace=True)
test.rename(columns={"Unnamed: 0" : "Donor_id"}, inplace=True)

#view dataframe info to ensure there are no missing values
train.info()
test.info()

#calculate correlation for test and train
train_corr = train.corr()
test_corr = test.corr()

#set up heatmap to view the correlations between the features
train_map = plt.figure(1)
sns.heatmap(train_corr)
test_map = plt.figure(2)
sns.heatmap(test_corr)

#put needed rows in to X_train and y_train
X_train = train.iloc[:, [1,2,3,4]].values
y_train = train.iloc[:, -1].values

print(X_train)
print(y_train)

#assign X_test
X_test = test.iloc[:,[1,2,3,4]].values

print(X_test)

#assign scaler variable
Scaler = StandardScaler()

#fit scaler to X_train and X_test
X_train = Scaler.fit_transform(X_train)
X_test = Scaler.fit_transform(X_test)

#use random forest classifier (fit model and print accuracy)
rand_forest = RandomForestClassifier(random_state=0)
rand_forest.fit(X_train, y_train)
acc = rand_forest.score(X_train, y_train)
print(acc)

#make predictions
y_pred = rand_forest.predict(X_test)
print(y_pred)

#show plot
plt.show()
