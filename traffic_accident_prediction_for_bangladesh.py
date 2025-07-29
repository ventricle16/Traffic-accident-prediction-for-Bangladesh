import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/content/traffic_accident.csv')
df.head(3)

# @title Weather vs Road_Type

from matplotlib import pyplot as plt
df.plot(kind='scatter', x='Weather', y='Road_Type', s=32, alpha=.8)
plt.gca().spines[['top', 'right',]].set_visible(False)

# @title Road_Type vs Time_of_Day

from matplotlib import pyplot as plt
df.plot(kind='scatter', x='Road_Type', y='Time_of_Day', s=32, alpha=.8)
plt.gca().spines[['top', 'right',]].set_visible(False)

# @title Traffic_Density vs Speed_Limit

from matplotlib import pyplot as plt
df.plot(kind='scatter', x='Traffic_Density', y='Speed_Limit', s=32, alpha=.8)
plt.gca().spines[['top', 'right',]].set_visible(False)

# @title Time_of_Day vs Traffic_Density

from matplotlib import pyplot as plt
import seaborn as sns
def _plot_series(series, series_name, series_index=0):
  palette = list(sns.palettes.mpl_palette('Dark2'))
  xs = series['Time_of_Day']
  ys = series['Traffic_Density']

  plt.plot(xs, ys, label=series_name, color=palette[series_index % len(palette)])

fig, ax = plt.subplots(figsize=(10, 5.2), layout='constrained')
df_sorted = df.sort_values('Time_of_Day', ascending=True)
_plot_series(df_sorted, '')
sns.despine(fig=fig, ax=ax)
plt.xlabel('Time_of_Day')
_ = plt.ylabel('Traffic_Density')

# @title Time_of_Day vs Speed_Limit

from matplotlib import pyplot as plt
import seaborn as sns
def _plot_series(series, series_name, series_index=0):
  palette = list(sns.palettes.mpl_palette('Dark2'))
  xs = series['Time_of_Day']
  ys = series['Speed_Limit']

  plt.plot(xs, ys, label=series_name, color=palette[series_index % len(palette)])

fig, ax = plt.subplots(figsize=(10, 5.2), layout='constrained')
df_sorted = df.sort_values('Time_of_Day', ascending=True)
_plot_series(df_sorted, '')
sns.despine(fig=fig, ax=ax)
plt.xlabel('Time_of_Day')
_ = plt.ylabel('Speed_Limit')

# @title Road_Type

from matplotlib import pyplot as plt
df['Road_Type'].plot(kind='line', figsize=(8, 4), title='Road_Type')
plt.gca().spines[['top', 'right']].set_visible(False)

# @title Traffic_Density

from matplotlib import pyplot as plt
df['Traffic_Density'].plot(kind='line', figsize=(8, 4), title='Traffic_Density')
plt.gca().spines[['top', 'right']].set_visible(False)

"""Dataset description"""

#how many features?
print(f'there {df.shape[1]} features in the dataset')

print('Binary classification problem. Because we are predicting two labels from the features.')

print(f'There are {df.size} data points.')

print(set(df.dtypes))

import seaborn as sb
plt.figure(figsize=(10,6))
sb.heatmap(df.corr(), annot=True)
plt.title('Correlation Heatmap')
plt.show()

"""Imbalanced dataset"""

df['Accident'].value_counts()

labels = ['Accident', 'No Accident']
values = [239, 559]

plt.bar(labels, values)
plt.xlabel('Labels')
plt.ylabel('Values')
plt.title('Bar chart of lables vs values')
plt.show()

"""Database_Pre-processing

Dealing with NULL values
"""

print(df.isna().sum()) # findout null valuef from dataset

non_numeric_column = df.select_dtypes(exclude=['number']).columns.tolist()
print(non_numeric_column)

len(df)

null_rows = df.isnull().any(axis=1).sum()
print(null_rows)

null_cols = df.columns[df.isna().any()].tolist()
print(null_cols)

for col in null_cols:
    if df[col].dtype == df['Weather'].dtype: # if datatype == categorical
        mode = df[col].mode()[0]

        df[col] = df[col].fillna(mode)
    else: # datatype == numeric
        df[col] = df[col].fillna(df[col].median())

print(df.isna().sum())

"""Dealing with cat values"""

non_numeric_column = df.select_dtypes(exclude=['number']).columns.tolist()
print(non_numeric_column)
#print(non_numeric_cols)

"""Dealing with categorical values"""

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for cat_col in non_numeric_column:
    df[cat_col] = le.fit_transform(df[cat_col])

df.dtypes

df['Accident'] = df['Accident'].astype('int64')
df.dtypes

for col in df.columns.to_list():
    col_var = np.var(df[col])
    print(f'{col}: {round(col_var, 2)}')

"""Feature Scaling"""

y = [0, 1]
std = np.std(y)
mean = np.mean(y)

print((0-mean)/std)
print((1-mean)/std)

features = df.drop('Accident', axis=1).columns.to_list()
print(features)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])
df.head()

for col in df.columns.to_list():
    col_var = np.var(df[col])
    print(f'{col}: {round(col_var, 2)}')

"""Dataset Splitting"""

df['Accident'].value_counts()

df_1 = df[df['Accident']==1]
df_0 = df[df['Accident']==0]

reduce_df_0 = df_0.sample(n=239, replace=False)

df = pd.concat([reduce_df_0, df_1])

df = df.sample(frac=1)
df['Accident'].value_counts()

df.shape

from sklearn.model_selection import train_test_split
x = df.drop(columns=['Accident'])
y = df['Accident']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=1, stratify=y)

"""Modeling
KNN
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import neighbors, datasets

model_1 = KNeighborsClassifier(n_neighbors=5)
model_1.fit(x_train, y_train)


y_pred = model_1.predict(x_test)
conf_matrix_1 = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
acc1 = accuracy_score(y_test, y_pred)
print(acc1)
print(conf_matrix_1)
print(report)

"""Decision Tree"""

from sklearn.tree import DecisionTreeClassifier


model2 = DecisionTreeClassifier()
model2.fit(x_train, y_train)

y_pred = model2.predict(x_test)
conf_matrix_2 = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
acc2 = accuracy_score(y_test, y_pred)
print(acc2)
print(conf_matrix_2)
print(report)

"""Logistic Regression"""

from sklearn.linear_model import LogisticRegression

model1 = LogisticRegression()
model1.fit(x_train, y_train)


y_pred = model1.predict(x_test)
conf_matrix_3 = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
acc3 = accuracy_score(y_test, y_pred)
print(acc3)
print(conf_matrix_3)
print(report)

"""Random Forest"""

from sklearn.ensemble import RandomForestClassifier

model3 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42 , criterion='entropy')
model3.fit(x_train, y_train)


y_pred = model3.predict(x_test)
conf_matrix_4 = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
acc4 = accuracy_score(y_test, y_pred)
print(acc4)
print(conf_matrix_4)
print(report)

df.shape

import matplotlib.pyplot as plt
# Accuracy values for the models (replace with your actual accuracy values)
models = ['KNN', 'Logistic Regression', 'Decision Tree', 'Random Forest']
accuracies = [acc1 , acc2 , acc3, acc4 ]
 # Example accuracy values
# Create the bar chart
plt.figure(figsize=(8, 6))
plt.bar(models, accuracies, color=['blue', 'green', 'orange', 'purple'], alpha=0.7)
# Add labels and title
plt.title('Prediction Accuracy of Models', fontsize=14)
plt.xlabel('Models', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.ylim(0.0, 1.0)
# Set y-axis limits for better visualization
plt.grid(axis='y', linestyle='--', alpha=0.7)
# Annotate accuracy values on top of the bars
for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f'{acc:.2f}', ha='center', fontsize=10)
# Show the plot
plt.tight_layout()
plt.show()

from sklearn.naive_bayes import GaussianNB

model5 = GaussianNB()
model5.fit(x_train, y_train)


y_pred = model5.predict(x_test)
conf_matrix_5 = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
acc5 = accuracy_score(y_test, y_pred)
print(acc5)
print(conf_matrix_5)
print(report)



"""### Support Vector Machine (SVM)"""

from sklearn.svm import SVC

model6 = SVC()
model6.fit(x_train, y_train)

y_pred = model6.predict(x_test)
conf_matrix_6 = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
acc6 = accuracy_score(y_test, y_pred)
print(acc6)
print(conf_matrix_6)
print(report)

import matplotlib.pyplot as plt
# Accuracy values for the models (replace with your actual accuracy values)
models = ['KNN', 'Logistic Regression', 'Decision Tree', 'Random Forest', 'Naive Bayes', 'SVM']
accuracies = [acc1 , acc2 , acc3, acc4, acc5, acc6 ]
 # Example accuracy values
# Create the bar chart
plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color=['blue', 'green', 'orange', 'purple', 'red', 'cyan'], alpha=0.7)
# Add labels and title
plt.title('Prediction Accuracy of Models', fontsize=14)
plt.xlabel('Models', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.ylim(0.0, 1.0)
# Set y-axis limits for better visualization
plt.grid(axis='y', linestyle='--', alpha=0.7)
# Annotate accuracy values on top of the bars
for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f'{acc:.2f}', ha='center', fontsize=10)
# Show the plot
plt.tight_layout()
plt.show()