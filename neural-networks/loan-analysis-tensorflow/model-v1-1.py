# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# %%
# import data and feature csv
data_info = pd.read_csv('lending_club_info.csv',index_col='LoanStatNew')
def feat_info(col_name): # feature information function
  print(data_info.loc[col_name]['Description'])

df = pd.read_csv('lending_club_loan_two.csv')

df.head()

# %%
sns.countplot(x='loan_status',data=df)

# %%
plt.figure(figsize=(12,4))
sns.displot(df['loan_amnt'],kde=False,bins=40)

# %%
df.corr().transpose()

# %%
plt.figure(figsize=(20,15))
sns.heatmap(df.corr(),annot=True,cmap='plasma')

# %%
# correlation close to 1 does not contribute to the model, as it acts similar as the target variable
feat_info('installment')
feat_info('loan_amnt')

# %%
plt.figure(figsize=(12,5))
sns.scatterplot(x='installment',y='loan_amnt',data=df)

# %%
sns.boxplot(x='loan_status',y='loan_amnt',data=df)
# if the loan ammnt is higher theres a higher chance that it will be charged off

# %%
df.groupby('loan_status')['loan_amnt'].describe()

# %%
# unique possible grades and subgrades of customers
df['grade'].unique()

# %%
df['sub_grade'].unique()

# %%
sns.countplot(x='grade',data=df,hue='loan_status')

# %%
plt.figure(figsize=(12,4))
subgrade_order = sorted(df['sub_grade'].unique()) # sorted call of unique subgrades
sns.countplot(x='sub_grade',data=df,order=subgrade_order,palette='coolwarm',hue='loan_status')

# %%
# F and G graded loans are not paid back often -- analyze these grades in greater detail
f_and_g = df[(df['grade']=='G') | (df['grade']=='F')]

plt.figure(figsize=(12,4))
subgrade_order = sorted(f_and_g['sub_grade'].unique()) # sorted call of unique subgrades
sns.countplot(x='sub_grade',data=f_and_g,order=subgrade_order,palette='coolwarm',hue='loan_status')

# %%
# creating a new column 'loan_repaid' where 0 == not repaid, 1 == fully repaid
df['loan_repaid'] = df['loan_status'].map({'Fully Paid':1,'Charged Off':0})

# %%
df[['loan_repaid','loan_status']]

# %%
df.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar')
# correlation of attributes with wether of not a loan is repaid

# %%
# data preprocessing and data cleaning

# %%
df.isnull().sum() # sum of missing values in each column

# %%
100 * df.isnull().sum() / len(df) # missing values in % 

# %%
# employnment title and employnment lenght <= can we drop them?
# employnment title == job title
# employnment lenght == employnment lenght in years
df['emp_title'].value_counts()

# %%
df = df.drop('emp_title',axis=1)
# too many individual/unique categoties, we cannot replace them with a categorical variable

# %%
sorted(df['emp_length'].dropna().unique())

# %%
emp_length_order = ['< 1 year','1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years']

# %%
plt.figure(figsize=(12,4))
sns.countplot(x='emp_length',data=df,order=emp_length_order,hue='loan_status')
# if the blue-orange ratio is similar in all categories, then it is not an important variable for the NN

# %%
emp_co = df[df['loan_status']=='Charged Off'].groupby('emp_length').count()['loan_status']
emp_fp = df[df['loan_status']=='Fully Paid'].groupby('emp_length').count()['loan_status']

# %%
# % between charged off / fully paid as of emp_len
emp_co/(emp_fp+emp_co)

# %%
emp_len = emp_co/(emp_fp+emp_co)

# %%
emp_len.plot(kind='bar') # the difference is not extreme enough to validate this feature => drop

# %%
df = df.drop('emp_length',axis=1)

# %%
df.isnull().sum()

# %%
df['title'].head() # subcategory desc. of the purpose

# %%
df = df.drop('title',axis=1)

# %%
feat_info('mort_acc')

# %%
df['mort_acc'].value_counts()

# %%
# wich of other features correlates highly with mort_acc, so that we can use the information stored in that variable
df.corr()['mort_acc'].sort_values()

# %%
# total account has a reasonable positive correlation -- can use it to fill in missing values
# df.groupby('total_acc').mean() # averages of different categories in total account
df.groupby('total_acc').mean()['mort_acc'] # replace of mort acc values based on total account means

# %%
total_acc_avg = df.groupby('total_acc').mean()['mort_acc']

# %%
# filling out missing values in mort_acc
def fill_mort_acc(total_acc,mort_acc):
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc

# %%
df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'],x['mort_acc']),axis=1)

# %%
df.isnull().sum()

# %%
df = df.dropna() # the rest is not significant, it is a very little part of the database

# %%
df.isnull().sum()

# %% [markdown]
# Data cleaning: categorical data, string data

# %%
# listing the non-numeric columns
df.select_dtypes(['object']).columns

# %%
feat_info('term')

# %%
dummies = pd.get_dummies(df[['verification_status','application_type','initial_list_status','purpose']],drop_first=True)
df = pd.concat([df.drop(['verification_status','application_type','initial_list_status','purpose'],axis=1),dummies],axis=1)

# %%
df['home_ownership'].value_counts()
# None and any is very few -- we can put those in the other category
# value counts has been ran again, after the replace function...

# %%
# replace non and any with other
df['home_ownership'] = df['home_ownership'].replace(['NONE','ANY'],'OTHER')

# %%
dummies = pd.get_dummies(df['home_ownership'],drop_first=True)
df = pd.concat([df.drop(['home_ownership'],axis=1),dummies],axis=1)

# %%
df['address']

# %%
# extract the zipcode from the address attribute
df['address'].apply(lambda address:address[-5:])

# %%
df['zip_code'] = df['address'].apply(lambda address:address[-5:])

# %%
df['zip_code'].value_counts()

# %%
dummies = pd.get_dummies(df['zip_code'],drop_first=True)
df = pd.concat([df.drop(['zip_code'],axis=1),dummies],axis=1)

# %%
df = df.drop('address',axis=1)

# %%
feat_info('issue_d')

# %%
# when we decide who to give a loan, we do not have a date about the fund
df = df.drop('issue_d',axis=1)

# %%
feat_info('earliest_cr_line')

# %%
# converting it to a date feature, or grab is based off its position
df['earliest_cr_line']

# %%
df['earliest_cr_line'] = df['earliest_cr_line'].apply(lambda date: int(date[-4:]))

# %%
df['earliest_cr_line']

# %%
df = df.drop('grade',axis=1)

# %%
subgrade_dummies = pd.get_dummies(df['sub_grade'],drop_first=True)

# %%
df = pd.concat([df.drop('sub_grade',axis=1),subgrade_dummies],axis=1)

# %%
dummies = pd.get_dummies(df['term'],drop_first=True)
df = pd.concat([df.drop(['term'],axis=1),dummies],axis=1)

# %%
dummies = pd.get_dummies(df['loan_status'],drop_first=True)
df = pd.concat([df.drop(['loan_status'],axis=1),dummies],axis=1)

# %%
df.head(5)

# %% [markdown]
# **Data preprocessing**

# %%
from sklearn.model_selection import train_test_split

# %%
# this has already been dropped in a previous runtime
# df = df.drop('loan_status',axis=1) # dropping the loan status, as loan repaid already has that information

# %%
X = df.drop('loan_repaid',axis=1).values
y = df['loan_repaid'].values

# %%
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2,random_state=101)

# %%
from sklearn.preprocessing import MinMaxScaler

# %%
scaler = MinMaxScaler()

# %%
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %% [markdown]
# **Creating the NN model**

# %%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout
from tensorflow.keras.constraints import max_norm

# %%
model = Sequential() # instance of a model

# %%
model.add(Dense(78,activation='relu')) # input layer
model.add(Dropout(0.2))
model.add(Dense(78,activation='relu')) 
model.add(Dropout(0.2))
model.add(Dense(78,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam')

# %%
model.fit(
    x=X_train,
    y=y_train,
    epochs=25,
    validation_data=(X_test,y_test),
)

# %%
losses = pd.DataFrame(model.history.history)

# %%
losses[['loss','val_loss']].plot()

# %%
from sklearn.metrics import confusion_matrix,classification_report

# %%
predictions = model.predict(X_test)

# %%
predictions = predictions.astype(np.int64)

# %%
print(classification_report(predictions,y_test))

# %%
print(confusion_matrix(predictions,y_test))

# %%



