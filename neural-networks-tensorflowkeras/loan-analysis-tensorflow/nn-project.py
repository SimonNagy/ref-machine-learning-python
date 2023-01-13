import pandas as pd
import numpy as np
import seaborn as sns
import maptplotlib.pyplot as plt

data_info = pd.read_csv('lending_club_info.csv',index_col='LoanStatNew')
def feat_info(col_name): # feature information function
  print(data_info.loc[col_name]['Description'])

df = pd.read_csv('lending_club_loan_two.csv')

#visualizations
sns.countplot(x='loan_status',data=df)

plt.figure(figsize=(12,4))
sns.displot(df['loan_amnt'],kde=False,bins=40)

df.corr().transpose()

plt.figure(figsize=(20,15))
sns.heatmap(df.corr(),annot=True,cmap='plasma')

# correlation close to 1 does not contribute to the model, as it acts similar to the label
feat_info('installment')

feat_info('loan_amnt')

sns.scatterplot(x='installment',y='loan_amnt',data=df)

sns.boxplot(x='loan_status',y='loan_amnt',data=df)
# if the loan amnt is higher theres a higher amount that it will be charged off

df.groupby('loan_status')['loan_amnt'].describe()

df['grade'].unique()

df['sub_grade'].unique()

sns.countplot(x='grade',data=df,hue='loan_status')

plt.figure(figsize=(12,4))
subgrade_order = sorted(df['sub_grade'].unique()) # sorted call of the unique subgrades
sns.countplot(x='sub_grade',data=df,order=subgrade_order,palette='coolwarm',hue='loan_status')

# filter data to be displayed
f_and_g = df[(df['grade']=='G')|(df['grade']=='F')]
plt.figure(figsize=(12,4))
subgrade_order = sorted(f_and_g['sub_grade'].unique())
sns.countplot(x='sub_grade',data=f_and_g,order=subgrade_order,palette='coolwarm',hue='loan_status')

# data transformation <= loan repaid && loan status
df['loan_repaid'] = df['loan_status'].map({'Fully Paid':1,'Charged Off':0})

plt.figure(figsize=(12,5))
df.corr()['loan_repaid'].sort_values().drop('loan_repaid').drop('loan_status').plot(kind='bar')