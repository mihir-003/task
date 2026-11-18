
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


df = pd.read_csv("D:\\ML-R\\task\\Churn_Modelling.csv")


df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)


new_gender = df.groupby('Gender')['Exited'].sum().reset_index()
print(new_gender)
print()
new_gender.to_csv("new_Gender.csv", index=False)


new_geography = df.groupby('Geography')['Exited'].sum().reset_index()
print(new_geography)
print()
new_geography.to_csv("new_Geography.csv", index=False)


le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])


df = pd.get_dummies(df, columns=['Geography'], drop_first=True)


scaler = MinMaxScaler()
new_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)


new_df.to_csv("new_Churn_Modelling.csv", index=False)
print(new_df.head())
