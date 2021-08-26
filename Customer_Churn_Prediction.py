import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix , classification_report
import seaborn as sn
%matplotlib inline

df=pd.read_csv("Telco-Customer-Churn.csv")
#df.sample(5)
df.drop('customerID',axis='columns',inplace=True) #customerID for data analysis
#df.dtypes

#df.TotalCharges.values

#pd.to_numeric(df.TotalCharges) errors as it has space

#pd.to_numeric(df.TotalCharges,errors='coerce')  #if errors makes it NA

#pd.to_numeric(df.TotalCharges,errors='coerce').isnull() #rows which have spaces

#df.iloc[488] #integer location similar to array index

newdf= df[df.TotalCharges !=' '] #keep rows if df.TotalCharges !=' ' o/w drop it

#newdf.shape

newdf.TotalCharges = pd.to_numeric(newdf.TotalCharges)  #str to float

#newdf.TotalCharges

#for tenure

tenure_churn_no = newdf[newdf.Churn == "No"].tenure #no of months 
tenure_churn_yes = newdf[newdf.Churn == "Yes"].tenure

#newdf[newdf.Churn == "Yes"][newdf.tenure>60].shape
#newdf[newdf.Churn == "No"][newdf.tenure>60].shape


plt.hist([tenure_churn_yes,tenure_churn_no], color=['red','green'], label=['ChurnYes', 'ChurnNo']);
plt.xlabel("tenure(months)")
plt.ylabel("No of customers")
plt.title("Customer Churn Prediction Visualization with tenure",color ="blue")
plt.legend()

#for monthly charges

MC_churn_no = newdf[newdf.Churn == "No"].MonthlyCharges  #no of months 
MC_churn_yes = newdf[newdf.Churn == "Yes"].MonthlyCharges

plt.hist([MC_churn_yes,MC_churn_no], color=['red','green'], label=['ChurnYes', 'ChurnNo']);
plt.xlabel("MonthlyCharges")
plt.ylabel("No of customers")
plt.title("Customer Churn Prediction Visualization with MonthlyCharges",color ="blue")
plt.legend()

def print_unique_col_values(df):
    for column in df:
        if df[column].dtypes == 'object':
            print(f'{column} : {df[column].unique()}')
        
#print_unique_col_values(newdf)

#datacleaning
newdf.replace('No internet service','No',inplace =True)
newdf.replace('No phone service','No',inplace =True)
newdf['gender'].replace({'Female': 1, 'Male' : 0},inplace =True)

yes_no_columns = ['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport',
                  'StreamingTV','StreamingMovies','PaperlessBilling','Churn']

for column in yes_no_columns:
    newdf[column].replace({'Yes': 1 , 'No':0}, inplace = True)
    
def print_unique_values(df):   
    for col in newdf:
        print(f'{col} : {df[col].unique()}')
        
#print_unique_values(newdf)

newdf = pd.get_dummies(data=newdf,columns=['InternetService','Contract','PaymentMethod']) #one hot encoding & makes some cols  1,0
cols_to_scale = ['MonthlyCharges','TotalCharges','tenure']
scaler = MinMaxScaler()
newdf[cols_to_scale] = scaler.fit_transform(newdf[cols_to_scale])
#newdf.sample(3)

#train and test split
x=newdf.drop('Churn' ,axis='columns')
y=newdf['Churn']
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.2, random_state=5)
#X_train.shape
#Y_test.shape

model = keras.Sequential([
    keras.layers.Dense(20, input_shape=(26,), activation='relu'),
    keras.layers.Dense(1,activation='sigmoid'),
])

model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=['accuracy'])
model.fit(X_train, Y_train, epochs = 5)
#model.fit(X_train, Y_train, epochs = 100)
