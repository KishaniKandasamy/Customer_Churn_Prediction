import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
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
