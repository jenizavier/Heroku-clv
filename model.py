# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 23:51:46 2022

@author: DELL
"""

from __future__ import division

from datetime import datetime,timedelta,date
import pandas as pd
%matplotlib inline
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
#pip install plotly
import plotly as py
import plotly.offline as pyoff
import plotly.graph_objs as go

import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split


df = pd.read_excel("E:\PROJECT 2\CreditAnalysis_data.xlsx")
#initate plotly
pyoff.init_notebook_mode()
df.head()
df.isnull().sum(axis=0)
df.columns
#converting the type of Invoice Date Field from string to datetime.

df['created'] = pd.to_datetime(df['created'])
#creating YearMonth field for the ease of reporting and visualization
df['InvoiceYearMonth'] = df['created'].map(lambda date: 100*date.year + date.month)
df.describe()
#remove missing values
df=df[pd.notnull(df["ordereditem_product_id"])]
#records with positive values kept and omitting negative values
df=df[(df["ordereditem_quantity"]>0)]
#Add new column which shows total sales
df["Total_Sales"]=df["ordereditem_quantity"]*df["ordereditem_unit_price_net"]
needed_Cols=["order_id","InvoiceYearMonth","Total_Sales"]
df_new=df[needed_Cols]
df_new.head()

#print records ofunique order_ids
df_new["order_id"].nunique()
#Las order date
Last_Order_Date=df_new["InvoiceYearMonth"].max()
#pip install lifetimes

from lifetimes.utils import summary_data_from_transaction_data
lf_df_new=summary_data_from_transaction_data(df_new,"order_id","InvoiceYearMonth",monetary_value_col="Total_Sales",observation_period_end="2018-12")
df['group'].value_counts()
#to calculate which state has most records

df_ap = df.query("group=='Hyderabad'").reset_index(drop=True)
df_hy = df.query("group=='Gurugram'").reset_index(drop=True)
df_Dl = df.query("group=='Delhi-West '").reset_index(drop=True)

#RFM
#To calculate recency, we need to find out most recent purchase date of each customer and see how many days they are inactive for. After having no. of inactive days for each customer, we will apply K-means* clustering to assign customers a recency score.
#create a generic user dataframe to keep CustomerID and new segmentation scores
df_user = pd.DataFrame(df['order_id'].unique())
df_user.columns = ['order_id']
df_user.head()

#Since we are calculating recency, we need to know when last the person bought something. Let us calculate the last date of transaction for a person.
#get the max purchase date for each customer and create a dataframe with it
df_max_purchase = df_ap.groupby('order_id').created.max().reset_index()
df_max_purchase.columns = ['order_id','MaxPurchaseDate']
df_max_purchase.head()


df_max_purchase['Recency'] = (df_max_purchase['MaxPurchaseDate'].max() - df_max_purchase['MaxPurchaseDate']).dt.days
df_max_purchase.head()
#merge this dataframe to our new user dataframe
df_user = pd.merge(df_user, df_max_purchase[['order_id','Recency']], on='order_id')
df_user.head()


#Assigning a recency score
#We are going to apply K-means clustering to assign a recency score. But we should tell how many clusters we need to K-means algorithm. To find it out, we will apply Elbow Method. Elbow Method simply tells the optimal cluster number for optimal inertia. Code snippet and Inertia graph are as follows:from sklearn.cluster import KMeans

sse={} # error
df_recency = df_user[['Recency']]
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df_recency)
    df_recency["clusters"] = kmeans.labels_  #cluster names corresponding to recency values 
    sse[k] = kmeans.inertia_ #sse corresponding to clusters
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.show()

#Here it looks like 3 is the optimal one. Based on business requirements, we can go ahead with less or more clusters. We will be selecting 4 for this example
    
#build 4 clusters for recency and add it to dataframe
kmeans = KMeans(n_clusters=4)
df_user['RecencyCluster'] = kmeans.fit_predict(df_user[['Recency']])
df_user.head()   
    
df_user.groupby('RecencyCluster')['Recency'].describe() 
#ordering clusters
#function for ordering cluster numbers
def order_cluster(cluster_field_name, target_field_name,data,ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    data_new = data.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    data_new = data_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    data_new['index'] = data_new.index
    data_final = pd.merge(data,data_new[[cluster_field_name,'index']], on=cluster_field_name)
    data_final = data_final.drop([cluster_field_name],axis=1)
    data_final = data_final.rename(columns={"index":cluster_field_name})
    return data_final

df_user = order_cluster('RecencyCluster', 'Recency',df_user,False)
df_user.tail()
df_user.groupby('RecencyCluster')['Recency'].describe()
##Cluster 0 is most inactive and cluster 3 is most active

#FREQUENCY
#To create frequency clusters, we need to find total number orders for each customer. First calculate this and see how frequency look like in our customer database
#get order counts for each user and create a dataframe with it

#get order counts for each user and create a dataframe with it
df_frequency = df_ap.groupby('order_id').InvoiceYearMonth.count().reset_index()
df_frequency.columns = ['order_id','Frequency']
df_frequency.head() #how many orders does a customer have

#add this data to our main dataframe
df_user = pd.merge(df_user, df_frequency, on='order_id')

df_user.head()
#Frequency clusters
#Determine the right number of clusters for K-Means by elbow method

from sklearn.cluster import KMeans

sse={} # error
df_recency = df_user[['Frequency']]
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df_recency)
    df_recency["clusters"] = kmeans.labels_  #cluster names corresponding to recency values 
    sse[k] = kmeans.inertia_ #sse corresponding to clusters
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.show()

# Applying k-Means
kmeans=KMeans(n_clusters=4)
df_user['FrequencyCluster']=kmeans.fit_predict(df_user[['Frequency']])

#order the frequency cluster
df_user = order_cluster('FrequencyCluster', 'Frequency', df_user, True )
df_user.groupby('FrequencyCluster')['Frequency'].describe()
#Cluster with max frequency is cluster 3, least frequency cluster is cluster 0.
df.columns
#Revenue
#Let’s see how our customer database looks like when we cluster them based on revenue. We will calculate revenue for each customer, plot a histogram and apply the same clustering method.
#calculate revenue for each customer
df_ap['Revenue'] = df_ap['ordereditem_unit_price_net'] * df_ap['ordereditem_quantity']
df_revenue = df_ap.groupby('order_id').Revenue.sum().reset_index()
df_revenue.head()
#merge it with our main dataframe
df_user = pd.merge(df_user, df_revenue, on='order_id')
df_user.head()
# We have some customers with negative revenue as well. Let’s continue and apply k-means clustering:

# Elbow method to find out the optimum number of clusters for K-Means

from sklearn.cluster import KMeans

sse={} # error
df_recency = df_user[['Revenue']]
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df_recency)
    df_recency["clusters"] = kmeans.labels_  #cluster names corresponding to recency values 
    sse[k] = kmeans.inertia_ #sse corresponding to clusters
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.show()
#From elbow's method, we find that clusters can be 3 or 4. Lets take 4 as the number of clusters

 #Revenue clusters
 
 #apply clustering
kmeans = KMeans(n_clusters=4)
df_user['RevenueCluster'] = kmeans.fit_predict(df_user[['Revenue']])

#order the cluster numbers
df_user = order_cluster('RevenueCluster', 'Revenue',df_user,True)

#show details of the dataframe
df_user.groupby('RevenueCluster')['Revenue'].describe()
    
#Cluster 3 has max revenue, cluster 0 has lowest revenue
#Overall Score based on RFM Clsutering
#We have scores (cluster numbers) for recency, frequency & revenue. Let’s create an overall score out of them
#calculate overall score and use mean() to see details
df_user['OverallScore'] = df_user['RecencyCluster'] + df_user['FrequencyCluster'] + df_user['RevenueCluster']
df_user.groupby('OverallScore')['Recency','Frequency','Revenue'].mean()

#Score 8 is our best customer, score 0 is our worst customer.
df_user['Segment'] = 'Low-Value'
df_user.loc[df_user['OverallScore']>2,'Segment'] = 'Mid-Value' 
df_user.loc[df_user['OverallScore']>4,'Segment'] = 'High-Value' 
df_user


# filtered_data=df[['group',"order_id"]].drop_duplicates()
# filtered_data.group.value_counts()[:10].plot(kind='bar')

##CUSTOMER LIFE TIME VALUE
#Since our feature set is ready, let’s calculate 6 months LTV for each customer which we are going to use for training our model.
# Lifetime Value: Total Gross Revenue - Total Cost
# There is no cost specified in the dataset. That’s why Revenue becomes our LTV directly.

df_ap.head()
df_ap['created'].describe()

#We see that customers are active from 18 December 2017. 
#Let us consider customers from March onwards (so that they are not new customers). We shall divide them into 2 subgroups. One will be where timeframe of analysing is 3 months, another will be timeframe of 6 months.

df_3m = df_ap[(df_ap.created.dt.date < date(2018,6,1)) & (df_ap.created.dt.date >= date(2018,3,1))].reset_index(drop=True) #3 months time
df_6m = df_ap[(df_ap.created.dt.date >= date(2018,6,1)) & (df_ap.created.dt.date < date(2018,12,1))].reset_index(drop=True) # 6 months time
df_6m.columns
#calculate revenue and create a new dataframe for it
df_6m['Revenue'] = df_6m['ordereditem_unit_price_net'] * df_6m['ordereditem_quantity']
df_user_6m = df_6m.groupby('order_id')['Revenue'].sum().reset_index()
df_user_6m.columns = ['order_id','m6_Revenue']

df_user_6m.head()
df_user_6m[["m6_Revenue"]].max()
#plot LTV histogram
# import plotly
# import plotly.offline as pyoff
# import plotly.figure_factory as ff
# from plotly.offline import init_notebook_mode, iplot, plot
# import plotly.graph_objs as go

import plotly.io as pio
pio.renderers.default = 'browser'
plot_data = [
    go.Histogram(
        x=df_user_6m['m6_Revenue']
    )
]

plot_layout = go.Layout(
        title='6m Revenue'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
fig.show()
#Histogram clearly shows we have customers with negative LTV. 
#We have some outliers too. Filtering out the outliers makes sense 
#to have a proper machine learning model.

df_user_6m.boxplot('m6_Revenue')
from feature_engine.outliers import Winsorizer as win


winsor = win(capping_method='iqr',  tail='right', fold=1.5, variables=["m6_Revenue"])

df_user_6m['m6_Revenue'] = winsor.fit_transform(df_user_6m[['m6_Revenue']])
df_user_6m.boxplot('m6_Revenue')


pio.renderers.default = 'browser'
plot_data = [
    go.Histogram(
        x=df_user_6m['m6_Revenue']
    )
]

plot_layout = go.Layout(
        title='6m Revenue'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
fig.show()
#Ok, next step. We will merge our 3 months and df_ap and also merge 6 months dataframe and df_ap to see correlations between LTV and the feature set we have.
df_user.head()
df_ap.head()
df_user_6m.columns
df_user_6m[["m6_Revenue"]].max()
df_user.columns
df_merge = pd.merge(df_user, df_user_6m, on="order_id", how='left') #Only people who are in the timeline of df_user_6m
df_merge = df_merge.fillna(0)

df_graph = df_merge.query("m6_Revenue < 3446") #because max values are ending at 3446 as seen in graph above

plot_data = [
    go.Scatter(
        x=df_graph.query("Segment == 'Low-Value'")['OverallScore'],
        y=df_graph.query("Segment == 'Low-Value'")['m6_Revenue'],
        mode='markers',
        name='Low',
        marker= dict(size= 7,
            line= dict(width=1),
            color= 'blue',
            opacity= 0.8
           )
    ),
    go.Scatter(
        x=df_graph.query("Segment == 'Mid-Value'")['OverallScore'],
        y=df_graph.query("Segment == 'Mid-Value'")['m6_Revenue'],
        mode='markers',
        name='Mid',
        marker= dict(size= 9,
            line= dict(width=1),
            color= 'green',
            opacity= 0.5
           )
    ),
        go.Scatter(
        x=df_graph.query("Segment == 'High-Value'")['OverallScore'],
        y=df_graph.query("Segment == 'High-Value'")['m6_Revenue'],
        mode='markers',
        name='High',
        marker= dict(size= 11,
            line= dict(width=1),
            color= 'red',
            opacity= 0.9
             )
    ),
]

plot_layout = go.Layout(
        yaxis= {'title': "6m LTV"},
        xaxis= {'title': "RFM Score"},
        title='LTV'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
fig.show()

#We can visualise correlation between overall RFM score and revenue. Positive correlation is quite visible here. High RFM score means high LTV.

# Considering business part of this analysis, we need to treat customers differently based on their predicted LTV. For this example, we will apply clustering and have 3 segments (number of segments really depends on your business dynamics and goals):

# Low LTV
# Mid LTV
# High LTV
# We are going to apply K-means clustering to decide segments and observe their characteristics:
    
#creating 3 clusters
kmeans = KMeans(n_clusters=3)
kmeans.fit(df_merge[['m6_Revenue']])
df_merge['LTVCluster'] = kmeans.predict(df_merge[['m6_Revenue']])

#order cluster number based on LTV
df_merge = order_cluster('LTVCluster', 'm6_Revenue',df_merge,True)

#creatinga new cluster dataframe
df_cluster = df_merge.copy()

#see details of the clusters
df_cluster.groupby('LTVCluster')['m6_Revenue'].describe()
#cluster2 is the best with average 3212 LTV whereas 0 is the worst with 11.
# There are few more step before training the machine learning model:

# Need to do some feature engineering. We should convert categorical columns to numerical columns.
# We will check the correlation of features against our label, LTV clusters.
# We will split our feature set and label (LTV) as X and y. We use X to predict y.
# Will create Training and Test dataset. Training set will be used for building the machine learning model. We will apply our model to Test set to see its real performance.
    
#convert categorical columns to numerical



# df_class = pd.get_dummies(df_cluster)

#calculate and show correlations
corr_matrix = df_class.corr()
corr_matrix['LTVCluster'].sort_values(ascending=False)

#create X and y, X will be feature set and y is the label - LTV
X = df_cluster.iloc[:,[1,2,3,4,5,6,7,8]]

y = df_cluster.iloc[:,[-1]]
##calculate and show correlations
corr_matrix = df_cluster.corr()
corr_matrix['LTVCluster'].sort_values(ascending=False)
##convert categorical columns to numerical
def convert_to_int(word):
    word_dict = {'Low-Value':0, 'Mid-Value':1, 'High-Value':2}
    return word_dict[word]

X["Segment"] = X['Segment'].apply(lambda x : convert_to_int(x))

#split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=56)

#XGBoost Multiclassification Model
ltv_xgb_model = xgb.XGBClassifier(max_depth=5, learning_rate=0.1,objective= 'multi:softprob',n_jobs=-1).fit(X_train, y_train)

print('Accuracy of XGB classifier on training set: {:.2f}'
       .format(ltv_xgb_model.score(X_train, y_train)))
print('Accuracy of XGB classifier on test set: {:.2f}'
       .format(ltv_xgb_model.score(X_test[X_train.columns], y_test)))

y_pred = ltv_xgb_model.predict(X_test)
print(classification_report(y_test, y_pred))    

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier

models= {
    "Logistic Regression":LogisticRegression(),
    "Multilinear Regression":LinearRegression(),
    "k-Nearest Neighbors":KNeighborsClassifier(),
    "Decision Tree":DecisionTreeClassifier(),
    "Support Vector Machine(Linear Kernal)": LinearSVC(),
    "Support Vector Machine(RBF Kernal)": SVC(),
    "Nureal Network":MLPClassifier(),
    "Random Forest":RandomForestClassifier(),
    "Gredient Boosting":GradientBoostingClassifier()
}

for name, model in models.items():
    model.fit(X_train,y_train)
    print(name+': trained')
    
    
    
for name , model in models.items():
    print(name+ ": {:.2f}%".format(model.score(X_test,y_test)*100))

for name , model in models.items():
    print(name+ ": {:.2f}%".format(model.score(X_train,y_train)*100))
    
    ################
import pickle
# open a file, where you ant to store the data
pickle_out = open('model.pkl', 'wb')

# dump information to that file
pickle.dump(model, pickle_out)
pickle_out.close()










