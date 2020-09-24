# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 21:27:32 2020

@author: Yogeshwar 
"""

#Packages
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import sklearn.neighbors as skln
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.ensemble import AdaBoostClassifier

#loading data file
hotel = pd.read_csv('C:\Purdue\Spring 2020\CS 578\Project\hotel\hotel_bookings.csv')


#Data preprocessing
hotel_upd = hotel[hotel['adults']!=0]

#creating new feature
def f(row):
    if row['reserved_room_type'] == row['assigned_room_type']:
        val = 1
    else:
        val = 0
    return val

hotel_upd['reserved_eq_assigned_roomtype'] = hotel_upd.apply(f, axis=1)

def f2(row):
    if row['adr']<=0:
        val = 0
    else:
        val = row['adr']
    return val
    
hotel_upd['adr'] = hotel_upd.apply(f2, axis=1)

#creating new column for Checkout 
def f3(row):
    if row['reservation_status']=='Check-Out':
        val = 1
    else:
        val = 0
    return val

hotel_upd['CheckOut'] = hotel_upd.apply(f3, axis=1)

#removing NA values
def f4(row):
    if row['children']=='NA':
        val = 0
    else:
        val = row['adr']
    return val
    
hotel_upd['children'] = hotel_upd.apply(f4, axis=1)

#define accuracy measure(reservation_status = predicted reservation status)
def accuracy_score(row):
    if row[0]==row[1]:
        val=1
    else:
        val=0
    return val

#creating function to obtain ROC parameters
def calcRates(y_vals,probs):
    
    #convert to arrays
    y_vals = np.array(y_vals)
    probs = np.array(probs)

    # sort the indexs by their probabilities
    index = np.argsort(probs, kind="quicksort")[::-1]
    probs = probs[index]
    y_vals = y_vals[index]

    #Grab indices with distinct values
    d_index = np.where(np.diff(probs))[0]
    t_holds = np.r_[d_index, y_vals.size - 1]

    # sum up with true positives       
    tps = np.cumsum(y_vals)[t_holds]
    tpr = tps/tps[-1]
    #calculate the false positive
    fps = 1 + t_holds - tps
    fpr = fps/fps[-1]
    return fpr, tpr

#feature selection

#mutual information

col_cat = ['hotel','arrival_date_month','meal','market_segment','distribution_channel','deposit_type','customer_type','reserved_eq_assigned_roomtype']
info_list=list()
join_coly = 'reservation_status'

#marginal prob of response
marproby = hotel_upd.groupby(['reservation_status']).size().reset_index().rename(columns={0:'marcounty'})
marproby['marproby']= marproby['marcounty']/len(hotel_upd)


for i in col_cat:
        
    #marginal prob of predictor
    marprobx = hotel_upd.groupby([i]).size().reset_index().rename(columns={0:'marcountx'})
    marprobx['marprobx']= marprobx['marcountx']/len(hotel_upd)

    #joint probabilities 
    joinprob = hotel_upd.groupby([i,'reservation_status']).size().reset_index().rename(columns={0:'joincount'})
    joinprob['joinprob']= joinprob['joincount']/len(hotel_upd)

    #merging on column of interest
    join_colx = joinprob.columns[0]

    totalprob = pd.merge(joinprob,marprobx,on=join_colx,how='left')
    totalprob = pd.merge(totalprob,marproby,on=join_coly,how='left')

    #information calculation
    totalprob['info'] = totalprob['joinprob']*(np.log(totalprob['joinprob']/(totalprob['marprobx']*totalprob['marproby'])))
    info_list.append(sum(totalprob['info']))

col_cat=pd.DataFrame(col_cat)
info_list=pd.DataFrame(info_list)
frames = [col_cat, info_list]
mutualinfo = pd.concat(frames, axis=1)
mutualinfo.columns = ('categorical_feature','mutual_info')


#dummy variables columns
hotel_upd[['mar_Aviation','mar_Complementary','mar_Corporate','mar_Direct','mar_Groups','mar_Offline TA/TO','mar_Online TA','mar_Undefined']]=pd.get_dummies(hotel_upd['market_segment'])
hotel_upd[['dis_Corporate','dis_Direct','dis_GDS','dis_TA/TO','dis_Undefined']]=pd.get_dummies(hotel_upd['distribution_channel'])
hotel_upd[['No_Deposit','Non_Refund','Refundable']]=pd.get_dummies(hotel_upd['deposit_type'])


#removing irrelevent columns and those with mutual information <0.01
hotel_upd=hotel_upd.drop(['hotel','is_canceled','arrival_date_year','arrival_date_month','arrival_date_week_number','arrival_date_day_of_month','meal','country','market_segment','distribution_channel','reserved_room_type','assigned_room_type','deposit_type','agent','company','customer_type','reservation_status_date','CheckOut'],axis=1)
X= hotel_upd.drop(['reservation_status'], axis=1)
y=hotel_upd['reservation_status']

#checking datatypes
print(hotel_upd.dtypes)

#train & test
hotel_train, hotel_test = train_test_split(hotel_upd, test_size=0.2, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



#count of response variable
hotel_train.groupby('reservation_status').count()

hotel_test.groupby('reservation_status').count()


####################### perceptron 
mean_list=list()
eta_list=list()

#looping over different values of eta (hyperparameter)
for eta in range(1, 11):
    eta_list.append(eta/5)
    per1 = Perceptron(fit_intercept=True,max_iter=1000,eta0=eta/5,random_state=0)
    
    #bootstrapping
    B=5
    B_list= list()

    for i in range(0,B):
        #creating training data via bootstrapping
        B_train = hotel_train.sample(frac=0.8,replace=True)
        BX_train = B_train.drop(['reservation_status'], axis=1)
        BY_train = B_train['reservation_status']
        
        #remaining samples go to cross validation set
        B_cross = hotel_train[~hotel_train.index.isin(B_train.index)]
        BX_cross = B_cross.drop(['reservation_status'], axis=1)
        BY_cross = B_cross['reservation_status']
    
        #fitting multiclass perceptron 
        per1.fit(BX_train,BY_train)  
        
        #predict on cross validation set
        per1_pred = pd.DataFrame(per1.predict(BX_cross))
        
        #store actual value of reservation_status in dataframe
        BY_cross = (pd.DataFrame(BY_cross)).reset_index(drop=True)
        
        #append both (do not change order)
        frames = [BY_cross, per1_pred]
        per_result = pd.concat(frames,axis=1,ignore_index=True)
        
        #applying accuracy score function and calculating accuracy
        per_result['score']=per_result.apply(accuracy_score,axis=1)
        M=sum(per_result['score'])/len(per_result)

        B_list.append(M)
    
    #mean accuracy for one value of eta    
    M_mean=sum(B_list)/len(B_list)
    
    #appending mean accuracy in list
    mean_list.append(M_mean)


#plot of eta vs mean accuracy
plt.scatter(eta_list,mean_list)
plt.title('Hyperparameter Tuning using Bootstrapping')
plt.xlabel('Eta: Constant by which the updates are multiplied')
plt.ylabel('Cross Validation Set Accuracy')
plt.plot(eta_list, mean_list, '-o')


#using value of eta=0.4 since it has highest cross validation accuracy
per1 = Perceptron(fit_intercept=True,max_iter=1000,eta0=0.4,random_state=0)
per1.fit(x_train,y_train)


#predicting on test set
per1_pred = pd.DataFrame(per1.predict(x_test))

#store actual value of reservation_status in dataframe
y_test_per = (pd.DataFrame(y_test)).reset_index(drop=True)
        
#append both (do not change order)
frames = [y_test_per, per1_pred]
per_result = pd.concat(frames,axis=1,ignore_index=True)

#applying accuracy score function and calculating accuracy
per_result['score']=per_result.apply(accuracy_score,axis=1)
M=sum(per_result['score'])/len(per_result)
print(M)
#75.37 % test accuracy



#################  adaboost using decision stump 

#using k fold cross validation to find best #estimators to use
#creating folds
k=5
N=len(hotel_train)
Min_index=0
folds=list()
for i in range(1,k+1):
    if (i==k+1):
        Max_index=(round(N/k))*i+(N % k) 
    else:
        Max_index=(round(N/k))*i
        if(Max_index>N):
            Max_index=N-1
        folds.append(range((Min_index+1),Max_index+1))
        Min_index=Max_index 
print(folds)

kmean_list=list()
n_est_list=list()

#looping over different values of n_estimators (hyperparameter)
for q in range(1, 19):
    if (q<11):
        n_est=q
    else:
        n_est=((q-10)*5)+10
    print(n_est)
    n_est_list.append(n_est)
    ada = AdaBoostClassifier(n_estimators=n_est)
    
    K_list= list()
    for j in range(0,k): 
        K_cross=hotel_train.iloc[folds[j]] 
        K_train=[] 
        for i in range(0,k):
            if (i!=j): 
                K_train.append(hotel_train.iloc[folds[i]])     
        K_train=pd.concat(K_train)
        K_train=pd.DataFrame(K_train)
        KX_train=K_train.drop(["reservation_status"],axis=1)
        KY_train=K_train["reservation_status"]
        K_cross=pd.DataFrame(K_cross)
        KX_cross=K_cross.drop(["reservation_status"],axis=1)
        KY_cross=K_cross["reservation_status"]
    
        #fitting adaboost 
        ada.fit(KX_train,KY_train)  
        
        #predict on cross validation set
        ada_pred = pd.DataFrame(ada.predict(KX_cross))
        
        #store actual value of reservation_status in dataframe
        KY_cross = (pd.DataFrame(KY_cross)).reset_index(drop=True)
        
        #append both (do not change order)
        frames = [KY_cross, ada_pred]
        ada_result = pd.concat(frames,axis=1,ignore_index=True)
        
        #applying accuracy score function and calculating accuracy
        ada_result['score']=ada_result.apply(accuracy_score,axis=1)
        M_ada=sum(ada_result['score'])/len(ada_result)

        K_list.append(M_ada)
    
    #mean accuracy for one value of n_estimator    
    K_mean=sum(K_list)/len(K_list)
    
    #appending mean accuracy in list
    kmean_list.append(K_mean)

#plot of eta vs mean accuracy
plt.scatter(n_est_list,kmean_list)
plt.title('Hyperparameter Tuning using K Fold Cross Validation')
plt.xlabel('n_estimators: #base learners')
plt.ylabel('Cross Validation Set Accuracy')
plt.plot(n_est_list, kmean_list, '-o')

#using value of n_estimator=20 since it has highest cross validation accuracy
ada = AdaBoostClassifier(n_estimators=20)
ada.fit(x_train,y_train)

ada.base_estimator_
ada.feature_importances_
ada.classes_

#predicting on test set
ada_pred = pd.DataFrame(ada.predict(x_test))

#store actual value of reservation_status in dataframe
y_test_ada = (pd.DataFrame(y_test)).reset_index(drop=True)
        
#append both (do not change order)
frames_ada = [y_test_ada, ada_pred]
ada_result = pd.concat(frames_ada,axis=1,ignore_index=True)

#applying accuracy score function and calculating accuracy
ada_result['score']=ada_result.apply(accuracy_score,axis=1)
M_ada=sum(ada_result['score'])/len(ada_result)
print(M_ada)
#81.2 % test accuracy

#obtaining probabilites 
ada_probs = ada.predict_proba(x_test)

roc_y_test = pd.DataFrame(y_test)
roc_y_test[['Check-Out','Canceled','No-Show']]=pd.get_dummies(roc_y_test['reservation_status'])

fpr1, tpr1 = calcRates(roc_y_test['Check-Out'], ada_probs[:,0])
fpr2, tpr2 = calcRates(roc_y_test['Canceled'], ada_probs[:,1])
fpr3, tpr3 = calcRates(roc_y_test['No-Show'], ada_probs[:,2])

plt.plot(fpr1,tpr1,label='Check-Out')
plt.plot(fpr2,tpr2,label='Canceled')
plt.plot(fpr3,tpr3,label='No-Show')
plt.plot([0, 1], [0, 1],'r--')
plt.legend(loc="lower right")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve for Adaboost')
plt.show()


################## SVM

#considering only random 1500 samples, 1000 train, 500 test 

svm_train, svm_test = train_test_split(hotel_upd, train_size=1000, test_size=500, random_state=42)

#using k fold cross validation to find best C to use
#creating folds
k=5
N=len(svm_train)
Min_index=0
folds=list()
for i in range(1,k+1):
    if (i==k):
        Max_index=(round(N/k))*i
        folds.append(range((Min_index+1),Max_index))
    else:
        Max_index=(round(N/k))*i
        if(Max_index>N):
            Max_index=N-1
        folds.append(range((Min_index+1),Max_index+1))
        Min_index=Max_index 
print(folds)



######### this entire loop takes ~45 minutes to run even with the small sample size
SVMlist = list()
cL= [0.01,0.1,0.5,1,2.5,5,10]
for c in cL:
    print(c)
    SVM = SVC(gamma='auto',probability=True, C = c,kernel="linear")
    
    K_list= list()
    for j in range(0,k): 
        K_cross=svm_train.iloc[folds[j]] 
        K_train=[] 
        for i in range(0,k):
            if (i!=j): 
                K_train.append(svm_train.iloc[folds[i]])     
        K_train=pd.concat(K_train)
        K_train=pd.DataFrame(K_train)
        
        x_svm_train=K_train.drop(["reservation_status"],axis=1)
        y_svm_train=K_train["reservation_status"]
        K_cross=pd.DataFrame(K_cross)
        x_svm_cross=K_cross.drop(["reservation_status"],axis=1)
        y_svm_cross=K_cross["reservation_status"]

        #fitting SVM
        SVM.fit(x_svm_train,y_svm_train)
    
        #predict on cross validation set
        SVM_pred = pd.DataFrame(SVM.predict(x_svm_cross))
            
        #store actual value of reservation_status in dataframe
        Y_act = (pd.DataFrame(y_svm_cross)).reset_index(drop=True)
            
        frames = [Y_act, SVM_pred]
        SVM_result = pd.concat(frames,axis=1,ignore_index=True)
            
        #applying accuracy score function and calculating accuracy
        SVM_result['score']=SVM_result.apply(accuracy_score,axis=1)
        M=float(sum(SVM_result['score']))/len(SVM_result)
        
        K_list.append(M)

    #mean accuracy for one value of n_estimator    
    SVM_mean=sum(K_list)/len(K_list)
    
    #appending mean accuracy in list
    SVMlist.append(SVM_mean)


#plot of eta vs mean accuracy
plt.scatter(cL,SVMlist)
plt.title('Hyperparameter Tuning using K Fold Cross Validation')
plt.xlabel('Hyperparameter C')
plt.ylabel('Cross Validation Set Accuracy')
plt.plot(cL, SVMlist, '-o')


#using value of C=5 since it has highest cross validation accuracy
x_svm_train = svm_train.drop(['reservation_status'], axis=1)
y_svm_train = svm_train['reservation_status']
x_svm_test = svm_test.drop(['reservation_status'], axis=1)
y_svm_test = svm_test['reservation_status']

SVM = SVC(gamma='auto',probability=True, C = 5,kernel="linear")
SVM.fit(x_svm_train,y_svm_train)

#predicting on test set
SVM_pred = pd.DataFrame(SVM.predict(x_svm_test))

SVM_probs = SVM.predict_proba(x_svm_test)
roc_y_svm_test = pd.DataFrame(y_svm_test)
roc_y_svm_test[['Check-Out','Canceled','No-Show']]=pd.get_dummies(roc_y_svm_test['reservation_status'])
SVM_pred = pd.DataFrame(SVM.predict(x_svm_test))

#store actual value of reservation_status in dataframe
Y_act = (pd.DataFrame(y_svm_test)).reset_index(drop=True)
        
frames = [Y_act, SVM_pred]
SVM_result = pd.concat(frames,axis=1,ignore_index=True)
        
#applying accuracy score function and calculating accuracy
SVM_result['score']=SVM_result.apply(accuracy_score,axis=1)
M=float(sum(SVM_result['score']))/len(SVM_result)
print(M)
#79% test accuracy

fpr1, tpr1 = calcRates(roc_y_svm_test['Check-Out'], SVM_probs[:,0])
fpr2, tpr2 = calcRates(roc_y_svm_test['Canceled'], SVM_probs[:,1])
fpr3, tpr3 = calcRates(roc_y_svm_test['No-Show'], SVM_probs[:,2])
plt.plot(fpr1,tpr1,label='Check-Out')
plt.plot(fpr2,tpr2,label='Canceled')
plt.plot(fpr3,tpr3,label='No-Show')
plt.plot([0, 1], [0, 1],'r--')
plt.legend(loc="lower right")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC curve for SVM with hyperparameter C=5')
plt.show()


##################### KNN 
## using k=5 folds
k=5
N=len(hotel_train)
Min_index=0
folds=list()
for i in range(1,k+1):
    if (i==k+1):
        Max_index=(round(N/k))*i+(N % k) 
    
    else:
        Max_index=(round(N/k))*i
        if (Max_index>N):
            Max_index=N-1
    folds.append(range((Min_index+1),Max_index+1))
    Min_index=Max_index 
print(folds)
## score collects all accuracy score values(5 values for 5 folds) at k fold
## Acc takes mean of five accuracy values for each k
Acc=[] 
score=[]
for p in range (2,10):
    for j in range(0,k): 
        test=hotel_train.iloc[folds[j]] 
        train=[] 
        for i in range(0,k):
            if (i!=j): 
                train.append(hotel_train.iloc[folds[i]])     
        train=pd.concat(train)
        train=pd.DataFrame(train)
        x_train_kfold=train.drop(["reservation_status"],axis=1)
        y_train_kfold=train["reservation_status"]
        test=pd.DataFrame(test)
        x_test_kfold=test.drop(["reservation_status"],axis=1)
        y_test_kfold=test["reservation_status"]
        metric="euclidean"
        knn=skln.KNeighborsClassifier(n_neighbors=p,metric=metric,algorithm='ball_tree')
        knn.fit(x_train_kfold,y_train_kfold)
        
        #predict on cross validation set
        yp=pd.DataFrame(knn.predict(x_test_kfold))
        
        #store actual value of reservation_status in dataframe
        y_test_kfold2 = (pd.DataFrame(y_test_kfold)).reset_index(drop=True)
        
        #append both (do not change order)
        frames = [y_test_kfold2, yp]
        knn_result = pd.concat(frames,axis=1,ignore_index=True)
        
        #applying accuracy score function and calculating accuracy
        knn_result['score']=knn_result.apply(accuracy_score,axis=1)
        M_knn=sum(knn_result['score'])/len(knn_result)
        
        score.append(M_knn)
    Acc.append(np.mean(score))
    score=[]
##plotting cross validation score vs k values   
k_range=[i for i in range(2,10)]
plt.figure(figsize=(16, 3))
plt.subplot(121)
# plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(k_range, Acc)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-validation accuracy')

### ROC curve for best hyperparameter i.e. k=5
#predicting on test set
knn=skln.KNeighborsClassifier(n_neighbors=5,metric=metric,algorithm='ball_tree')
knn.fit(x_train,y_train)
knn_pred = pd.DataFrame(knn.predict(x_test))

#store actual value of reservation_status in dataframe
y_test_knn = (pd.DataFrame(y_test)).reset_index(drop=True)
        
#append both (do not change order)
frames_knn = [y_test_knn, knn_pred]
knn_result = pd.concat(frames_knn,axis=1,ignore_index=True)

#applying accuracy score function and calculating accuracy
knn_result['score']=knn_result.apply(accuracy_score,axis=1)
M_knn=sum(knn_result['score'])/len(knn_result)
print(M_knn)


#obtaining probabilites 
knn_probs = knn.predict_proba(x_test)

roc_y_test = pd.DataFrame(y_test)
roc_y_test[['Check-Out','Canceled','No-Show']]=pd.get_dummies(roc_y_test['reservation_status'])

fpr1, tpr1 = calcRates(roc_y_test['Check-Out'], knn_probs[:,0])
fpr2, tpr2 = calcRates(roc_y_test['Canceled'], knn_probs[:,1])
fpr3, tpr3 = calcRates(roc_y_test['No-Show'], knn_probs[:,2])


plt.plot(fpr1,tpr1,label='Check-Out')
plt.plot(fpr2,tpr2,label='Canceled')
plt.plot(fpr3,tpr3,label='No-Show')
plt.plot([0, 1], [0, 1],'r--')
plt.legend(loc="lower right")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC curve for KNN with hyperparameter k=5')
plt.show()

##Please run below code at last as we have used some variable above with same name
################# Unsupervised Algorithms
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_samples as ss

hotelUNS = hotel 
hotelUNS_upd = hotelUNS[hotelUNS['adults']!=0]
hotelUNS_upd['reserved_eq_assigned_roomtype'] = hotelUNS_upd.apply(f, axis=1)
hotelUNS_upd['adr'] = hotelUNS_upd.apply(f2, axis=1)
hotelUNS_upd['CheckOut'] = hotelUNS_upd.apply(f3, axis=1)
hotelUNS_upd[['CityhotelUNS','ResorthotelUNS']]=pd.get_dummies(hotelUNS_upd['hotel'])
hotelUNS_upd[['meal_BB','meal_FB','meal_HB','meal_SC','meal_Undefined']]=pd.get_dummies(hotelUNS_upd['meal'])
hotelUNS_upd[['mar_Aviation','mar_Complementary','mar_Corporate','mar_Direct','mar_Groups','mar_Offline TA/TO','mar_Online TA','mar_Undefined']]=pd.get_dummies(hotelUNS_upd['market_segment'])
hotelUNS_upd[['dis_Corporate','dis_Direct','dis_GDS','dis_TA/TO','dis_Undefined']]=pd.get_dummies(hotelUNS_upd['distribution_channel'])
hotelUNS_upd[['No_Deposit','Non_Refund','Refundable']]=pd.get_dummies(hotelUNS_upd['deposit_type'])
hotelUNS_upd[['Contract','Group','Transient','Transient_Party']]=pd.get_dummies(hotelUNS_upd['customer_type'])

X= hotelUNS_upd.drop(['reservation_status'], axis=1)
y=hotelUNS_upd['reservation_status']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
x_train = x_train.drop(x_train.iloc[:,0:33],  axis=1)
x_test = x_test.drop(x_test.iloc[:,0:33],  axis=1)
## Can not plot all feature values on 2-D plot, so applying PCA to reduce dimensions of data
X = x_test.values.astype(np.float64)  
pca = PCA(n_components=2)
pcs = pca.fit_transform(X)
df_pcs = pd.DataFrame(data=pcs, columns=["PC1","PC2"])
r = np.array([df_pcs.PC1,df_pcs.PC2]).T
##kmeans with 4 clusters
kmeans = KMeans(n_clusters=4)
kmeans.fit(r)
y_kmeans = kmeans.predict(r)
plt.scatter(r[:, 0], r[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
##minibatch with 4 clusters
from sklearn.cluster import MiniBatchKMeans
MiniBatchKMeans = MiniBatchKMeans(n_clusters=4)
MiniBatchKMeans.fit(r)
y_MiniBatchKMeans = MiniBatchKMeans.predict(r)
plt.scatter(r[:, 0], r[:, 1], c=y_MiniBatchKMeans, s=50, cmap='viridis')

centers = MiniBatchKMeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
## minibatch with 5 clusters
from sklearn.cluster import MiniBatchKMeans
MiniBatchKMeans = MiniBatchKMeans(n_clusters=5)
MiniBatchKMeans.fit(r)
y_MiniBatchKMeans = MiniBatchKMeans.predict(r)
plt.scatter(r[:, 0], r[:, 1], c=y_MiniBatchKMeans, s=50, cmap='viridis')

centers = MiniBatchKMeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);



### please run entire batch of below code at once
## Plotting Elbow and Silhoutte plot
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(16, 6)


clusters = np.arange(2,11,1)
dists = []
means = []
stdevs = []

for l in clusters:
    km = KMeans(n_clusters=l)
    labels = km.fit_predict(r)
    
    ss_vals = ss(r, labels, metric="euclidean")
    ss_avg = np.mean(ss_vals)
    ss_std = np.std(ss_vals)
    
    dists.append(km.inertia_)
    means.append(ss_avg)
    stdevs.append(ss_std)
    
ax1.plot(clusters, dists, c="black", linestyle=":", marker="+", markersize=10)

ax1.set_xlabel("k")
ax1.set_ylabel("Distortion (Within-Cluster SSE)")
ax1.tick_params(axis='y', which='both', left=False, labelleft=False)
ax1.set_title("Elbow Method")

ax2.scatter(clusters, means, c="black")
ax2.errorbar(clusters, means, yerr=stdevs, fmt="o", c="black", alpha=0.25)    

ax2.set_title("Silhouette Plot")
ax2.set_ylim([0,1])
ax2.set_xlabel("k")
ax2.set_ylabel("Silhouette Coefficient")

plt.show()