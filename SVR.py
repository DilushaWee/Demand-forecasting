#Author: Dilusha
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
# df=pd.read_csv('training_electricity_half_hour_with_all_future_tem_con.csv')
# df=pd.read_csv('./VIC_consumption_pv_weather/training_electricity_half_hour_with_all_future_tem_con_6days_with_lr_new_8_peak.csv')
df=pd.read_csv('./VIC_consumption_pv_weather/training_electricity_half_hour_with_all_future_tem_con_6days_with_lr_new_8_peak_classification.csv')


df=df.loc[df.y_1!=0]


lag=48
features2use=[]
target='y_48'

# def linear_regression(df_entity,features_lr):
#     linear_lr_list=[]
#
#     X=np.array(range(1,8)).reshape((-1, 1))
#     x=np.array(range(7,8)).reshape((-1, 1))
#     Y = df_entity[features_lr]
#
#     # y_tr=df_entity[target]
#
#     for row in range(0,len(df_entity)):
#
#         y_train=np.array(Y.iloc[row]).reshape(-1,1)
#         # print(y_train)
#         regr = linear_model.LinearRegression()
#         regr.fit(X, y_train)
#         y_pred = regr.predict(x)
#         # print(y_pred[0][0],y_tr[row])
#         linear_lr_list.append(y_pred[0][0])
#     return linear_lr_list



for lag in range(0,96):

  if lag > 47:
    lag = str(lag + 1)
    features2use.append('Temperature_' + lag)

  else:
   lag = str(lag + 1)
   features2use.append('Temperature_' + lag)
   features2use.append('Total_consumption_' + lag)

# 6 days lag
# features_lr=[]

count=5
while count >=0:
 lag_con=str((count+1)*48+1)
 features2use.append('Total_consumption_' + lag_con)
 # features_lr.append('Total_consumption_' + lag_con)
 count=count-1
# features_lr.append('Total_consumption_48')
features2use.append('Linear_regression_daily')
# linear_reg=linear_regression(df,features_lr)
# df['Linear_regression_daily']=linear_reg



col_names=list(df.columns)

col_names.remove('date')
col_names.remove('minute')
col_names.remove('year')
col_names.remove('hour')
col_names.remove('month')
col_names.remove('y_1')
col_names.remove('y_2')
col_names.remove('y_12')
col_names.remove('y_24')
col_names.remove('y_48')
col_names.remove('Peak_daily')
col_names.remove('Peak_daily_max')
col_names.remove('Peak_daily_max_1')
col_names.remove('Peak_daily_max_48')
col_names.remove('Peak_overall')
col_names.remove('Peak_overall_1')
col_names.remove('Peak_overall_48')
col_names.remove('Change_overall')

print(col_names)
features2use=col_names

# df=df[(df['month']==12) | (df['month']==1) | (df['month']==2)]
# df_training_entity=df.loc[df['year']<2019]
# df_testing_entity=df.loc[df['year']==2019]

df_training_entity=df.loc[df['year']<2018]
# df_training_entity=df_training_entity.head(100)
df_testing_entity=df.loc[df['year']==2018]


print(features2use)


# parameter tuning
C_range = 1. ** np.arange(-1,1)
gamma_range = 1. ** np.arange(-1,1)
nFolds = 5
nTrain_tuning = 1000
scaler=StandardScaler()
X=df_training_entity[features2use]
y=df_training_entity[target]
#
X_train=X
y_train=y
# scaler.fit(X)
X=scaler.fit_transform(X)
X_train=scaler.fit_transform(X_train)
# scaler.fit_transform(X)
# X=X.as_matrix()
x=df_testing_entity[features2use]
x=scaler.transform(x)
# scaler.transform(x)
# x=x.as_matrix()

#
# grid = GridSearchCV(SVR(kernel='rbf'), cv=nFolds,
#                     param_grid={"C": C_range, "gamma": gamma_range})
# svr_model = SVR(kernel='rbf', C=10, gamma=10)

# if nTrain_tuning >= len(y_train):
#     nTrain_tuning = len(y_train)
# index = np.random.permutation(len(y_train))
# index = index[:nTrain_tuning]
# print(X_train,'train set')
# print(y_train,type(y_train),'test set')

# grid.fit(X_train[index, :], y_train[index])
C_n=95

# svr_model = SVR(kernel='rbf', C=grid.best_params_["C"], gamma=grid.best_params_["gamma"])
svr_model = SVR(kernel= 'rbf', C= C_n,gamma=0.01)
    # svr_model = SVR(kernel= 'poly', C= 1.0, degree= 2)

# svr_model = SVR(kernel='rbf', C=10, gamma=0.1, epsilon=.1)
y_pred = svr_model.fit(X,y).predict(x)


df_testing_entity['pred']=y_pred


r2=r2_score(df_testing_entity[target],df_testing_entity['pred'])
mean_absolute_error=mean_absolute_error(df_testing_entity[target],df_testing_entity['pred'])
mape=np.mean(np.abs((df_testing_entity['pred']-df_testing_entity[target]) / df_testing_entity[target]))
print(C_n,' MAPE:', mape,'R^2:', r2, 'MAE:', mean_absolute_error)

df_testing_entity.to_csv('AEMO-VIC-2018-SVR_48.csv',index=False)