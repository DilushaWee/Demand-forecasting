#Author:Dilusha Weeraddana
from sklearn.model_selection import GridSearchCV
import xgboost
import pandas as pd
import numpy as np
import numpy as np

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error

# parameters = {
#     'max_depth': range (2, 10, 1),
#     'learning_rate': [0.1, 0.01, 0.05],
#     'gamma': [0.01, 0.1, 0.3, 0.5, 1, 1.5, 2],
#     "min_child_weight": [1, 3, 5, 7]
#
# }

parameters = {
    'max_depth': range (2, 10, 1),
    'subsample': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1]
}
model = xgboost.XGBRegressor(colsample_bytree=0.4,
                             gamma=0.01,
                             # learning_rate=0.25,
                             # min_child_weight=1.5,
                             # n_estimators=100,
                             n_estimators=1000,
                             # max_depth=8,
                             reg_alpha=0.75,
                             reg_lambda=0.45,
                             learning_rate=0.05,
                             min_child_weight=7,


                             # subsample=0.6,
                             seed=42)

grid_search = GridSearchCV(
    estimator=model,
    param_grid=parameters,

    verbose=True
)

df=pd.read_csv('training_electricity_half_hour_with_all_future_tem_con_6days_with_lr_new_peak.csv')
df_testing_entity = df.loc[df['year'] == 2018]
df_training_entity = df.loc[df['year'] < 2018]
df_training_entity_peak = df_training_entity.loc[df_training_entity['Peak_overall_48'] ==1]
df_training_entity_normal = df_training_entity.loc[df_training_entity['Peak_overall_48'] ==0]
########
chosen_idx = np.random.choice(len(df_training_entity), replace=False, size=5000)
df_training_entity = df_training_entity.iloc[chosen_idx]
########

########
chosen_idx_peak = np.random.choice(len(df_training_entity_peak), replace=False, size=100)
df_training_entity_peak = df_training_entity_peak.iloc[chosen_idx_peak]
########

########
chosen_idx_normal = np.random.choice(len(df_training_entity_normal), replace=False, size=5000)
df_training_entity_normal = df_training_entity_normal.iloc[chosen_idx_normal]
########

# df_training_entity=df_training_entity.head(1000)
# print(len(df_training_entity))
# print(df_training_entity['month'])

lag=48
features2use=[]
target='y_48'


col_names=list(df.columns)

col_names.remove('date')
col_names.remove('minute')
col_names.remove('year')
col_names.remove('hour')
col_names.remove('month')
col_names.remove('y_1')
col_names.remove('y_2')
col_names.remove('y_6')
col_names.remove('y_12')
col_names.remove('y_24')
col_names.remove('y_48')
# col_names.remove('Peak_daily')
# col_names.remove('Peak_daily_max')
# col_names.remove('Peak_daily_max_1')
# col_names.remove('Peak_daily_max_48')
# col_names.remove('Peak_overall')
# col_names.remove('Peak_overall_1')
col_names.remove('Peak_daily_48')
col_names.remove('Peak_overall_48')
col_names.remove('Change_overall_48')
col_names.remove('Linear_regression_daily')
col_names.remove('HDD')
col_names.remove('CDD')

print(col_names)
features2use=col_names

X_all=df_training_entity[features2use]
Y_all=df_training_entity[target]
X_peak=df_training_entity_peak[features2use]
Y_peak=df_training_entity_peak[target]
X_normal=df_training_entity_normal[features2use]
Y_normal=df_training_entity_normal[target]



def grid(X,Y,name):
 grid_search.fit(X, Y)
 print(grid_search.best_estimator_)
 print(name)

grid(X_peak,Y_peak,'peak')
# grid(X_normal,Y_normal,'normal')
# grid(X_all,Y_all,'all')