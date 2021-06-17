#Author:Dilusha
import pandas as pd
import numpy as np
from model import pipeline_regression as regr
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn import metrics


model_pipelines = {}
model2save='training_file_rf'
# model_name='GradientBoostingRegressor'
model_name='RandomForest'

target='y_48'
def train(df_training_entity, model2save,model_name,features2use):
    y = df_training_entity[target]

    X = df_training_entity[features2use]
    # X = df_training_entity[['Gas consumption','temperature','month list','number of rooms','year list','z score current gas', 'z score next gas','z score temp']]
    # X = df_training_entity[['temperature','Gas consumption','number of people','number of rooms','distance to the sea','income']]


    ### setup the model to be used

    md_regr = regr(model_pipelines[model_name])

    ### training the  model
    print('Start training ...')
    # print('Start training by', model_name, '...')
    md_regr.train(X, y)
    print('Training for ', model_name, 'is completed.')

    ### saving the  model
    md_regr.save_model(model2save)
    print('Trained model is saved.')

def predict(df_testing_entity, model2read, model_name, features2use):

    # X = df_testing_entity[['Gas consumption','month list']]
    # X = df_testing_entity[['Gas consumption','temperature','month list','number of rooms','year list','z score current gas', 'z score next gas','clustering_label']]
    X = df_testing_entity[features2use]
    # X = df_testing_entity[['Gas consumption','temperature','month list','number of rooms','year list','z score current gas', 'z score next gas','z score temp']]
    # X = df_testing_entity[
    #     ['temperature', 'Gas consumption', 'number of people', 'number of rooms', 'distance to the sea',
    #      'income']]

    md_regr = regr(model_pipelines[model_name])
    md_regr.load_model(model2read)

    y_pred = md_regr.predict(X)
    # fi=md_regr.reg.feature_importances_
    # summ_tem=np.sum(fi[0:48])
    # summ_con=np.sum(fi[48:96])
    # print(summ_tem)
    # print(summ_con)

    df_testing_entity['pred'] = y_pred
    return df_testing_entity


df=pd.read_csv('training_electricity_half_hour_with_all_future_tem_con.csv')
# df=pd.read_csv('training_electricity_half_hour_with_all_tem_con.csv')
# df=pd.read_csv('training_electricity_half_hour_with_all_tem_con_with_time_dayahead.csv')
# df_training=df_training.head(size)
# df=df.loc[df.y_1!=0]
# df_testing=pd.read_csv('testing_electricity_half_hour_with_weather.csv')
# df_testing=df_testing.head(size)
# df_testing=df_testing.loc[df_testing.y_1!=0]
df_training_entity=df.loc[df['year']<2019]
df_testing_entity=df.loc[df['year']==2019]
# df_testing_entity=df_testing_entity.loc[df_testing_entity['month']==1]


lag=48
features2use=[]
for lag in range(0,96):

  if lag > 47:
    lag = str(lag + 1)
    features2use.append('Temperature_' + lag)

  else:
   lag = str(lag + 1)
   features2use.append('Temperature_' + lag)
   features2use.append('Total_consumption_' + lag)

print(features2use)
# features2use.append('temperature')
# features2use.append('dew_point')
# features2use.append('humidity')
# features2use.append('date')
# features2use.append('month')
# features2use.append('year')
max_features=len(features2use)
print(len(features2use))

model_pipelines['RandomForest'] = RandomForestRegressor(n_estimators=114, min_samples_leaf=10,
                                                            random_state=None, max_features=max_features, max_depth=15,n_jobs=-1)
model_pipelines['GradientBoostingRegressor'] = GradientBoostingRegressor(n_estimators=114,
                                                            random_state=None, max_features=max_features)
train(df_training_entity, model2save, model_name, features2use)


df_predicting_entity = predict(df_testing_entity, model2save, model_name, features2use)
# abs_error=np.abs(df_predicting_entity['pred']-df_predicting_entity['y_1'])
# sum_abs_error=np.sum(abs_error)
# mae=np.mean(abs_error)
r2=r2_score(df_predicting_entity['pred'], df_predicting_entity[target])
mean_absolute_error=mean_absolute_error(df_predicting_entity['pred'], df_predicting_entity[target])
mape=np.mean(np.abs((df_predicting_entity['pred']-df_predicting_entity[target]) / df_predicting_entity[target]))
print('MAPE:', mape,'R^2:', r2, 'MAE:', mean_absolute_error)
df_predicting_entity.to_csv('AEMO-VIC-2019-Jan_30_mints.csv',index=False)

# MAPE: 0.010514332444020618 R^2: 0.9956568799325005 MAE: 50.94647485652504
# MAPE: 0.010558027775980668 R^2: 0.995548427714558 MAE: 51.1968382359883

