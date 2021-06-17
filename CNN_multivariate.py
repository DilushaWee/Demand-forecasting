#Author: Dilusha Weeraddana
# multivariate cnn example (2 features)
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


features2use_load=[]
features2use_tem=[]
lag=48
target='y_48'



def create_array(list_1,list_2,length):
    array_new =list()
    for i in range(0,len(list_1)):
      array_row=list()

      for j in range(0,length):
        array_x = list()

        array_x.append(list_1[i,j])
        array_x.append(list_2[i,j])
        array_row.append(array_x)
      array_new.append(array_row)
    print(array(array_new))
    return array(array_new)



for lag in range(0,48):

    lag = str(lag + 1)
    features2use_tem.append('Temperature_' + lag)

    features2use_load.append('Total_consumption_' + lag)



df = pd.read_csv('training_electricity_half_hour_with_all_future_tem_con_6days_with_lr_new_peak.csv')

df_training_entity=df.loc[df['year']<2019]
df_testing_entity=df.loc[df['year']==2019]


# scaling
# features_all=features2use_tem+features2use_load
# df_training_entity_X=df_training_entity[features_all]
# df_testing_entity_X=df_testing_entity[features_all]
# scaler= StandardScaler()
# scaler.fit_transform(df_training_entity_X)
# df_training_entity_X=scaler

n_features = 2

X_load=(df_training_entity[features2use_load])
scaler_load = StandardScaler()
scaler_load.fit(X_load)
scaler_load.fit_transform(X_load)
X_load=X_load.as_matrix()
print(X_load,'X_load')

X_tem=(df_training_entity[features2use_tem])
scaler_tem = StandardScaler()
scaler_tem.fit(X_tem)
scaler_tem.fit_transform(X_tem)
X_tem=X_tem.as_matrix()

# print(X_tem)
# print(X_load)
y=df_training_entity[target]
n_steps = len(features2use_load)
X=create_array(X_load,X_tem,n_steps)

# define model

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))

model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=200, verbose=2)
# demonstrate prediction
# x_input = array([[80, 85], [90, 95], [100, 105]])
# x_input = x_input.reshape((1, n_steps, n_features))
# yhat = model.predict(x_input, verbose=0)


x_load=(df_testing_entity[features2use_load])
x_load=scaler_load.transform(x_load)
print(x_load,'x_load')
# x_load=x_load.as_matrix()
x_tem=(df_testing_entity[features2use_tem])
x_tem=scaler_tem.transform(x_tem)
# x_tem=x_tem.as_matrix()


x_input=create_array(x_load,x_tem,n_steps)
# x_input = array([[80, 85], [90, 95], [100, 105]])
# x_input = x_input.reshape((1, n_steps, n_features))
y_pred = model.predict(x_input, verbose=0)

df_testing_entity['pred'] = y_pred


df_testing_entity.to_csv('vic test for 2016_multivar_cnn.csv', index=False)

mae = np.mean(np.abs(df_testing_entity['pred'] - df_testing_entity[target]))
mpe = np.mean((np.abs(df_testing_entity['pred'] - df_testing_entity[target]) / df_testing_entity[target]))

r2 = r2_score(df_testing_entity['pred'], df_testing_entity[target])
print(mae, 'mae')
print(r2, 'r2')
print(mpe, 'mpe')