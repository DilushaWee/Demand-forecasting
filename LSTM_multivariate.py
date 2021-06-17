#Author: Dilusha Weeraddana
# multivariate lstm example (2 features)
from numpy import array
from numpy import hstack
from numpy import split
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.optimizers import SGD
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from math import sqrt

features2use_load=[]
features2use_tem=[]
lag=48
target='y_48'

def split_dataset(train,test):
	# split into standard weeks
    # days=np.round(len(data)/48)
    # last_year=365*48
    # train, test = data[1:-(last_year*48)], data[-17525:-48]
    # restructure into windows of weekly data
    train = array(split(train, len(train) / 48))
    test = array(split(test, len(test) / 48))
    return train, test

def to_supervised(train, n_input, n_out=12):
	# flatten data
	data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
	X, y = list(), list()
	in_start = 0
	# step over the entire history one time step at a time
	for _ in range(len(data)):
		# define the end of the input sequence
		in_end = in_start + n_input
		out_end = in_end + n_out
		# ensure we have enough data for this instance
		if out_end <= len(data):
			X.append(data[in_start:in_end, :])
			y.append(data[in_end:out_end, 0])
		# move along one time step
		in_start += 1
	# print(X,'xxxxxxxxxxxxxxxxxx')
	# print(y,'yyyyyyyyyyyyyyyyyy')
	return array(X), array(y)
def build_model(train, n_input):
	train_x, train_y = to_supervised(train, n_input)
	# print(train_x,'hhh')
	verbose, epochs, batch_size = 2, 120, 16
	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
	# print(n_timesteps, n_features, n_outputs, 'jjjj')
	# reshape output into [samples, timesteps, features]
	train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
	# define model
	model = Sequential()
	model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
	model.add(RepeatVector(n_outputs))
	model.add(LSTM(200, activation='relu', return_sequences=True))
	model.add(TimeDistributed(Dense(100, activation='relu')))
	model.add(TimeDistributed(Dense(1)))
	model.compile(loss='mean_squared_error', optimizer=SGD

	(lr=0.01, momentum=0.9,clipnorm=1.0))

	# model.compile(loss='mse', optimizer='adam')
	model.fit(train_x, train_y, epochs=epochs, verbose=verbose)
	return model


# define parameters
#

# fit network


def forecast(model, history, n_input):
	# flatten data
	data = array(history)
	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
	# retrieve last observations for input data
	input_x = data[-n_input:, :]
	#print(input_x)
	# reshape into [1, n_input, n]
	input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
	# forecast the next week
	yhat = model.predict(input_x, verbose=0)
	# we only want the vector forecast
	yhat = yhat[0]
	# print(yhat)
	return yhat
def evaluvate_overall(actual,predicted):
	mae = mean_absolute_error(actual[:, 5], predicted[:, 5])
	r2=r2_score(actual[:, 5], predicted[:, 5])
	print(mae,r2)
	dataset = pd.DataFrame({'actual1': list(actual[:, 0]), 'prediction1': list((predicted[:, 0]))})

	for num in range(1,12):
	  dataset_each = pd.DataFrame({'actual'+str(num+1): list(actual[:, num]), 'prediction'+str(num+1): list((predicted[:, num]))})
	  dataset=pd.concat([dataset,dataset_each],axis=1)
	# act=pd.DataFrame(actual)
	# predi=pd.DataFrame(list(predicted))
	# frames=[act,predi]
	# combined=pd.concat(frames,axis=1)
	# combined.to_csv('ghgs.csv')
	dataset.to_csv('AEMO_200_100_12Steps_200epchs_noFW_v2.csv',index=False)


def evaluate_forecasts(actual, predicted):
	scores = list()
	# calculate an RMSE score for each day
	for i in range(actual.shape[1]):
		# calculate mse
		mse = mean_squared_error(actual[:, i], predicted[:, i])
		# calculate rmse
		rmse = sqrt(mse)
		# store
		scores.append(rmse)
	# calculate overall RMSE
	s = 0
	for row in range(actual.shape[0]):
		for col in range(actual.shape[1]):
			s += (actual[row, col] - predicted[row, col])**2
	score = sqrt(s / (actual.shape[0] * actual.shape[1]))
	return score, scores
def evaluate_model(train, test, n_input):
	# fit model
	print(train)
	model = build_model(train, n_input)
	# history is a list of weekly data
	history = [x for x in train]
	# print(history)
	# walk-forward validation over each week
	predictions = list()
	for i in range(len(test)):
		# predict the week
		yhat_sequence = forecast(model, history, n_input)
		# print(yhat_sequence,'y_hat')
		# store the predictions
		predictions.append(yhat_sequence)
		# get real observation and add to history for predicting the next week
		history.append(test[i, :])
	# evaluate predictions days for each week
	predictions = array(predictions)
	evaluvate_overall(test[:, :, 0], predictions)
	# score, scores = evaluate_forecasts(test[:, :, 0], predictions)
	# return score, scores
def data_preproc(df):
	length=len(df)
	length_revised=int(((np.round(length/48))-1)*48)

	# print(length_revised)
	df=df.head(length_revised)
	return df
df = pd.read_csv('../vic_consumption_weather_pv_futureweather.csv',usecols=['opdemandMW','Air Temperature in degrees C','Year Month Day Hour Minutes in YYYY','Air Temperature in degrees C_nextDay'])
df_training_entity=df.loc[df['Year Month Day Hour Minutes in YYYY']<2018]
df_training_entity=df_training_entity.head(96)
df_training_entity=df_training_entity[['opdemandMW','Air Temperature in degrees C']]
df_testing_entity=df.loc[df['Year Month Day Hour Minutes in YYYY']==2018]
df_testing_entity=df_testing_entity.head(96)
# print(df_testing_entity)

df_testing_entity=df_testing_entity[['opdemandMW','Air Temperature in degrees C']]
df_training_entity=data_preproc(df_training_entity)
df_testing_entity=data_preproc(df_testing_entity)
train, test = split_dataset(df_training_entity.values,df_testing_entity.values)
# print(train)
# print(test)

n_input = 48
evaluate_model(train, test, n_input)


# X,x=scaling(X,x)
# X = X.reshape((X.shape[0], X.shape[1], n_features))
# x = x.reshape((x.shape[0], x.shape[1], n_features))


# X_load=X_load.as_matrix()
# x_load=x_load.as_matrix()
#
# X_tem=X_tem.as_matrix()
# x_tem=x_tem.as_matrix()



