import numpy as np
import os
import time
import matplotlib.pyplot as plt
import random
import pickle

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score, mean_absolute_percentage_error, balanced_accuracy_score
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

from utility.metrics import median_absolute_percentage_error, time_metrics_folds, getFeaturesComputationTimeFold



# --------------------------------------------------------------------
# File containing the class LR3M which will have 3 linear regressions
#	to predict the time of the 3 algorithm general, 2phase, 1phase.
#
# --------------------------------------------------------------------

class RFAdaProxy:

	def __init__(self):

		self.methodNames = ["general","2phases","1phase"]
		self.colorsMethods = ['blue', 'red', 'green']
		
		self.regGeneral = LinearRegression()
		self.reg2Phase =  RandomForestRegressor(random_state = 165391, n_estimators = 2, max_depth = 20)
		self.reg1Phase = RandomForestRegressor(random_state = 165391, n_estimators = 2, max_depth = 20)

		base = DecisionTreeRegressor(random_state = 165391, max_depth = 20)
		self.regProxy = AdaBoostRegressor(random_state = 165391, base_estimator = base, n_estimators = 5)


		self.index_feature_proxy = np.arange(8)
		self.index_feature_2Phase = np.arange(9)
		self.index_feature_1Phase = np.arange(9)
		self.index_feature_general = np.array([1])

		self.index_y_proxy = 3
		self.index_y_general = 0
		self.index_y_2Phase = 1
		self.index_y_1Phase = 2

		self.name = "RFAdaProxy (b)"

	def fit(self, x, y):

		print("Training the proxy regressor ...")
		self.regProxy.fit(x[:, self.index_feature_proxy], y[:, self.index_y_proxy])
		x_proxy = self.regProxy.predict(x[:, self.index_feature_proxy])

		x_train = np.column_stack((x, x_proxy))

		print("Training the general time regressor ...")
		self.regGeneral.fit(x_train[:, self.index_feature_general], y[:, self.index_y_general])
		print("Training the two phase time regressor ...")
		self.reg2Phase.fit(x_train[:, self.index_feature_2Phase], y[:, self.index_y_2Phase])
		print("Training the one phase time regressor ...")
		self.reg1Phase.fit(x_train[:, self.index_feature_1Phase], y[:, self.index_y_1Phase])
	
	
	def predict(self, x):
		

		x_proxy = self.regProxy.predict(x[:,  self.index_feature_proxy])
		x_pred = np.column_stack((x, x_proxy))

		times = np.zeros((x.shape[0], 3))

		times[:,0] = self.regGeneral.predict(x_pred[:, self.index_feature_general])
		times[:,1] = self.reg2Phase.predict(x_pred[:, self.index_feature_2Phase])
		times[:,2] = self.reg1Phase.predict(x_pred[:, self.index_feature_1Phase])

		y = np.argmin(times, axis = 1)

		return y

	def saveModel(self, path):

		file = open(path+"/RFAdaProxy.sav", 'wb')
		pickle.dump(self, file)
		file.close()



	def loadModel(self, path):

		file = open(path+"/RFAdaProxy.sav", 'rb')
		model = pickle.load(file)
		file.close()

		self.regProxy = model.regProxy

		self.regGeneral = model.regGeneral
		self.reg2Phase =  model.reg2Phase
		self.reg1Phase = model.reg1Phase

