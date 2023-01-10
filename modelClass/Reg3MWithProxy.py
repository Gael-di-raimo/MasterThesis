import numpy as np
import os
import time
import matplotlib.pyplot as plt
import random
import pickle

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score, mean_absolute_percentage_error, balanced_accuracy_score
from sklearn.base import clone
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from utility.metrics import median_absolute_percentage_error, time_metrics_folds, getFeaturesComputationTimeFold



# -------------------------------------------------------------------------
# File containing the class R3MwProxy which will have 4 linear regressions,
#	the first one predicting the number of non-zeros elements in the
#	solution. This value is then used in the 3 regressions to predict
#	the time of the 3 algorithms: general, 2phase, 1phase.
# -------------------------------------------------------------------------

class R3MwProxy:

	def __init__(self, modelName, regGeneral = None, reg2Phase = None, reg1Phase = None, regNNZX = None):#, featureNames):

		self.methodNames = ["general","2phases","1phase"]
		self.colorsMethods = ['blue', 'red', 'green']

		#set general regressor if specified
		if regGeneral is None:
			self.regGeneral = LinearRegression()
		else:
			self.regGeneral = clone(regGeneral)


		#set general regressor if specified
		if reg2Phase is None:
			self.reg2Phase = LinearRegression()
		else:
			self.reg2Phase = clone(reg2Phase)


		#set general regressor if specified
		if reg1Phase is None:
			self.reg1Phase = LinearRegression()
		else:
			self.reg1Phase = clone(reg1Phase)

		#set general regressor if specified
		if regNNZX is None:
			self.regNNZX = LinearRegression()
		else:
			self.regNNZX = clone(regNNZX)

		self.name = modelName

		self.regNames = []

		for i in range(3):
			self.regNames.append("LinearRegression")

	def fitRegGeneral(self, xTrain, yTrain):
		self.regGeneral.fit(xTrain, yTrain)

	def fitReg2Phase(self, xTrain, yTrain):
		self.reg2Phase.fit(xTrain, yTrain)

	def fitReg1Phase(self, xTrain, yTrain):
		self.reg1Phase.fit(xTrain, yTrain)

	def fitRegNNZX(self, xTrain, yTrain):
		self.regNNZX.fit(xTrain, yTrain)


	def predictGeneral(self, x):
		return self.regGeneral.predict(x)

	def predict2Phase(self, x):
		return self.reg2Phase.predict(x)

	def predict1Phase(self, x):
		return self.reg1Phase.predict(x)

	def predictNNZX(self, x):
		return self.regNNZX.predict(x)


	def getR2scoreGeneral(self, xTest, yTest):
		return r2_score(np.float64(yTest), np.float64(self.regGeneral.predict(xTest)))

	def getR2score2Phase(self, xTest, yTest):
		return r2_score(np.float64(yTest), np.float64(self.reg2Phase.predict(xTest)))

	def getR2score1Phase(self, xTest, yTest):
		return r2_score(np.float64(yTest), np.float64(self.reg1Phase.predict(xTest)))

	def getR2scoreNNZX(self, xTest, yTest):
		return r2_score(np.float64(yTest), np.float64(self.regNNZX.predict(xTest)))



	def predictAlgo(self, xGeneral, x2Phase, x1Phase):

		predsTimes = np.zeros((xGeneral.shape[0], 3))

		predsTimes[:,0] = self.regGeneral.predict(xGeneral)
		predsTimes[:,1] = self.reg2Phase.predict(x2Phase)
		predsTimes[:,2] = self.reg1Phase.predict(x1Phase)

		predAlgo = np.argmin(predsTimes, axis=1)

		return predAlgo
	
	def loadProxyFitted(self, proxy_model_name, i):
		filename = "proxyTrainedPerFold/"+proxy_model_name+"/"+proxy_model_name+"_fold_"+str(i)+".sav"
		
		if not os.path.isfile(filename):
			return None

		file = open(filename, 'rb')
		model = pickle.load(file)
		file.close()

		return model

	def saveModel(self, model, proxy_model_name, i):
		path = "proxyTrainedPerFold"


		if not os.path.isdir(path):
			os.mkdir(path)

		if not os.path.isdir(path+"/"+proxy_model_name):
			os.mkdir(path+"/"+proxy_model_name)


		path = path+"/"+proxy_model_name
		filename = path+"/"+proxy_model_name+"_fold_"+str(i)+".sav"

		file = open(filename, 'wb')

		pickle.dump(model, file)
		
		file.close()

	def pred_time_cv(self, folds, times_folds, indices_used_per_method, modelName = None, fct_to_apply = None, proxy_model = None, proxy_model_name = None, test_mode = False):
		
		modelName = self.name

		modelGeneral = self.regGeneral
		model2Phase = self.reg2Phase
		model1Phase = self.reg1Phase


		if test_mode:
			len_metric = 1
		else:
			len_metric = len(folds)

		r2_train_folds = np.zeros((len_metric, 3))
		r2_test_folds = np.zeros((len_metric, 3))

		MAPE_train_folds = np.zeros((len_metric, 3))
		MAPE_test_folds = np.zeros((len_metric, 3))

		balanced_accuracies = np.zeros(len_metric)

		fold_model_choice_time = np.zeros(len_metric)
		fold_best_choice_time = np.zeros(len_metric)
		fold_average_choice_time = np.zeros(len_metric)
		fold_choose_1phase_time = np.zeros(len_metric)

		fold_pred_time_general = np.zeros(len_metric)
		fold_pred_time_2phase = np.zeros(len_metric)
		fold_pred_time_1phase = np.zeros(len_metric)

		fold_pred_time = np.zeros(len_metric)
		fold_pred_proxy_time = np.zeros(len_metric)
		fold_time_feature = np.zeros(len_metric)


		fold_weight_train = np.zeros(len_metric)
		fold_weight_test = np.zeros(len_metric)

		fold_pred_algo = []
		fold_best_algo = []


		#compute R^2 and balanced accuracy for each fold  
		for i in range(len_metric):

			print("Fold %d"%(i))
			
			modelCloneGeneral = clone(modelGeneral)
			modelClone2Phase = clone(model2Phase)
			modelClone1Phase = clone(model1Phase)

			if proxy_model is None:
				modelCloneRegXNNZ = clone(self.regNNZX)


			fold_test = folds[i]

			x_list = []
			x_list_proxy = []
			y_list = []


			#fuse the folds for training
			for j in range(len(folds)):
				if j != i:
					fold = folds[j]
					x = fold[0]

					x_list.append(x)

					x_list_proxy.append(x[:, indices_used_per_method[3]])

					y_list.append(fold[1])

			#fuse training folds
			x_train = np.concatenate(x_list)

			y_train = np.concatenate(y_list)
			y_train_proxy = y_train[:,3]
			y_train = y_train[:,np.array([0,1,2])]

			#read test fold
			y_test = fold_test[1]	
			y_test_proxy = y_test[:,3]
			y_test = y_test[:,np.array([0,1,2])]
			
			x_test = fold_test[0]
			x_test_proxy = x_test[:, indices_used_per_method[3]]
			x_train_proxy = x_train[:, indices_used_per_method[3]]


			#fit the proxy that predict the number of non-zero elements in x

			#load the proxy model if saved
			if proxy_model is None or test_mode:
				proxyRegFitted = modelCloneRegXNNZ.fit(x_train_proxy, y_train_proxy)
			else:
				proxyRegFitted = self.loadProxyFitted(proxy_model_name, i)
				
				if proxyRegFitted is None:
					print("ProxyModel was not saved")
					proxyRegFitted = clone(proxy_model)
					proxyRegFitted = proxyRegFitted.fit(x_train_proxy, y_train_proxy)
					self.saveModel(proxyRegFitted, proxy_model_name, i)
				else:
					print("ProxyModel was saved")

			proxy_train_pred = proxyRegFitted.predict(x_train_proxy)

			t0 = time.time_ns()
			proxy_test_pred = proxyRegFitted.predict(x_test_proxy)
			t1 = time.time_ns()
			
			t_feature = np.sum(getFeaturesComputationTimeFold(times_folds[i]))
			fold_time_feature[i] = t_feature

			t_pred_proxy = (t1-t0)

			#fit the regressor that predict the algorithm times
			y_train_pred = np.zeros((y_train.shape[0], y_train.shape[1]))
			y_test_pred = np.zeros((y_test.shape[0], y_test.shape[1]))
			

			#add the proxy to the features
			x_train = np.column_stack((x_train, proxy_train_pred))

			x_test = np.column_stack((x_test, proxy_test_pred))

			#apply a fct to the feature if specified

			if fct_to_apply is None:
				x_train_general  = x_train[:, indices_used_per_method[0]]
				x_train_2phase  = x_train[:, indices_used_per_method[1]]
				x_train_1phase  = x_train[:, indices_used_per_method[2]]

				x_test_general  = x_test[:, indices_used_per_method[0]]
				x_test_2phase  = x_test[:, indices_used_per_method[1]]
				x_test_1phase  = x_test[:, indices_used_per_method[2]]

			else:
				x_train_general  = fct_to_apply[0](x_train[:, indices_used_per_method[0]])
				x_train_2phase  = fct_to_apply[1](x_train[:, indices_used_per_method[1]])
				x_train_1phase  = fct_to_apply[2](x_train[:, indices_used_per_method[2]])

				x_test_general  = fct_to_apply[0](x_test[:, indices_used_per_method[0]])
				x_test_2phase  = fct_to_apply[1](x_test[:, indices_used_per_method[1]])
				x_test_1phase  = fct_to_apply[2](x_test[:, indices_used_per_method[2]])
				
			#correct dimensions

			if len(x_train_general.shape) == 1:
				x_train_general = np.expand_dims(x_train_general, 1)

			if len(x_train_2phase.shape) == 1:
				x_train_2phase = np.expand_dims(x_train_2phase, 1)

			if len(x_train_1phase.shape) == 1:
				x_train_1phase = np.expand_dims(x_train_1phase, 1)

			if len(x_test_general.shape) == 1:
				x_test_general = np.expand_dims(x_test_general, 1)

			if len(x_test_2phase.shape) == 1:
				x_test_2phase = np.expand_dims(x_test_2phase, 1)

			if len(x_test_1phase.shape) == 1:
				x_test_1phase = np.expand_dims(x_test_1phase, 1)

			modelCloneGeneral.fit(x_train_general, y_train[:,0])
			modelClone2Phase.fit(x_train_2phase, y_train[:,1])
			modelClone1Phase.fit(x_train_1phase, y_train[:,2])

			y_train_pred[:,0] = modelCloneGeneral.predict(x_train_general)
			y_train_pred[:,1] = modelClone2Phase.predict(x_train_2phase)
			y_train_pred[:,2] = modelClone1Phase.predict(x_train_1phase)

			t0 = time.time_ns()
			y_test_pred[:,0] = modelCloneGeneral.predict(x_test_general)
			t1 = time.time_ns()
			
			t_pred_general = (t1-t0)
			

			t0 = time.time_ns()
			y_test_pred[:,1] = modelClone2Phase.predict(x_test_2phase)
			t1 = time.time_ns()
			
			t_pred_2phase = (t1-t0)

			t0 = time.time_ns()
			y_test_pred[:,2] = modelClone1Phase.predict(x_test_1phase)
			t1 = time.time_ns()
			
			t_pred_1phase = (t1-t0)


			t_pred = t_pred_general + t_pred_2phase + t_pred_1phase


			fold_pred_proxy_time[i] = t_pred_proxy

			if modelName == "LRNNZBb":
				print("set time 0")
				fold_pred_proxy_time[i] = 0
				fold_time_feature[i] = 0
				t_feature = 0


			fold_pred_time[i] = t_pred


			fold_pred_time_general[i] = t_pred_proxy + t_pred_general + t_feature
			fold_pred_time_2phase[i] = t_pred_proxy + t_pred_2phase + t_feature
			fold_pred_time_1phase[i] = t_pred_proxy + t_pred_1phase + t_feature

			#compute the r2 scores and MAPE
			r2_train_folds[i, 0] = r2_score(y_train[:,0], y_train_pred[:,0])
			r2_train_folds[i, 1] = r2_score(y_train[:,1], y_train_pred[:,1])
			r2_train_folds[i, 2] = r2_score(y_train[:,2], y_train_pred[:,2])

			r2_test_folds[i, 0] = r2_score(y_test[:,0], y_test_pred[:,0])
			r2_test_folds[i, 1] = r2_score(y_test[:,1], y_test_pred[:,1])
			r2_test_folds[i, 2] = r2_score(y_test[:,2], y_test_pred[:,2])

			MAPE_train_folds[i, 0] = mean_absolute_percentage_error(y_train[:,0], y_train_pred[:,0])
			MAPE_train_folds[i, 1] = mean_absolute_percentage_error(y_train[:,1], y_train_pred[:,1])
			MAPE_train_folds[i, 2] = mean_absolute_percentage_error(y_train[:,2], y_train_pred[:,2])

			MAPE_test_folds[i, 0] = mean_absolute_percentage_error(y_test[:,0], y_test_pred[:,0])
			MAPE_test_folds[i, 1] = mean_absolute_percentage_error(y_test[:,1], y_test_pred[:,1])
			MAPE_test_folds[i, 2] = mean_absolute_percentage_error(y_test[:,2], y_test_pred[:,2])



			#predict the fastest algorithm and compute the balanced accuracy
			pred_algo =  np.argmin(y_test_pred, axis=1)


			random_algo = np.mean(y_test, axis=1)
			best_algo = np.argmin(y_test, axis=1)
			choose_1phase = np.ones(best_algo.shape)
			choose_1phase = np.multiply(choose_1phase,2).astype(int)
			

			fold_model_choice_time[i] = np.sum(y_test[np.arange(0, y_test.shape[0]), pred_algo])
			fold_average_choice_time[i] = np.sum(random_algo)
			fold_best_choice_time[i] =  np.sum(y_test[np.arange(0, y_test.shape[0]), best_algo])
			fold_choose_1phase_time[i] =  np.sum(y_test[np.arange(0, y_test.shape[0]), choose_1phase])

			fold_pred_algo.append(pred_algo)
			fold_best_algo.append(best_algo)
			balanced_accuracies[i] = balanced_accuracy_score(best_algo, pred_algo)

			fold_weight_train[i] = x_train_general.shape[0]
			fold_weight_test[i] = x_test_general.shape[0]


		fold_weight_train = fold_weight_train/np.sum(fold_weight_train)
		fold_weight_test = fold_weight_test/np.sum(fold_weight_test)

		


		r2_train = np.mean(r2_train_folds, axis = 0)
		r2_test = np.mean(r2_test_folds, axis = 0)

		MAPE_train = np.mean(MAPE_train_folds, axis = 0)
		MAPE_test = np.mean(MAPE_test_folds, axis = 0)
		
		
		model_choice_time = np.mean(fold_model_choice_time/10**7)
		average_choice_time = np.mean(fold_average_choice_time/10**7)
		best_choice_time = np.mean(fold_best_choice_time/10**7)
		onephase_choice_time = np.mean(fold_choose_1phase_time/10**7)


		average_pred_proxy_time = np.mean(fold_pred_proxy_time)/10**9
		average_algo_pred_time = np.mean(fold_pred_time)/10**9
		average_time_feature = np.mean(fold_time_feature)/10**9

		if test_mode == False:
			tot_time_metric = time_metrics_folds(times_folds, fold_pred_proxy_time + fold_pred_time)
			
			time_metric_general = time_metrics_folds(times_folds, fold_pred_time_general)
			time_metric_2phase = time_metrics_folds(times_folds, fold_pred_time_2phase)
			time_metric_1phase = time_metrics_folds(times_folds, fold_pred_time_1phase)


		if not(modelName is None) and proxy_model_name is None:
			print("---------------%s---------------" % (modelName))
		elif not(modelName is None)and not (proxy_model_name is None):
			print("---------------%s with %s---------------" % (modelName,proxy_model_name))

		else:
			print("-----------------------------------")

		print("With regression model for general %s"%(type(modelGeneral).__name__))
		print("With regression model for 2phase %s"%(type(model2Phase).__name__))
		print("With regression model for 1phase %s"%(type(model1Phase).__name__))

		print("R2 score train = %s" % (r2_train))
		print("R2 score test = %s"% (r2_test))

		print("Model proxy pred time (test) = %f s" % (average_pred_proxy_time))
		print("Model algo pred time (test) = %f s" % (average_algo_pred_time))
		print("Model feature time (test) = %f s" % (average_time_feature))
		print("Model choice time (test) = %f s" % (model_choice_time))
		print("Model choice total time (test) = %f s" % (model_choice_time + average_pred_proxy_time + average_algo_pred_time + average_time_feature))

		print("Best choice time = %f s"% (best_choice_time))
		print("Average choice time = %f s"% (average_choice_time))
		print("One phase choice time = %f s"% (onephase_choice_time))
		print("average balanced accuracy  = %f"% (np.mean(balanced_accuracies)))


		if not(modelName is None) and proxy_model_name is None:
			print("---------------%s---------------" % (modelName))
		elif not(modelName is None)and not (proxy_model_name is None):
			print("---------------%s with %s---------------" % (modelName,proxy_model_name))

		else:
			print("-----------------------------------")

		if test_mode:
			return [], [], [], [], []
		scores = [r2_test, r2_train, MAPE_test, MAPE_train, np.mean(balanced_accuracies)]
		average_times_folds = [average_pred_proxy_time, average_algo_pred_time, average_time_feature, model_choice_time + average_pred_proxy_time + average_algo_pred_time]
		
		times_metrics = [tot_time_metric, time_metric_general, time_metric_2phase, time_metric_1phase]


		return fold_pred_algo, fold_best_algo, scores, average_times_folds, times_metrics