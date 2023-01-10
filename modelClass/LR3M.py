import numpy as np
import os
import time
import matplotlib.pyplot as plt
import random
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_percentage_error, balanced_accuracy_score
from sklearn.base import clone
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor




# --------------------------------------------------------------------
# File containing the class LR3M which will have 3 linear regressions
#	to predict the time of the 3 algorithm general, 2phase, 1phase.
#
# --------------------------------------------------------------------

class LR3M:

	def __init__(self, modelName, featureNames, fctOnFeatures = False):

		self.methodNames = ["general","2phases","1phase"]
		self.colorsMethods = ['blue', 'red', 'green']
		self.featureNames = featureNames

		self.fctOnFeatures = fctOnFeatures
		
		self.regGeneral = LinearRegression()
		self.reg2Phase = LinearRegression()
		self.reg1Phase = LinearRegression()

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

	def fit(self, x, y):

		if self.fctOnFeatures:
			print("Fct must be apply on feature, fit each regressors")
			return 
		else:

			x_train_general = x[:,1]

			if len(x_train_general.shape) == 1:
				x_train_general = np.expand_dims(x_train_general, 1)

			x_train = x[:, 3]
			if len(x_train.shape) == 1:
				x_train = np.expand_dims(x_train, 1)

			self.regGeneral.fit(x_train_general, y[:,0])
			self.reg2Phase.fit(x_train, y[:,1])
			self.reg1Phase.fit(x_train, y[:,2])

	def predict(self, x):
		x_general = x[:,1]

		if len(x_general.shape) == 1:
			x_general = np.expand_dims(x_general, 1)

		x = x[:, 3]
		
		if len(x.shape) == 1:
			x = np.expand_dims(x, 1)

		times = np.zeros((x.shape[0], 3))

		times[:,0] = self.regGeneral.predict(x_general)
		times[:,1] = self.reg2Phase.predict(x)
		times[:,2] = self.reg1Phase.predict(x)

		return np.argmin(times, axis = 1)

	def saveModel(self, path):

		file = open(path+"/LRNNZBb.sav", 'wb')
		pickle.dump(self, file)
		file.close()



	def loadModel(self, path):

		file = open(path+"/LRNNZBb.sav", 'rb')
		model = pickle.load(file)
		file.close()


		self.regGeneral = model.regGeneral
		self.reg2Phase =  model.reg2Phase
		self.reg1Phase = model.reg1Phase


	def predictGeneral(self, x):
		return self.regGeneral.predict(x)

	def predict2Phase(self, x):
		return self.reg2Phase.predict(x)

	def predict1Phase(self, x):
		return self.reg1Phase.predict(x)
	


	def getR2scoreGeneral(self, xTest, yTest):
		return r2_score(np.float64(yTest), np.float64(self.regGeneral.predict(xTest)))

	def getR2score2Phase(self, xTest, yTest):
		return r2_score(np.float64(yTest), np.float64(self.reg2Phase.predict(xTest)))

	def getR2score1Phase(self, xTest, yTest):
		return r2_score(np.float64(yTest), np.float64(self.reg1Phase.predict(xTest)))



	def predictAlgo(self, xGeneral, x2Phase, x1Phase):

		predsTimes = np.zeros((xGeneral.shape[0], 3))

		predsTimes[:,0] = self.regGeneral.predict(xGeneral)
		predsTimes[:,1] = self.reg2Phase.predict(x2Phase)
		predsTimes[:,2] = self.reg1Phase.predict(x1Phase)

		predAlgo = np.argmin(predsTimes, axis=1)

		return predAlgo


	def pred_time_cv(self, folds, indices_used_per_method, modelName = None, fct_to_apply = None):
		
		modelName = self.name

		modelGeneral = self.regGeneral
		model2Phase = self.reg2Phase
		model1Phase = self.reg1Phase

		r2_train_folds = np.zeros((len(folds), 3))
		r2_test_folds = np.zeros((len(folds), 3))

		balanced_accuracies = np.zeros(len(folds))

		fold_model_choice_time = np.zeros(len(folds))
		fold_best_choice_time = np.zeros(len(folds))
		fold_average_choice_time = np.zeros(len(folds))
		fold_choose_1phase_time = np.zeros(len(folds))
		fold_pred_time = np.zeros(len(folds))


		fold_weight_train = np.zeros(len(folds))
		fold_weight_test = np.zeros(len(folds))

		fold_pred_algo = []
		fold_best_algo = []


		#compute R^2 and balanced accuracy for each fold  
		for i in range(len(folds)):


			print("Fold %d"%(i))
			modelCloneGeneral = clone(modelGeneral)
			modelClone2Phase = clone(model2Phase)
			modelClone1Phase = clone(model1Phase)


			fold_test = folds[i]

			x_list_general = []
			x_list_2phase = []
			x_list_1phase = []
			y_list = []

			#fuse the folds for training
			for j in range(len(folds)):
				if j != i:
					fold = folds[j]
					x = fold[0]

					if fct_to_apply is None:
						x_list_general.append(x[:, indices_used_per_method[0]])
						x_list_2phase.append(x[:, indices_used_per_method[1]])
						x_list_1phase.append(x[:, indices_used_per_method[2]])
					else:
						x_list_general.append(fct_to_apply[0](x[:, indices_used_per_method[0]]))
						x_list_2phase.append(fct_to_apply[1](x[:, indices_used_per_method[1]]))
						x_list_1phase.append(fct_to_apply[2](x[:, indices_used_per_method[2]]))

					y_list.append(fold[1])

			x_train_general = np.concatenate(x_list_general)
			x_train_2phase = np.concatenate(x_list_2phase)
			x_train_1phase = np.concatenate(x_list_1phase)
			
			if len(x_train_general.shape) == 1:
				x_train_general = np.expand_dims(x_train_general, 1)

			if len(x_train_2phase.shape) == 1:
				x_train_2phase = np.expand_dims(x_train_2phase, 1)

			if len(x_train_1phase.shape) == 1:
				x_train_1phase = np.expand_dims(x_train_1phase, 1)

			y_train = np.concatenate(y_list)
			
			y_test = fold_test[1]
			x_test = fold_test[0]

			#apply a fct to the feature if specified
			if fct_to_apply is None:
				x_test_general  = x_test[:, indices_used_per_method[0]]
				x_test_2phase  = x_test[:, indices_used_per_method[1]]
				x_test_1phase  = x_test[:, indices_used_per_method[2]]
			else:
				x_test_general  = fct_to_apply[0](x_test[:, indices_used_per_method[0]])
				x_test_2phase  = fct_to_apply[1](x_test[:, indices_used_per_method[1]])
				x_test_1phase  = fct_to_apply[2](x_test[:, indices_used_per_method[2]])

			if len(x_test_general.shape) == 1:
				x_test_general = np.expand_dims(x_test_general, 1)

			if len(x_test_2phase.shape) == 1:
				x_test_2phase = np.expand_dims(x_test_2phase, 1)

			if len(x_test_1phase.shape) == 1:
				x_test_1phase = np.expand_dims(x_test_1phase, 1)


			y_train_pred = np.zeros((y_train.shape[0], y_train.shape[1]))
			y_test_pred = np.zeros((y_test.shape[0], y_test.shape[1]))


			#fit the regressor that predict the algorithm times
			modelCloneGeneral.fit(x_train_general, y_train[:,0])
			modelClone2Phase.fit(x_train_2phase, y_train[:,1])
			modelClone1Phase.fit(x_train_1phase, y_train[:,2])


			y_train_pred[:,0] = modelCloneGeneral.predict(x_train_general)
			y_train_pred[:,1] = modelClone2Phase.predict(x_train_2phase)
			y_train_pred[:,2] = modelClone1Phase.predict(x_train_1phase)

			t0 = time.time_ns()
			y_test_pred[:,0] = modelCloneGeneral.predict(x_test_general)
			y_test_pred[:,1] = modelClone2Phase.predict(x_test_2phase)
			y_test_pred[:,2] = modelClone1Phase.predict(x_test_1phase)
			t1 = time.time_ns()

			t_pred = (t1-t0)/(10**9)

			fold_pred_time[i] = t_pred

			#compute the r2 scores
			r2_train_folds[i, 0] = r2_score(y_train[:,0], y_train_pred[:,0])
			r2_train_folds[i, 1] = r2_score(y_train[:,1], y_train_pred[:,1])
			r2_train_folds[i, 2] = r2_score(y_train[:,2], y_train_pred[:,2])

			r2_test_folds[i, 0] = r2_score(y_test[:,0], y_test_pred[:,0])
			r2_test_folds[i, 1] = r2_score(y_test[:,1], y_test_pred[:,1])
			r2_test_folds[i, 2] = r2_score(y_test[:,2], y_test_pred[:,2])


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
		
		model_choice_time = np.mean(fold_model_choice_time)
		average_choice_time = np.mean(fold_average_choice_time)
		best_choice_time = np.mean(fold_best_choice_time)
		onephase_choice_time = np.mean(fold_choose_1phase_time)
		average_pred_time = np.mean(fold_pred_time)

		if not(modelName is None):
			print("---------------%s---------------" % (modelName))
		else:
			print("-----------------------------------")

		print("With regression model %s"%(type(modelGeneral).__name__))
		print("R2 score train = %s" % (r2_train))
		print("R2 score test = %s"% (r2_test))

		print("Model choice time (test) = %f s" % (model_choice_time/(10**7)))
		print("Model choice time (test) + pred time = %f s" % (model_choice_time/(10**7) + average_pred_time))
		print("Best choice time = %f s"% (best_choice_time/(10**7)))
		print("Average choice time = %f s"% (average_choice_time/(10**7)))
		print("One phase choice time = %f s"% (onephase_choice_time/(10**7)))
		print("average balanced accuracy  = %f"% (np.mean(balanced_accuracies)))

		if not(modelName is None):
			print("---------------%s---------------" % (modelName))
		else:
			print("-----------------------------------")



		return fold_pred_algo, fold_best_algo