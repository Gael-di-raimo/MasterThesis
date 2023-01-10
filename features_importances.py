import numpy as np
import os
import time
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

from utility.my_k_fold_cv import load_folds
from utility.utility_plot import plt_hist

from utility.metrics import getMethodComputationTimePerOrderFolds, getFeaturesComputationTimeFold

def getMethodComputationTimePerOrder(dt):
	# --------------------------------------------------------------------
	#  Return the time in 100 ns each sample of the dt will take
	# to solve the system if it take the best algorithm the second best 
	# and the worst.
	#
	#	return: timeBestAlgo, timeSecondBestAlgo, timeWorstAlgo
	# --------------------------------------------------------------------

	timeL = np.arange(11, 14)
	timeU = np.arange(14, 17)

	y = np.concatenate((dt[:, timeL], dt[:, timeU]))

	worstAlgo = np.argmax(y, axis=1)
	bestAlgo = np.argmin(y, axis=1)

	tmp_y = y.copy()
	tmp_y[np.arange(0, y.shape[0]),worstAlgo] = -1
	tmp_y[np.arange(0, y.shape[0]),bestAlgo] = -1

	secondBestAlgo = np.argmax(tmp_y, axis = 1)


	timeBestAlgo = y[np.arange(0, y.shape[0]), bestAlgo]
	timeSecondBestAlgo = y[np.arange(0, y.shape[0]), secondBestAlgo]
	timeWorstAlgo = y[np.arange(0, y.shape[0]), worstAlgo]

	return timeBestAlgo, timeSecondBestAlgo, timeWorstAlgo

def pltBar(bars, barNames, figName, color, xlabel, ylabel):
	if not os.path.isdir("plots/feature_selection"):
		os.mkdir("plots/feature_selection")

	tick_font_size = 16

	plt.clf()

	fig = plt.figure(figsize =(17, 7))
	plt.bar(barNames, bars, color = color, width = 0.6)

	plt.xlabel(xlabel, fontsize = tick_font_size + 2)
	plt.ylabel(ylabel, fontsize = tick_font_size + 2)
	
	plt.yticks(fontsize = tick_font_size)
	plt.xticks(fontsize = tick_font_size)

	plt.savefig("plots/feature_selection/"+figName+".png")

def feature_importance():

	dt_path = "datasets/mydt/folds"

	#load folds here

	x_indices_L = np.array([0,1,2,3,20,21,22,23])
	x_indices_U =  np.array([0,5,6,7,24,25,26,27])
	y_indices_L = 4
	y_indices_U = 8 

	folds, times_folds = load_folds(dt_path, x_indices_L, x_indices_U, y_indices_L, y_indices_U, return_times = True)


	x_list = []
	y_list = []
	time_list = []

	for i in range(len(folds)):

		if i < 7:
			fold = folds[i]
			x_list.append(fold[0])
			y_list.append(fold[1])
			time_list.append(times_folds[i])

	times = np.concatenate(time_list)
	feature = np.concatenate(x_list)
	nz_sol = np.concatenate(y_list)

	#loading the datasets with b

	reg =  DecisionTreeRegressor()
	reg.fit(feature, nz_sol)


	tick_font_size = 16
	barNames = ["solve-type", "$L_{n\\_row}$ or $U_{n\\_row}$", "$L_{nz}$ or $U_{nz}$","$b_{nz}$","$f_0$","$f_1$","$f_2$","$f_3$"]
	
	pltBar(reg.feature_importances_, barNames, "feature_importances_folds", 'b', "Feature name", "Feature importance")
	


	timeBestAlgo, timeSecondBestAlgo, timeWorstAlgo = getMethodComputationTimePerOrderFolds(times)

	timesFeatures = getFeaturesComputationTimeFold(times)
	
	diffTimeBestSecond = timeSecondBestAlgo - timeBestAlgo
	diffTimeBestWorst = timeWorstAlgo - timeBestAlgo



	diffTimeBestSecondReshape = np.column_stack((diffTimeBestSecond, diffTimeBestSecond, diffTimeBestSecond, diffTimeBestSecond))
	diffTimeBestWorstReshape = np.column_stack((diffTimeBestWorst, diffTimeBestWorst, diffTimeBestWorst, diffTimeBestWorst))
	

	propFeatureOverGainBestSecond = (np.mean(timesFeatures, axis = 0)/np.mean(diffTimeBestSecondReshape,axis = 0))
	propFeatureOverGainBestWorst = (np.mean(timesFeatures, axis = 0)/np.mean(diffTimeBestWorstReshape,axis = 0))


	print(propFeatureOverGainBestSecond)
	print("Percentage left (best second) = %f" % (1-np.sum(propFeatureOverGainBestSecond)))

	print(propFeatureOverGainBestWorst)
	print("Percentage left (best worst) = %f" % (1-np.sum(propFeatureOverGainBestWorst)))

	pltBar(propFeatureOverGainBestSecond, ["$f_0$","$f_1$","$f_2$","$f_3$"], "percentage_time_feature_over_gain_1to2", 'y', "Feature name", "Ratio")
	pltBar(propFeatureOverGainBestWorst, ["$f_0$","$f_1$","$f_2$","$f_3$"], "percentage_time_feature_over_gain_1to3", 'y', "Feature name", "Ratio")


	
	print("Percent of Feature computation time over possible gain (best to second) dt_train = %s "%(propFeatureOverGainBestSecond))

	#see the prop of time when the feature computation time is higher than the one of the time difference between the two fastest algorithm
	for i in range(4):

		mask = timesFeatures[:, i] > diffTimeBestSecondReshape[:,i]
		propFeatureOverGainBestSecond = np.mean(timesFeatures[mask,i])/(np.mean(diffTimeBestSecondReshape[mask,i]))*100

		print("Mean time feature %d when its computation time is greater than winnable time best second algo is %f ns" %(i,np.mean(timesFeatures[mask,i])/100))
		print("Mean time possible save time bestsecond  %f ns" %(np.mean(diffTimeBestSecondReshape[mask,i])/100))
		print("Computation time best = %f ns and second best= %f ns" %(np.mean(timeBestAlgo[mask])/100,np.mean(timeSecondBestAlgo[mask])/100))


		print("Percent of time feature %d over possible gain time in dt_train = %f "%(i,np.mean(timesFeatures[mask,i])/np.mean(diffTimeBestSecondReshape[mask,i])))
		print("Percent of samples the Feature %d is not worse it in dt_train = %f "%(i,timesFeatures[mask,i].shape[0]/timesFeatures.shape[0]*100))


if __name__ == "__main__":
	feature_importance()