import numpy as np


def median_absolute_percentage_error(y_true, y_pred):
	# --------------------------------------------------------------------
	#  This metric is inspired from the mean_absolute_error of sklearn
	#		https://github.com/scikit-learn/scikit-learn/blob/f3f51f9b6/sklearn/metrics/_regression.py#L296
	# --------------------------------------------------------------------

	
	epsilon = np.finfo(np.float64).eps
	mape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)

	return np.median(mape)



def time_metrics(dt, t_pred):
	# ---------------------------------------------------------------------
	# This function returns the time metrics given the dataset dt which 
	#	contains the time taken to compute each feature and t_pred
	#	which is the time in sec taken to make the predictions on all the 
	#	dataset.
	# ---------------------------------------------------------------------



	time_best_algo, time_second_best_algo, time_worst_algo = getMethodComputationTimePerOrder(dt)

	diff_time_best_second = time_second_best_algo - time_best_algo
	diff_time_best_worst = time_worst_algo - time_best_algo


	time_diff_best_second = np.sum(time_second_best_algo - time_best_algo)/(10**7)
	time_diff_best_worst = (np.sum(time_worst_algo) - np.sum(time_best_algo))/(10**7)


	time_features = getFeaturesComputationTime(dt)
	tot_time_features = np.sum(getFeaturesComputationTime(dt))

	beta_12 = (t_pred + tot_time_features)/time_diff_best_second
	beta_13 = (t_pred + tot_time_features)/time_diff_best_worst


	#prop of samples in the dataset that the time of computation is higher than the time difference between algorithm

	time_features = np.sum(time_features) # sum the time for all feature

	mask = time_features + t_pred/len(dt) > diff_time_best_second
	gamma_12 = np.count_nonzero(mask)/len(dt)

	mask = time_features + t_pred/len(dt) > time_diff_best_worst

	gamma_13 = np.count_nonzero(mask)/len(dt)


	return np.array([beta_12, beta_13, gamma_12, gamma_13])


def time_metrics_folds(time_folds, t_pred_folds):
	# ---------------------------------------------------------------------
	# This function returns the time metrics given the dataset dt which 
	#	contains the time taken to compute each feature and t_pred
	#	which is the time in sec taken to make the predictions on all the 
	#	dataset.
	# ---------------------------------------------------------------------


	metric_per_fold = np.zeros((len(time_folds), 4))

	for i in range(len(time_folds)):
		
		time_fold = time_folds[i]

		t_pred = t_pred_folds[i]/100#t_pred is in ns so put it to 100ns as every other times


		time_best_algo, time_second_best_algo, time_worst_algo = getMethodComputationTimePerOrderFolds(time_fold)

		diff_time_best_second = time_second_best_algo - time_best_algo
		diff_time_best_worst = time_worst_algo - time_best_algo


		time_diff_best_second = np.sum(time_second_best_algo - time_best_algo)
		time_diff_best_worst = np.sum(time_worst_algo) - np.sum(time_best_algo)

		time_features = getFeaturesComputationTimeFold(time_fold)
		tot_time_features = np.sum(time_features)

		beta_12 = (t_pred + tot_time_features)/time_diff_best_second
		beta_13 = (t_pred + tot_time_features)/time_diff_best_worst


		#prop of samples in the dataset that the time of computation is higher than the time difference between algorithm

		time_features = np.sum(time_features) # sum the time for all feature



		mask = time_features + t_pred/len(time_fold) > diff_time_best_second
		gamma_12 = np.count_nonzero(mask)/len(time_fold)

		mask = time_features + t_pred/len(time_fold) > time_diff_best_worst

		gamma_13 = np.count_nonzero(mask)/len(time_fold)

		metric_per_fold[i, :] = np.array([beta_12, beta_13, gamma_12, gamma_13])


	return np.mean(metric_per_fold, axis = 0)


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

def getMethodComputationTimePerOrderFolds(fold_time):
	# --------------------------------------------------------------------
	#  Return the time in 100 ns each sample of the dt will take
	# to solve the system if it take the best algorithm the second best 
	# and the worst.
	#
	#	return: timeBestAlgo, timeSecondBestAlgo, timeWorstAlgo
	# --------------------------------------------------------------------

	y = fold_time[:, np.arange(0, 3)]

	worstAlgo = np.argmax(y, axis=1)
	bestAlgo = np.argmin(y, axis=1)

	tmp_y = y.copy()

	#change the already selected time to -1 to after find the second
	#best algorithm
	tmp_y[np.arange(0, y.shape[0]), worstAlgo] = -1
	tmp_y[np.arange(0, y.shape[0]), bestAlgo] = -1

	secondBestAlgo = np.argmax(tmp_y, axis = 1)


	timeBestAlgo = y[np.arange(0, y.shape[0]), bestAlgo]
	timeSecondBestAlgo = y[np.arange(0, y.shape[0]), secondBestAlgo]
	timeWorstAlgo = y[np.arange(0, y.shape[0]), worstAlgo]

	return timeBestAlgo, timeSecondBestAlgo, timeWorstAlgo


def getFeaturesComputationTimeFold(time_fold):	
	

	times = time_fold[:, np.array([3,4,5,6])]
	
	return times

def getFeaturesComputationTime(dt, f_numbers =  np.array([0,1,2,3])):	
	
	if len(f_numbers) == 0:
		return 0


	indicesTimeL = f_numbers + 28
	indicesTimeU = f_numbers + 32

	times = np.concatenate((dt[:, indicesTimeL], dt[:, indicesTimeU]))/(10**7)
	
	return times