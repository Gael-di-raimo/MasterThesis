import numpy as np
import os
import sys
import time
import random

from decimal import Decimal
from scipy.io import mmread
from scipy.sparse import csc_matrix
from scipy.sparse import eye

import matplotlib.pylab as plt

from utility.trav_dataset import  trav_lulog
from utility.utility_table import get_instance_id, make_instance_table
from sklearn.model_selection import KFold

# -----------------------------------------------------------
#  Creates mydt.csv a file with all necessary information. This
#	file is made from all lu_log.csv files of the dataset 
#	mtx-01. To get information concerning the column of
#	dt.csv see info_dt.txt in folder datasets.
# -----------------------------------------------------------




def divide_dt_by_instances_in_two(dt, percentage_train = 0.5, tol = 0.05, test_mode = False):
	
	wrong_tol = True

	print("Dividing dt")
	
	print(percentage_train)
	nb_try = 1

	instance_array = dt[:, 17]
	instance_nb, counts = np.unique(instance_array, return_counts = True)
	print(len(instance_nb))
	print(len(instance_nb)*percentage_train)

	selected = len(instance_nb)*percentage_train
	print(selected/len(instance_nb))
	

	while wrong_tol:

		
		index_selected = np.random.choice(range(len(instance_nb)), int(len(instance_nb)*percentage_train), replace = False)

		if abs(np.sum(counts[index_selected])/len(dt)-percentage_train) <= tol or test_mode:
			print(np.sum(counts[index_selected])/len(dt))
			wrong_tol = False
			
			mask = np.zeros(len(dt), dtype=bool)

			for i in range(len(index_selected)):

				tmp_mask = dt[:, 17] == instance_nb[index_selected[i]]

				mask = np.logical_or(mask, tmp_mask)

			print("Count = %d mask = %d" %(np.sum(counts[index_selected]),np.count_nonzero(mask)))
		if wrong_tol:
			print("Didnt find a division with respected tol at try %d percentage_train is %f" %(nb_try, np.sum(counts[index_selected])/len(dt)))

		nb_try +=1



	dt1 = dt[mask, :]

	dt2 = dt[np.logical_not(mask), :]

	print("Found a division with prop dt_train = %f and prop dt_test = %f" %(len(dt1)/len(dt), len(dt2)/len(dt)))

	return dt1, dt2

def make_k_fold(dt_path, k, percentage_train, tol):

	#check if folds already created
	if not os.path.isdir(dt_path+'/folds'):
		os.mkdir(dt_path+'/folds')

	dt = np.loadtxt(dt_path+"/dt.csv", delimiter=",")


	#Divide for the performance assessment
	dt_cv, dt_cv_test = divide_dt_by_instances_in_two(dt, percentage_train = percentage_train, tol = 0.01, test_mode = True)

	#save dt_test
	np.savetxt(dt_path+"/folds/dt_cv_test.csv", dt_cv_test, delimiter=",", fmt = '%i')

	#do the k fold with the cv set

	kf = KFold(n_splits=k)

	instance_nb = np.unique(dt_cv[:,17])


	fold_nb = 0


	for i, (train_index, test_index) in enumerate(kf.split(instance_nb)):


		mask_fold = np.zeros(len(dt_cv), dtype=bool)

		for j in test_index:
			tmp_mask = dt_cv[:, 17] == instance_nb[j]
			mask_fold = np.logical_or(mask_fold, tmp_mask)

		dt_fold = dt_cv[mask_fold,:]

		#save the fold in a file
		np.savetxt(dt_path+"/folds/dt_fold_"+str(i)+".csv", dt_fold, delimiter=",", fmt = '%i')


def append_dt(path, args):
	
	dt = args[0]
	max_samples_per_instance = args[1]

	print("Trying to read "+path+' log file')
	
	timers_file = open(path+'/timers.txt')
	features_file = open(path+'/features.txt')


	t_lines = timers_file.readlines()
	f_lines = features_file.readlines()

	nb_samples = len(t_lines)
	dt_tmp = np.zeros((min(max_samples_per_instance, nb_samples), 36), dtype = np.int64)
	
	index_samples = random.sample(range(0, nb_samples), min(max_samples_per_instance, nb_samples)) 
	
	name_instance = path.split('/')[-1]
	table_path = args[2]
	instance_id = get_instance_id(name_instance, table_path)

	index_tmp = 0

	for i in index_samples:
		
		index_line = i

		#read general solve data
		t_line = t_lines[index_line].split()
		f_line = f_lines[index_line].split()

		dt_tmp[index_tmp, 0] = np.int64(t_line[0])

		factorization_id = t_line[1]
		solve_id = t_line[2]

		clock_frequency = np.int64(t_line[3])
		clock_frequency = float(clock_frequency)/10000000# to have the time in 100ns

		#adding L info
		dt_tmp[index_tmp,1] = np.int64(f_line[4])
		dt_tmp[index_tmp,2] = np.int64(f_line[5])
		dt_tmp[index_tmp,3] = np.int64(f_line[6])
		dt_tmp[index_tmp,4] = np.int64(f_line[7])
		
		#adding U info
		dt_tmp[index_tmp,5] = np.int64(f_line[16])
		dt_tmp[index_tmp,6] = np.int64(f_line[17])
		dt_tmp[index_tmp,7] = np.int64(f_line[18])
		dt_tmp[index_tmp,8] = np.int64(f_line[19])

		
		dt_tmp[index_tmp,9] = np.int64(0) #no heuristic should try to see in the corresponding file in mtx-02
		dt_tmp[index_tmp,10] = np.int64(0)
		



		#adding the times to the data
		
		#L solve times
		dt_tmp[index_tmp,11] = np.int64(t_line[4])/clock_frequency
		dt_tmp[index_tmp,12] = np.int64(t_line[5])/clock_frequency
		dt_tmp[index_tmp,13] = np.int64(t_line[6])/clock_frequency

		#U solve times
		dt_tmp[index_tmp,14] = np.int64(t_line[7])/clock_frequency
		dt_tmp[index_tmp,15] = np.int64(t_line[8])/clock_frequency
		dt_tmp[index_tmp,16] = np.int64(t_line[9])/clock_frequency


		dt_tmp[index_tmp,17] = np.int64(instance_id)
		dt_tmp[index_tmp,18] = np.int64(factorization_id)
		dt_tmp[index_tmp,19] = np.int64(solve_id)

		#adding features for the prediction of x.nnz
		clock_frequency = np.int64(f_line[3])
		clock_frequency = float(clock_frequency)/10000000# to have the time in 100ns

		#L solve feature for x.nnz
		dt_tmp[index_tmp,20] = np.int64(f_line[8])
		dt_tmp[index_tmp,21] = np.int64(f_line[10])
		dt_tmp[index_tmp,22] = np.int64(f_line[12])

		if f_line[14] == 'inf':
			dt_tmp[index_tmp,23] = np.longdouble(np.power(2,1023))
		else:
			try:
				dt_tmp[index_tmp,23] = np.longdouble(f_line[14])
			except OverflowError as e:
				dt_tmp[index_tmp,23] = np.longdouble(np.power(2,1023))

		#U solve feature for x.nnz
		dt_tmp[index_tmp,24] = np.int64(f_line[20])
		dt_tmp[index_tmp,25] = np.int64(f_line[22])
		dt_tmp[index_tmp,26] = np.int64(f_line[24])

		if f_line[26] == 'inf':
			dt_tmp[index_tmp,27] = np.longdouble(np.power(2,i))
		else:
			try:
				dt_tmp[index_tmp,27] = np.longdouble(f_line[26])
			except OverflowError as e:
				dt_tmp[index_tmp,27] = np.longdouble(np.power(2,1023))
			


		#time L solve feature computation
		dt_tmp[index_tmp,28] = np.int64(f_line[9])/clock_frequency
		dt_tmp[index_tmp,29] = np.int64(f_line[11])/clock_frequency
		dt_tmp[index_tmp,30] = np.int64(f_line[13])/clock_frequency
		dt_tmp[index_tmp,31] = np.int64(f_line[15])/clock_frequency

		#time U solve feature computation
		dt_tmp[index_tmp,32] = np.int64(f_line[21])/clock_frequency
		dt_tmp[index_tmp,33] = np.int64(f_line[23])/clock_frequency
		dt_tmp[index_tmp,34] = np.int64(f_line[25])/clock_frequency
		dt_tmp[index_tmp,35] = np.int64(f_line[27])/clock_frequency



		index_tmp = index_tmp + 1

	if len(dt) == 0:

		args[0] = dt_tmp

	else:
		args[0] = np.vstack((args[0], dt_tmp))

	timers_file.close()
	features_file.close()
	
	return 0

def make_dataset_from_log(log_folder):
	
	dt = []

	args = []
	args.append(dt)
	args.append(5000)
	args.append('datasets/instance_table.txt')

	trav_lulog(log_folder, append_dt, args = args, log_file_name = "timers.txt")
	dt = args[0]

	list_fmt = []


	np.savetxt('datasets/mydt/dt.csv', dt, delimiter=',', fmt = '%i')

	

if __name__ == "__main__":

	solve_folder = "generated_times_features"
	log_folder = "algorithm/dt"
	make_instance_table(log_folder)
	print("Salut")
	make_dataset_from_log(solve_folder)

	make_k_fold("datasets/mydt", 10, 0.8, 0.05)
