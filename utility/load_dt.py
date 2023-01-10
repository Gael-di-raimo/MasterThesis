import numpy as np
import os
import random


from sklearn.model_selection import train_test_split

# -----------------------------------------------------------
#  Function used to load dataset
# -----------------------------------------------------------

#Function that check if the test set method split is done
def dt_split_done(data_path):

	train_done = os.path.isfile(data_path+"/dt_train.csv")
	test_done = os.path.isfile(data_path+"/dt_test.csv")
	val_done = os.path.isfile(data_path+"/dt_val.csv")

	return train_done and test_done and val_done

#Function that split the dataset into LS, VS and TS

def split_dataset(data_path, percentage = 1):

	print("Splitting the dataset")

	if dt_split_done(data_path):

		print("Dataset splitted already saved, loading it ...")

		dt_train = np.loadtxt(data_path+"/dt_train.csv", delimiter = ",")
		dt_test = np.loadtxt(data_path+"/dt_test.csv", delimiter = ",")
		dt_val = np.loadtxt(data_path+"/dt_val.csv", delimiter = ",")
		
		
		return dt_train, dt_test, dt_val
	
	print("Dataset was not splitted, splitting it ...")

	if percentage != 1:
		sampled_index = random.sample(range(len(dt)), int(len(dt)*percentage))

		dt = dt[sampled_index,:]
		pred = pred[sampled_index]
		heuristic = heuristic[sampled_index]

	dt = np.loadtxt(data_path+"/dt.csv", delimiter = ",")

	dt_train, dt_tmp, index_train, index_tmp = train_test_split(dt, range(len(dt)), train_size = 0.5, random_state= 165391)

	dt_test, dt_val, index_test, index_val = train_test_split(dt_tmp, range(len(dt_tmp)), train_size = 0.5, random_state= 165391)
	
	np.savetxt(data_path+"/dt_train.csv", dt_train, delimiter=",", fmt = '%i')
	np.savetxt(data_path+"/dt_test.csv", dt_test, delimiter=",", fmt = '%i')
	np.savetxt(data_path+"/dt_val.csv", dt_val, delimiter=",", fmt = '%i')

	return dt_train, dt_test, dt_val 

#check if the dt with the feature selected for the model are already saved
def dt_model_split_done(data_path):

	train_done = os.path.isfile(data_path+"/x_train.csv") and os.path.isfile(data_path+"/y_train.csv")
	test_done =  os.path.isfile(data_path+"/x_test.csv") and os.path.isfile(data_path+"/y_test.csv")
	val_done =  os.path.isfile(data_path+"/x_val.csv") and os.path.isfile(data_path+"/y_val.csv")

	return train_done and test_done and val_done

#Load a file with only necessary feature in order to save time

def load_dt_model(data_path, model_name, feature_indices_L, feature_indices_U, pred_indices_L = np.arange(11,14), pred_indices_U = np.arange(14,17), percentage = 1):
	"""
	if pred_indices_L is None:
		pred_indices_L = np.array(range(11,14))

	if  pred_indices_U is None:
		pred_indices_U = np.array(np.range(14,17))
	"""
	if not os.path.isdir(data_path+"/"+model_name):
		os.mkdir(data_path+"/"+model_name)

	if dt_model_split_done(data_path+"/"+model_name):
		
		x_train = np.loadtxt(data_path+"/"+model_name+"/x_train.csv", delimiter = ",")
		x_test = np.loadtxt(data_path+"/"+model_name+"/x_test.csv", delimiter = ",")

		y_train = np.loadtxt(data_path+"/"+model_name+"/y_train.csv", delimiter = ",")
		y_test = np.loadtxt(data_path+"/"+model_name+"/y_test.csv", delimiter = ",")

	else:

		dt_train, dt_test, dt_val = split_dataset(data_path, percentage = percentage)

		x_train = np.concatenate((dt_train[:, feature_indices_L], dt_train[:, feature_indices_U])) # stack nz rhs of L and U 
		x_test = np.concatenate((dt_test[:, feature_indices_L], dt_test[:, feature_indices_U]))

		y_train = np.concatenate((dt_train[:, pred_indices_L], dt_train[:, pred_indices_U])) # stack time of L and U solve 
		y_test = np.concatenate((dt_test[:, pred_indices_L], dt_test[:, pred_indices_U]))
		
		np.savetxt(data_path+"/"+model_name+"/x_train.csv", x_train, delimiter=',', fmt = '%i')
		np.savetxt(data_path+"/"+model_name+"/x_test.csv", x_test, delimiter=',', fmt = '%i')
		np.savetxt(data_path+"/"+model_name+"/y_train.csv", y_train, delimiter=',', fmt = '%i')
		np.savetxt(data_path+"/"+model_name+"/y_test.csv", y_test, delimiter=',', fmt = '%i')

	return x_train, x_test, y_train, y_test


def load_dt_NNZXreg(dt_path):
	
	if not os.path.isdir(dt_path+"/NNZXReg"):
		os.mkdir(dt_path+"/NNZXReg")

	

	if not(os.path.isfile(dt_path+"/NNZXReg/dt_train_NNZXReg.csv") and 
		os.path.isfile(dt_path+"/NNZXReg/dt_test_NNZXReg.csv") and
		os.path.isfile(dt_path+"/NNZXReg/dt_val_NNZXReg.csv")):
		print(dt_path+"/dt_train.csv")
		dt = np.loadtxt(dt_path+"/dt_train.csv", delimiter=",")

		dt_train, dt_tmp, index_train, index_tmp = train_test_split(dt, range(len(dt)), train_size = 0.5, random_state= 165391)
		dt_test, dt_val, index_test, index_val = train_test_split(dt_tmp, range(len(dt_tmp)), train_size = 0.5, random_state= 165391)

		np.savetxt(dt_path+"/NNZXReg/dt_train_NNZXReg.csv", dt_train, delimiter=",", fmt = '%i')
		np.savetxt(dt_path+"/NNZXReg/dt_test_NNZXReg.csv", dt_test, delimiter=",", fmt = '%i')
		np.savetxt(dt_path+"/NNZXReg/dt_val_NNZXReg.csv", dt_val, delimiter=",", fmt = '%i')

	
	else:

		dt_train = np.loadtxt(dt_path+"/NNZXReg/dt_train_NNZXReg.csv", delimiter = ",")
		dt_test = np.loadtxt(dt_path+"/NNZXReg/dt_test_NNZXReg.csv", delimiter = ",")
		dt_val = np.loadtxt(dt_path+"/NNZXReg/dt_val_NNZXReg.csv", delimiter = ",")


	return dt_train, dt_test, dt_val

def load_dt_NNZXreg2(dt_path):
	
	if not os.path.isdir(dt_path+"/NNZXReg"):
		os.mkdir(dt_path+"/NNZXReg")

	

	if not(os.path.isfile(dt_path+"/NNZXReg/dt_train_NNZXReg.csv") and 
		os.path.isfile(dt_path+"/NNZXReg/dt_test_NNZXReg.csv") and
		os.path.isfile(dt_path+"/NNZXReg/dt_val_NNZXReg.csv")):
		
		dt = np.loadtxt(dt_path+"/dt_train.csv", delimiter=",")

		dt_train, dt_test, dt_val = make_new_split(dt)

		np.savetxt(dt_path+"/NNZXReg/dt_train_NNZXReg.csv", dt_train, delimiter=",", fmt = '%i')
		np.savetxt(dt_path+"/NNZXReg/dt_test_NNZXReg.csv", dt_test, delimiter=",", fmt = '%i')
		np.savetxt(dt_path+"/NNZXReg/dt_val_NNZXReg.csv", dt_val, delimiter=",", fmt = '%i')

	
	else:

		dt_train = np.loadtxt(dt_path+"/NNZXReg/dt_train_NNZXReg.csv", delimiter = ",")
		dt_test = np.loadtxt(dt_path+"/NNZXReg/dt_test_NNZXReg.csv", delimiter = ",")
		dt_val = np.loadtxt(dt_path+"/NNZXReg/dt_val_NNZXReg.csv", delimiter = ",")


	return dt_train, dt_test, dt_val


def divide_dt_by_instances_in_two(dt, percentage = 0.5):
	
	instance_array = dt[:, 17]

	instance_nb, counts = np.unique(instance_array, return_counts = True)
	index_selected = np.random.choice(range(len(instance_nb)), int(len(instance_nb)*percentage))

	mask = np.zeros(len(dt), dtype=bool)

	for i in range(len(index_selected)):

		tmp_mask = dt[:, 17] == instance_nb[index_selected[i]]
		mask = np.logical_or(mask, tmp_mask)

	dt1 = dt[mask, :]

	dt2 = dt[np.logical_not(mask), :]

	if len(dt1) >= len(dt2):
		return dt1, dt2
	else:
		return dt2, dt1

def test_instance_nb(dt1, dt2):

	instance_nb, counts = np.unique(dt1[:,17], return_counts = True)
	instance_nb_2, counts = np.unique(dt2[:,17], return_counts = True)

	for i in instance_nb:
		if i in instance_nb_2:
			print("Error found a nb in both dt")
			return False
	return True


def try_split(dt, returned_dt):

	dt_train, dt_test = divide_dt_by_instances_in_two(dt)

	tot_samples = len(dt_train) + len(dt_test)

	if len(dt_train)/tot_samples > 0.55:
		print("Proportion of samples not good LS is %f and TS+VS is %f" %(len(dt_train)/tot_samples, 1-len(dt_train)/tot_samples))
		return False


	dt_test, dt_val = divide_dt_by_instances_in_two(dt_test)

	if len(dt_test)/tot_samples < 0.25 and len(dt_test) /tot_samples > 0.32:
		print("Proportion of samples not good LS is %f and TS is %f VS is %f" %(len(dt_train)/tot_samples, len(dt_test)/tot_samples,len(dt_val)/tot_samples))
		return False

	returned_dt.append(dt_train)
	returned_dt.append(dt_test)
	returned_dt.append(dt_val)

	return True


def make_new_split(dt):
	returned_dt = []

	i = 0
	
	while try_split(dt, returned_dt) == False:
		i+=1
		print("Did not find a good split attempt %d" %(i))

	dt_train = returned_dt[0]
	dt_test = returned_dt[1]
	dt_val = returned_dt[2]

	return dt_train, dt_test, dt_val

def save_new_split(data_path):

	dt = np.loadtxt(data_path+"/dt.csv", delimiter=",")
	dt_train, dt_test, dt_val = make_new_split(dt)

	np.savetxt(data_path+"/new/dt_new_train.csv", dt_train, delimiter=",", fmt = '%i')
	np.savetxt(data_path+"/new/dt_new_test.csv", dt_test, delimiter=",", fmt = '%i')
	np.savetxt(data_path+"/new/dt_new_val.csv", dt_val, delimiter=",", fmt = '%i')
	

if __name__ == "__main__":

	save_new_split("E:/Cour Master INFO/TFE/code/TFE_Models/datasets/mydt")

