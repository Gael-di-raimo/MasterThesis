import numpy as np
import os
import sys
import time
import random

from scipy.io import mmread
from scipy.sparse import csc_matrix
from scipy.sparse import eye

import matplotlib.pylab as plt

from utility.trav_dataset import  trav_lulog

# -----------------------------------------------------------
#  Utility function for the instance_table which is used for
#	convertion of the sub dataset to number and vice versa.
# -----------------------------------------------------------

def is_table_made(table_path):
	return os.path.isfile(table_path)

def load_table(table_path):

	table_name_to_id = {}
	table_id_to_name = {}


	table_file = open(table_path,'r')

	table_lines = table_file.readlines()

	for line in table_lines:

		splitted_line = line.split()

		table_id_to_name[int(splitted_line[0])] = splitted_line[1]

		table_name_to_id[splitted_line[1]] = int(splitted_line[0])


	table_file.close()

	return table_name_to_id, table_id_to_name

def get_instance_id(name_instance, table_path):
	
	table_name_to_id, table_id_to_name = load_table(table_path)
	if name_instance in table_name_to_id.keys():
		return table_name_to_id[name_instance]
	else:
		print(name_instance)
		return None

def append_table_entry(log_path, args):
	names = args[0]
	print("inTable")
	print(names)
	names.append(log_path.split('/')[-1])

	args[0] = names

	return

def make_instance_table(dt_path):

	names = []

	args = []
	args.append(names)

	trav_lulog(dt_path, append_table_entry, args = args)

	names = args[0]

	table_path = "datasets/instance_table.txt"

	table_file = open(table_path,'w')

	i = 0
	names.sort()

	for name in names:
		table_file.write("%d %s\n" % (i, name))
		i = i + 1

	table_file.close()

if __name__ == "__main__":

	dt_path = 'generated_times_features'
	make_instance_table(dt_path)

