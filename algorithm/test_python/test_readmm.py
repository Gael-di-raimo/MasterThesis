import os
import numpy as np

from scipy.io import mmread
from scipy.sparse import csc_matrix, coo_matrix
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve_triangular

from CMatrix import CMatrix

def print_array_21_decimals(array):

	print("[", end ='')
	for i in range(len(array)-1):
		print("%.21lf, " % array[i], end ='')

	print("%.21lf]" % array[len(array)-1],end ='')


"""

This function will load the CMatrix and will load the corresponding mtx file with scipy. 
These loaded matrix will be compared the function and return true if they are the same
and false otherwise.

"""

def compare_cond_matrix(cmatrix_path, matrix_path):

	cMatrix = CMatrix(cmatrix_path)
	coo_m = mmread(matrix_path)

	if cMatrix.mformat:
		matrix = coo_m.tocsc()
	else:
		matrix = coo_m.tocsr()

	if cMatrix.nRow == matrix.shape[0]:

		print("Row number does not corresponds.")
		print("Scipy nb rows: "+str(matrix.shape[0]))
		print("Mymtx nb rows: "+str(cMatrix.nRow))

	if cMatrix.nCol == matrix.shape[1]:

		print("Column number does not corresponds.")
		print("Scipy nb columns: "+str(matrix.shape[1]))
		print("Mymtx nb columns: "+str(cMatrix.nCol))

	if len(cMatrix.indptr) == 0:
		print(matrix.indptr)
		print(cMatrix.indptr)
	
	if len(cMatrix.indptr) == 0 and len(matrix.indptr) == 0 and len(cMatrix.indices) == 0 and len(matrix.indices) == 0:

		if len(cMatrix.data) == 0 and len(matrix.data) == 0:
			return True

	if not np.array_equal(matrix.indptr, cMatrix.indptr):
		print("Scipy inptr: "+str(matrix.indptr))
		print("Mymtx inptr: "+str(cMatrix.indptr))
		return False

	if not np.array_equal(matrix.indices, cMatrix.indices):
		print("Scipy indices: "+str(matrix.indices))
		print("Mymtx indices: "+str(cMatrix.indices))

		return False

	#checking error percentage of the data (it can occur from rounding of the data written in and out)
	if not np.less(np.divide(np.subtract(matrix.data, cMatrix.data), cMatrix.data + 0.00000000000000000001), 0.001).all():

		print("Scipy data: ", end ='')
		print_array_21_decimals(matrix.data)
		print("\n")

		print("Mymtx data: ", end ='')
		print_array_21_decimals(cMatrix.data)
		print("\n")

		return False


	return True


def test_cmatrix(mpath):

	#generate a c matrix

	if not os.path.isfile("test.exe"):
		os.system("gcc ../mainReadMM.c ../mtx.c ../headers/mtx.h -o test")

	#Solve type and triangular form are not used here 0 0 is ok
	os.system("test 0 0 \""+mpath+"\" \"e:/RhsPath\" \"out.txt\"")
	
	#compare the matrix of scipy with the one created
	res = compare_cond_matrix("out.txt", mpath)

	if res == False:
		print("Found a matrix that is different from the one of scipy at "+mpath+ " .")
		exit(1)



def trav_mtx(dataset_path, fct, args = None):


	toExplore = set()
	folders = os.listdir(dataset_path)

	for file in folders:
		path = dataset_path + "/" + file
		toExplore.add(path)

	while len(toExplore) != 0:

		toExplorePath = toExplore.pop()
		if toExplorePath[len(toExplorePath)-3:len(toExplorePath)] == "mtx":
			
			if args == None:
				fct(toExplorePath)
			else:
				fct(toExplorePath, args)
		else:
			if os.path.isdir(toExplorePath):
				folders = os.listdir(toExplorePath)

				for file in folders:
					toExplore.add(toExplorePath + "/" + file)


	return 0
	

if __name__ == "__main__":

	if os.path.isfile("test.exe"):
		os.system("del test.exe")

	trav_mtx("E:/Cour Master INFO/TFE/instance_samples", test_cmatrix)
	
