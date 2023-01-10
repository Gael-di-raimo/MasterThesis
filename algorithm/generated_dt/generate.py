import os
import numpy as np
import time

from scipy.io import mmread, mmwrite
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix
from scipy.sparse import eye, identity
from scipy.sparse.linalg import spsolve_triangular, inv


def generate_L(size, nz_percentage):
	"""
    generate_L generate a matrix L with sepecified percentage of nz


    :param size: the size of L
    :param sum_nz_L: the desired percentage of nz
    :return: L
    """ 

	print("Generating a L with size %d with a nz percentage of %f" % (size, nz_percentage))

	Lnz_remaining = int(nz_percentage*size*size - size)

	if Lnz_remaining < 0:
		print("Ask to create a matrix with too low nz compared to the matrix. Just generating a identity matrix")
		return identity(size, format = 'coo')

	L_i = np.zeros(int(nz_percentage*size*size))
	L_j = np.zeros(int(nz_percentage*size*size))
	L_x = np.zeros(int(nz_percentage*size*size))
	
	L_i[0:size] = np.arange(0, size, 1) 
	L_j[0:size] = np.arange(0, size, 1)
	L_x[0:size] = np.ones(size)

	L_nz = size

	i = 0

	rng = np.random.default_rng()

	values = rng.integers(low= -100000, high = 100000, size = Lnz_remaining)
	values = values[values != 0]

	values = rng.choice(values, Lnz_remaining, replace = True)

	

	rand_cols = rng.triangular(0, 0, size-1, Lnz_remaining).astype(int)

	rand_cols, occurances = np.unique(rand_cols, return_counts=True)


	for j in range(len(rand_cols)):

		rand_rows = rng.choice(range(rand_cols[j]+1, size), occurances[j], replace = False)


		L_i[L_nz: L_nz + occurances[j]] = rand_rows
		L_j[L_nz: L_nz + occurances[j]] = np.ones((1, occurances[j]))*rand_cols[j]

		L_x[L_nz: L_nz + occurances[j]] = values[L_nz-size: L_nz-size + occurances[j]]

		L_nz = L_nz + occurances[j]
		Lnz_remaining = Lnz_remaining - occurances[j]


 
	L = coo_matrix((L_x, (L_i, L_j)), shape=(size, size))

	if L.nnz !=  nz_percentage*size*size:
		print("The number of non-zero %d added to L is not corresponding to its percentage %f with size %d should be %d" % (L.nnz , nz_percentage, size, nz_percentage*size*size ))

	return L

def generate_specific_L(x, sum_nz_L):
	"""
    generate_specific_L generate a matrix L with a sum of nz
    	corresponding to the nz of x equal to sum_nz_L

    :param x: a dense vector
    :param sum_nz_L: the desired sum of nz
    	corresponding to the nz ofx
    :return: L
    """ 

	indices = np.arange(x.shape[0])

	rng = np.random.default_rng()

	mask = x[:,0] != 0

	indices = indices[mask]
	
	print("Generating a L with shape %d, must add %d to L and x has %d nz" % (x.shape[0], sum_nz_L, len(indices)))

	if sum_nz_L > int(len(indices)*len(indices)/2):
		print("Ask to create a matrix with too much nz compared to the matrix.")
		sum_nz_L = int(len(indices)*len(indices)/2)


	Lnz_remaining = int(sum_nz_L - len(indices))

	
	if Lnz_remaining < 0:
		print("Error sum_nz_L < x.nz")
		exit(0)

	x_nnz = len(indices)

	size = x.shape[0]

	L_i = np.zeros(size + Lnz_remaining)
	L_j = np.zeros(size + Lnz_remaining)
	L_x = np.zeros(size + Lnz_remaining)
	
	L_i[0:size] = np.arange(0, size, 1) 
	L_j[0:size] = np.arange(0, size, 1)
	L_x[0:size] = np.ones(size)
	
	L_nz = size

	rand_cols = rng.triangular(0, 0, x_nnz-1, Lnz_remaining).astype(int)

	rand_cols, occurances = np.unique(rand_cols, return_counts=True)

	values = rng.integers(low= -100000, high = 100000, size = Lnz_remaining)
	values = values[values != 0]

	values = rng.choice(values, Lnz_remaining, replace = True)

	for j in range(len(rand_cols)):

		if occurances[j] > x_nnz - (rand_cols[j]+1):
			occurances[j] =  x_nnz - (rand_cols[j]+1)
			

		rand_rows = rng.choice(range(rand_cols[j]+1,  x_nnz), occurances[j], replace = False)


		L_i[L_nz: L_nz + occurances[j]] = indices[rand_rows]
		L_j[L_nz: L_nz + occurances[j]] = np.ones((1, occurances[j]))*indices[rand_cols[j]]

		L_x[L_nz: L_nz + occurances[j]] = values[L_nz-size: L_nz-size + occurances[j]]

		L_nz = L_nz + occurances[j]
		Lnz_remaining = Lnz_remaining - occurances[j]

	"""
	while Lnz_remaining > 0:

		for j in range(len(indices)-1):

			nb_val_to_add = rng.integers(low=0, high =  min(len(indices) - (j+1) + 1, Lnz_remaining + 1), size=1)[0]

			rand_indices = rng.choice(indices[ j + 1 : len(indices)], nb_val_to_add, replace=False)

			values = rng.integers(low= -10, high = 10, size = nb_val_to_add) + 0.1

			Lnz_remaining = Lnz_remaining - nb_val_to_add

			L_i[L_nz:L_nz + nb_val_to_add] = rand_indices
			L_j[L_nz:L_nz + nb_val_to_add] = np.ones((1, nb_val_to_add))*j
			L_x[L_nz:L_nz + nb_val_to_add] = values

			L_nz = L_nz + nb_val_to_add
	"""
	L = coo_matrix((L_x, (L_i, L_j)), shape=(size, size))

	if L.nnz - L.shape[0] + len(indices)  != sum_nz_L:
		print("The number of non-zero %d added to L is not equal to sum_nz_L %d" % (int(L.nnz - L.shape[0] + len(indices)), int(sum_nz_L)))

	return L

def generate_vec(nz_percentage, size):

	res= np.zeros((size,1))

	print("Generating a vector with size %d and a nz per of %f thus with a nz of %d" % (size, nz_percentage, int(nz_percentage*size)))
	
	indices = rng.choice(np.arange(0,size,1), int(nz_percentage*size), replace=False)
	values = rng.integers(low= -100000, high = 100000, size = int(nz_percentage*size))
	values = values[values != 0]

	values = rng.choice(values, int(nz_percentage*size), replace = True)


	np.put(res, indices, values)

	return res

def generate_vec_triangular(nz_percentage, size):

	res= np.zeros((size,1))

	print("Generating a vector triangular with size %d and a nz per of %f thus with a nz of %d" % (size, nz_percentage, int(nz_percentage*size)))
	
	#indices = rng.choice(np.arange(0,size,1), int(nz_percentage*size), replace=False)
	
	indices = rng.triangular(0, size-1, size-1,  int(nz_percentage*size) ).astype(int)
	indices, occurances = np.unique(indices, return_counts=True)

	while len(indices) != int(nz_percentage*size):

		new_indices = rng.triangular(0, size-1, size-1,  int(nz_percentage*size) -  len(indices) ).astype(int)
		indices, occurances = np.unique(np.concatenate([indices, new_indices]), return_counts=True)


	values = rng.integers(low= -100000, high = 100000, size = int(nz_percentage*size))
	values = values[values != 0]

	values = rng.choice(values, int(nz_percentage*size), replace = True)


	np.put(res, indices, values)

	return res



if __name__ == "__main__":

	test = False

	generate_dt1 = False
	generate_dt1_2 = False

	generate_dt2 = True
	generate_dt3 = False
	generate_dt4 = False

	if test:
		
		Lshape = 10

		rng = np.random.default_rng()
		x_percentage = 0.8
		x = generate_vec(x_percentage, Lshape)

		print("Start test")

		fname = "x"+str(0).zfill(8)+".mtx"
		mmwrite("data/test/"+fname, coo_matrix(x))

		sums_nz_L = np.arange(9, 16, 1)

		#generate the multiple L
		L_counter = 0

		for sum_nz_L in sums_nz_L:
			L = generate_specific_L(x, sum_nz_L)
			fname = "L"+str(L_counter).zfill(8)+".mtx"
			mmwrite("data/test/"+fname, coo_matrix(L))
			L_counter = L_counter + 1
		
		L  = generate_L(10, 0.2)
		fname = "LL"+str(0).zfill(8)+".mtx"
		mmwrite("data/test/"+fname, coo_matrix(L))



		x = generate_vec(x_percentage, Lshape)
		sum_nz_L = 3*int(x_percentage*Lshape)

		
		L = generate_specific_L(x, sum_nz_L)
		fname = "L"+str(10).zfill(8)+".mtx"
		mmwrite("data/test/"+fname, coo_matrix(L))


		b = L.multiply(x)
		fname = "b"+str(10).zfill(8)+".mtx"
		mmwrite("data/test/"+fname, coo_matrix(b))

		print(sum_nz_L)

		exit(0)


	L_mult_b_path = 'data/L_multiple_b'
	x_mult_L_path = 'data/x_multiple_L'
	L_mult_x_path = 'data/L_multiple_x'
	mult_L_mult_x_path = 'data/mutliple_L_multiple_x'
	mult_L_mult_b_path = 'data/mutliple_L_multiple_b'


	#generate the first dataset
	
	#some variables

	nb_b_samples = 10
	nb_x_samples = 10
	nb_L_samples = 100 #for 2nd dataset

	Lshape = 10000
	
	rng = np.random.default_rng()


	#generate the first dataset
	if generate_dt1:
		nz_percentage = 0.001
		L = generate_L(Lshape, nz_percentage)

		#save L
		fname = "L"+str(0).zfill(8)+".mtx"
		mmwrite(L_mult_b_path+"/"+fname, coo_matrix(L))

		#generates b
		b_counter = 0
		b_percentages = np.arange(0.0001,0.05,0.0005)

		nb_samples = rng.triangular(0, 0, len(b_percentages),  nb_b_samples*len(b_percentages) ).astype(int)
		
		indices, nb_samples = np.unique(nb_samples, return_counts=True)
		print(len(b_percentages))
		print(max(indices))
		
		i = 0
		for b_percentage in b_percentages:
			
			if i >= len(nb_samples):
				break

			for j in range(nb_samples[i]):

				b = generate_vec_triangular(b_percentage, Lshape)

				fname = "b"+str(b_counter).zfill(8)+".mtx"
				mmwrite(L_mult_b_path+"/"+fname, coo_matrix(b))

				b_counter = b_counter +1

			i +=1

	if generate_dt1_2:

		nz_percentage = 0.001
		L = generate_L(Lshape, nz_percentage)

		#save L
		fname = "L"+str(0).zfill(8)+".mtx"
		mmwrite(L_mult_x_path+"/"+fname, coo_matrix(L))

		#generates b
		x_counter = 0
		x_percentages = np.arange(0.001,0.3,0.004)
		
		nb_samples = rng.triangular(0, 0, len(x_percentages),  nb_x_samples*len(x_percentages) ).astype(int)
		i = 0
		
		for x_percentage in x_percentages:
			i += 1
			for j in range(nb_samples[i]):

				x = generate_vec_triangular(x_percentage, Lshape)

				fname = "x"+str(x_counter).zfill(8)+".mtx"
				mmwrite(L_mult_x_path+"/"+fname, coo_matrix(x))
		
				b = L*x
				
				fname = "b"+str(x_counter).zfill(8)+".mtx"
				mmwrite(L_mult_x_path+"/"+fname, coo_matrix(b))

				x_counter = x_counter +1

	#generate the second dataset
	if generate_dt2:
		#generate x

		x_percentage = 0.01
		x = generate_vec(x_percentage, Lshape)

		fname = "x"+str(0).zfill(8)+".mtx"
		mmwrite(x_mult_L_path+"/"+fname, coo_matrix(x))

		sums_nz_L = np.arange(1, 20, 0.1)*int(x_percentage*Lshape)

		print(sums_nz_L)
		#generate the multiple L
		L_counter = 0
		for sum_nz_L in sums_nz_L:

			L = generate_specific_L(x, sum_nz_L)
			fname = "L"+str(L_counter).zfill(8)+".mtx"
			mmwrite(x_mult_L_path+"/"+fname, coo_matrix(L))
			
			b = L.tocsr()*x

			fname = "b"+str(L_counter).zfill(8)+".mtx"
			mmwrite(x_mult_L_path+"/"+fname, coo_matrix(b))

			print("L"+str(L_counter).zfill(8) +" should have sum_nz_L = %d" %(int(sum_nz_L)))
			L_counter = L_counter + 1


	#generate the third dataset
	if generate_dt3:
		#generate the x

		x_percentages = np.arange(0.002,0.300,0.006)
		
		if(len(x_percentages) > 9999):
			print("Too much  x percentages to try")
			exit(1)

		x_counter = 0

		for x_percentage in x_percentages:

			x = generate_vec(x_percentage, Lshape)

			fname = "x"+str(x_counter).zfill(8)+".mtx"
			mmwrite(mult_L_mult_x_path+"/"+fname, coo_matrix(x))
			
			x_counter = x_counter + 1

			sums_nz_L = np.arange(2, 50, 10)*int(x_percentage*Lshape)

			#generate the  multiple L
			L_counter = 0

			for sum_nz_L in sums_nz_L:

				L = generate_specific_L(x, sum_nz_L)
				fname = "L"+str(x_counter).zfill(4)+str(L_counter).zfill(4)+".mtx"
				mmwrite(mult_L_mult_x_path+"/"+fname, coo_matrix(L))
					
				b = L.tocsr()*x

				fname = "b"+str(x_counter).zfill(4)+str(L_counter).zfill(4)+".mtx"
				mmwrite(mult_L_mult_x_path+"/"+fname, coo_matrix(b))

				L_counter = L_counter + 1


	if generate_dt4:


		nz_percentages = np.arange(0.001,0.0025,0.000125)
		b_percentages = np.arange(0.0001,0.05,0.005)
		print(nz_percentages)
		print(b_percentages)
		exit(0)
		L_counter = 0

		for k in range(len(nz_percentages)):
			
			nz_percentage =  nz_percentages[k]
			L = generate_L(Lshape, nz_percentage)

			#save L
			fname = "L"+str(L_counter).zfill(4)+str(0).zfill(4)+".mtx"
			mmwrite(mult_L_mult_b_path+"/"+fname, coo_matrix(L))

			#generates b
			b_counter = 0

			b_percentages = np.arange(0.0001,0.05,0.005)

			nb_samples = rng.triangular(0, 0, len(b_percentages),  nb_b_samples*len(b_percentages) ).astype(int)
			
			indices, nb_samples = np.unique(nb_samples, return_counts=True)
		
			
			i = 0
			for b_percentage in b_percentages:
				
				if i >= len(nb_samples):
					break

				for j in range(nb_samples[i]):

					b = generate_vec_triangular(b_percentage, Lshape)

					fname = "b"+str(L_counter).zfill(4)+str(b_counter).zfill(4)+".mtx"
					mmwrite(mult_L_mult_b_path+"/"+fname, coo_matrix(b))

					b_counter = b_counter +1

				i +=1
			L_counter += 1










	