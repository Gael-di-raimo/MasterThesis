import numpy as np
import os
from scipy.io import mmread, mmwrite
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse import eye

def load_prod_matrices(prod_matrices_path, data_name, factorization_id, solve_id):

	RLHDM_path = prod_matrices_path+"/"+data_name+"/R_L_H_D_M"+str(factorization_id)+".mtx"
	UC_path = prod_matrices_path+"/"+data_name+"/U_C"+str(factorization_id)+".mtx"
	H_path = prod_matrices_path+"/"+data_name+"/H"+str(factorization_id)+".mtx"

	#RLHDM = mmread(RLHDM_path)
	#UC = mmread(UC_path)
	H = mmread(H_path)

	return H.tocsc() #RLHDM.tocsc(), UC.tocsc(), H.tocsc()

def getAllMatrix(lu_log_path, solve_id, factorization_id, m_format = "csc"):

	m_names = ["b","y","x","R","L","H","e","D","M","U","C"]
	ret_matrices = []

	i = 0

	for m_name in m_names:

		m_id = str(solve_id).zfill(8)

		if i > 2:
			m_id =  str(factorization_id).zfill(8)

		if m_id != None:

			m_path = lu_log_path +"/"+m_name+m_id+".mtx"

			m_coo = mmread(m_path)

			#change the format to csr only for L/U, if m_format is "csr"
			if m_names in ["L","U"] and m_format == "csr":
				ret_matrices.append(m_coo.tocsr())
			else:
				ret_matrices.append(m_coo.tocsc())

		else:
			print("Failed to load matrix")
			ret_matrices.append(None)
		i+=1

	return ret_matrices

def getMatrix(dt_path, solve_id, factorization_id, m_name, m_format = "csc"):

	ret_matrices = []

	i = 0	

	m_id =  str(solve_id).zfill(8)

	if i > 2:
		m_id = str(factorization_id).zfill(8)

	if m_id != None:

		m_path = dt_path +"/"+m_name+m_id+".mtx"

		m_coo = mmread(m_path)

		#change the format to csr only for L/U, if m_format is "csr"
		if m_name in ["L","U"] and m_format == "csr":
			ret_matrices.append(m_coo.tocsr())
		else:
			ret_matrices.append(m_coo.tocsc())

	else:
		print("Failed to load matrix")
		ret_matrices.append(None)
	i+=1

	return ret_matrices[0]

def make_H(eta_r, eta_i):

	n = R.shape[0]
	H = eye(n, format = "csr")

	eta_n = eta_i.shape[0]


	for k in range(eta_n):

		eta = eye(n, format = "csr")
		i = eta_i[k, 0] - 1
		eta[i, :] = eta_r[k, :]
		eta[i, i] = 1

		H = H * eta;
	return H.tocsc()
