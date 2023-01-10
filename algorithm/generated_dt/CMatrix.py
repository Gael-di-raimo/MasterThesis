import numpy as np

class CMatrix:

	def __init__(self, cMatrixPath):

		fileCMatrix = open(cMatrixPath, 'r')
		lines = fileCMatrix.readlines()

		self.mformat = int(lines[0])

		strSizeInfo = lines[1].split()

		self.nRow = int(strSizeInfo[0])
		self.nCol = int(strSizeInfo[1])
		self.nnz = int(strSizeInfo[2])

		lineSplit = lines[2].split()
		self.indptr = np.zeros(len(lineSplit), dtype = np.dtype('int64'))

		for i in range(len(lineSplit)):
			self.indptr[i] = np.int64(lineSplit[i])

		lineSplit = lines[3].split()
		self.indices = np.zeros(len(lineSplit), dtype = np.dtype('int64'))

		for i in range(len(lineSplit)):
			self.indices[i] = np.int64(lineSplit[i])

		lineSplit = lines[4].split()
		self.data = np.zeros(len(lineSplit), dtype = np.dtype('double'))

		for i in range(len(lineSplit)):
			self.data[i] = np.double(lineSplit[i])

		fileCMatrix.close()

	def printCoo(self):

		for j in range(len(self.indptr)-1):

			ptr = self.indptr[j]
			ptrNext = self.indptr[j+1]

			for i in range(ptr, ptrNext):
				index = self.indices[i]
				data = self.data[i]

				if self.mformat:
					print("(%ld, %ld) \t %lf"% (index, j, data))
				else:
					print("(%ld, %ld) \t %lf"% (j, index, data))



