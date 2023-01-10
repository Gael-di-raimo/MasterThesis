#include "headers/mtx.h"

struct CooElement{
	int64_t row;
	int64_t col;
	double data;
};

typedef struct CooElement CooElement;

static int* GetShape(FILE* mFile, int64_t* nRow, int64_t* nCol, int64_t* nnz){

	int resScan = 0;

	while(resScan == 0){
		resScan = fscanf(mFile,"%lld %lld %lld\n", nRow, nCol, nnz);

		if(resScan == 0){

			char buff[256];
			fscanf(mFile,"%[^\n]", buff);

		}
	}

	return 0;
}

static int compCol(const void * a, const void * b){
	return ((CooElement*)a)->col - ((CooElement*)b)->col;
}

static int compRow(const void * a, const void * b){
	return ((CooElement*)a)->row - ((CooElement*)b)->row;
}

static CooElement* ReadCooMatrix(FILE* mFile, int64_t nnz, char* path){

	CooElement* cooMatrix = (CooElement*) malloc(sizeof(CooElement)*nnz);

	if(cooMatrix == NULL){
		fprintf(stderr, "Failed to allocate a cooMatrix for reading a matrix at path %s.\n", path);
		return NULL;
	}

	int64_t elemRead = 0;
	int resScan;
	while(elemRead < nnz){

		resScan = fscanf(mFile,"%lld %lld %lf\n", &(cooMatrix[elemRead].row), &(cooMatrix[elemRead].col),  &(cooMatrix[elemRead].data));

		if(resScan == 3){
			elemRead++;
		}
		else{
			char buff[256];
			resScan = fscanf(mFile,"%s\n", buff);
			fprintf(stderr, "Error while reading mtx file at path %s, read invalid line %s\n",path, buff);
			return NULL;

		}

	}

	return cooMatrix;
}

static int allocCondMatrixArrays(CondMatrix* condMatrix, char* path){

	int64_t sizeIndPtr;

	if(condMatrix->format){
		sizeIndPtr = condMatrix->nCol + 1;
	}
	else{
		sizeIndPtr = condMatrix->nRow + 1;
	}

	condMatrix->indPtr = (int64_t*) malloc(sizeof(int64_t)*sizeIndPtr);
	condMatrix->indices = (int64_t*) malloc(sizeof(int64_t)*condMatrix->nnz);
	condMatrix->data = (double*) malloc(sizeof(double)*condMatrix->nnz);

	if(condMatrix->indPtr == NULL || condMatrix->indices == NULL || condMatrix->data == NULL){

		fprintf(stderr, "Failed to create a condense matrix reading mtx file at path %s.\n", path);

		if(condMatrix->indPtr != NULL)
			free(condMatrix->indPtr);

		if(condMatrix->indices != NULL)
			free(condMatrix->indices);

		if(condMatrix->data != NULL)
			free(condMatrix->data);

		return 1;
	}
	return 0;
}

static int CreateCondMatrix(CondMatrix* condMatrix, CooElement* cooMatrix, char* path){

	if(condMatrix->format){
		//csc format so sort element per column
		qsort(cooMatrix, condMatrix->nnz, sizeof(CooElement), compCol);
	}
	else{
		//csr format so sort element per row
		qsort(cooMatrix, condMatrix->nnz, sizeof(CooElement), compRow);

	}

	int64_t i, prevIndInCoo, indInCoo, sizeIndPtr;

	if(condMatrix->format){
		sizeIndPtr = condMatrix->nCol + 1;
	}
	else{
		sizeIndPtr = condMatrix->nRow + 1;
	}

  	if(allocCondMatrixArrays(condMatrix, path)){
		return 1;
	}

	if(condMatrix == NULL){
		fprintf(stderr, "Failed to create a condense matrix reading mtx file at path %s.\n", path);
	}

	indInCoo = 0;

	if(condMatrix->nnz > 0)
		condMatrix->indPtr[0] = 0;

	if(condMatrix->indicesSorted){

		//if specified the indices are also sorted
		for(i = 0; i < sizeIndPtr ; i++){

			if(condMatrix->format){

				if(cooMatrix[indInCoo].col == i ){
					prevIndInCoo = indInCoo;
					indInCoo++;

					while(indInCoo < condMatrix->nnz && cooMatrix[indInCoo].col == i){
						indInCoo++;
					}

					qsort(&cooMatrix[prevIndInCoo], indInCoo - prevIndInCoo, sizeof(CooElement), compRow);

				}
			}
			else{
				if(cooMatrix[indInCoo].row == i){
					prevIndInCoo = indInCoo;
					indInCoo++;
					while(indInCoo < condMatrix->nnz && cooMatrix[indInCoo].row == i){
						indInCoo++;
					}
					qsort(&cooMatrix[prevIndInCoo], indInCoo - prevIndInCoo, sizeof(CooElement), compCol);
				}
			}

		}
	}
	
	indInCoo = 0;
	//filling the indptr array
	for(i = 1; i < sizeIndPtr - 1; i++){

		if(condMatrix->format){

			if(cooMatrix[indInCoo].col == i){
				indInCoo++;
				while(indInCoo < condMatrix->nnz && cooMatrix[indInCoo].col == i){
					indInCoo++;
				}
			}
		}
		else{
			if(cooMatrix[indInCoo].row == i){
				indInCoo++;
				while(indInCoo < condMatrix->nnz && cooMatrix[indInCoo].row == i){
					indInCoo++;
				}
			}
		}

		condMatrix->indPtr[i] = indInCoo;

	}

	condMatrix->indPtr[sizeIndPtr - 1] = condMatrix->nnz;

	//copy the row array
	for(i = 0; i < condMatrix->nnz; i++){
		if(condMatrix->format){
			condMatrix->indices[i] = cooMatrix[i].row - 1;
		}
		else{
			condMatrix->indices[i] = cooMatrix[i].col - 1;
		}
		condMatrix->data[i] = cooMatrix[i].data;
	}
	return 0;
}

int fprintCondMatrix(CondMatrix* condMatrix, char* path){

	FILE* file = fopen(path,"w");
	if(file == NULL){
		fprintf(stderr, "Error while trying to open %s when writting a condMatrix\n", path);
	}
	//writting the format
	if(condMatrix->format)
		fprintf(file,"1\n");
	else{
		fprintf(file,"0\n");
	}

	//writting the size
	fprintf(file,"%lld %lld %lld\n", condMatrix->nRow, condMatrix->nCol, condMatrix->nnz);
	/*if(condMatrix->nnz == 0){
		fprintf(file,"\n\n\n");
		fclose(file);
		return 0;

	}*/
	//writting the indptr array
	int i;

	for(i = 0; i < condMatrix->nCol + 1; i++){
		fprintf(file,"%lld ", condMatrix->indPtr[i]);
	}
	fprintf(file,"\n");

	//writting the indices array

	for(i = 0; i < condMatrix->nnz; i++){
		fprintf(file,"%lld ", condMatrix->indices[i]);
	}
	fprintf(file,"\n");

	//writting the data array

	for(i = 0; i < condMatrix->nnz; i++){
		fprintf(file,"%.20lf ", condMatrix->data[i]);
	}
	fprintf(file,"\n");
	fclose(file);
	return 0;

}

void freeCondMatrix(CondMatrix* matrix){
	if(matrix->indPtr != NULL){
		free(matrix->indPtr);
		free(matrix->indices);
		free(matrix->data);
	}
	free(matrix);
}

void printCondMatrix(CondMatrix* matrix){

	int64_t i = 0, sizeIndPtr;
	printf("Matrix:{\n");
	printf("\t indPtr = [");

	if(matrix->format){
		sizeIndPtr = matrix->nCol + 1;
	}
	else{
		sizeIndPtr = matrix->nRow + 1;
	}

	for(i = 0; i < sizeIndPtr; i++){
		printf("%lld, ",matrix->indPtr[i]);
	}

	printf("]\n");

	printf("\t indices = [");

	for(i = 0; i < matrix->nnz; i++){
		printf("%lld, ",matrix->indices[i]);
	}

	printf("]\n");

	printf("\t data = [");

	for(i = 0; i < matrix->nnz; i++){
		printf("%lf, ",matrix->data[i]);
	}

	printf("]\n}\n");

}

CondMatrix* allocateCondMatrix(int64_t nnz, int64_t nCol, int64_t nRow, bool format){

	//allocate a cond matrix with the correpsonding computed space for the solution

	//transform in a fct ?
	CondMatrix* matrix = (CondMatrix*) malloc(sizeof(CondMatrix));

	if(matrix == NULL){
		fprintf(stderr, "Error while allocating compressed matrix.");
		return NULL;
	}

	matrix->indPtr = (int64_t*) malloc(sizeof(int64_t)*(nCol+1));

	if(matrix->indPtr == NULL){
		fprintf(stderr, "Error while allocating  compressed matrix.");
		free(matrix);
		return NULL;
	}
	matrix->indPtr[0] = 0;
	matrix->indPtr[nCol] = nnz;
	
	if(nnz != 0){

		matrix->indices = (int64_t*) malloc(sizeof(int64_t)*nnz);

		if(matrix->indices == NULL){
			fprintf(stderr, "Error while allocating compressed matrix.");
			free(matrix->indPtr);
			free(matrix);
			return NULL;
		}
		matrix->data = (double*) malloc(sizeof(double)*nnz);

		if(matrix->data == NULL){

			fprintf(stderr, "Error while allocating compressed matrix.");
			free(matrix->indices);
			free(matrix->indPtr);
			free(matrix);

			return NULL;
		}
	}
	else{
		matrix->indices = NULL;
		matrix->data = NULL;
	}


	matrix->nnz = nnz;
	matrix->nCol = nCol;
	matrix->nRow = nRow;
	matrix->format = format;

	return matrix;
}

CondMatrix* ReadMatrix(char* path, bool format, bool indicesSorted){

	FILE * mFile;
	mFile  = fopen(path, "r");

	if(mFile == NULL){
		fprintf(stderr, "Failed to open the matrix at path %s.\n", path);
		return NULL;
	}

	int64_t nRow, nCol, nnz;
	GetShape( mFile, &nRow, &nCol, &nnz);

	//First read the data as coordinate matrix arrays

	CondMatrix* readMatrix;

	if(nnz != 0){

		// It will create a matrix for matrix with at least one non-zeros
		CooElement* cooMatrix = ReadCooMatrix(mFile, nnz, path);
		if(cooMatrix == NULL){
			fprintf(stderr, "Failed to read coordinate matrix to create condense matrix %s.\n", path);
			return NULL;
		}

		readMatrix = (CondMatrix*) malloc(sizeof(CondMatrix));

		if(readMatrix == NULL){
			fprintf(stderr, "Failed to allocate to return matrix %s.\n", path);
			free(cooMatrix);
			return NULL;
		}

		readMatrix->nnz = nnz;
		readMatrix->nCol = nCol;
		readMatrix->nRow = nRow;
		readMatrix->format = format;
		readMatrix->indicesSorted = indicesSorted;

		if(CreateCondMatrix(readMatrix, cooMatrix, path)){
			free(cooMatrix);
			free(readMatrix);
			return NULL;
		}
		free(cooMatrix);
	}
	else{
		readMatrix = (CondMatrix*) malloc(sizeof(CondMatrix));

		readMatrix->nnz = nnz;
		readMatrix->nCol = nCol;
		readMatrix->nRow = nRow;
		readMatrix->format = format;
		readMatrix->indicesSorted = indicesSorted;

		int64_t i, sizeIndPtr;
		if(readMatrix->format){
			sizeIndPtr = readMatrix->nCol + 1;
		}
		else{
			sizeIndPtr = readMatrix->nRow + 1;
		}

		readMatrix->indPtr = (int64_t*) malloc(sizeof(int64_t)*sizeIndPtr);

		for(i = 0; i < sizeIndPtr; i++){
			readMatrix->indPtr[i] = 0;
		}

		readMatrix->indices = NULL;
		readMatrix->data = NULL;

	}

	fclose(mFile);
	return readMatrix;
}
