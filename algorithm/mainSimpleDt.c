#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <stdbool.h>
#include <time.h>

#include <windows.h>	/* WinAPI */

#include <processthreadsapi.h>
#include "headers/mtx.h"
#include "headers/algorithm.h"

#include "headers/features.h"

struct Inputs{
	bool solveType;
	bool isLowerTriangular;
	char* mtxPath;
	int fid;
	int sid;
	char* outputPath;
	bool doTest;
};
typedef struct Inputs Inputs;


void printInputs(Inputs* inputs){

	printf("Inputs:{\n");
	printf("%d,\n",inputs->solveType);
	printf("%d,\n",inputs->isLowerTriangular);
	printf("%s,\n",inputs->mtxPath);
	printf("%lld,\n",inputs->fid);
	printf("%lld,\n",inputs->sid);
	printf("%s, \n",inputs->outputPath);
	printf("%d}\n",inputs->doTest);

}

void freeInputs(Inputs* inputs){
	free(inputs->mtxPath);
	free(inputs->outputPath);
}

void ReadCharPtrInput(char* input, char* buffer){

	int i = 0;

	while(input[i] != 0){
		buffer[i] = input[i];
		i++;
	}
	buffer[i] = '\0';
}

//Reads the input of the main fct and fill the struct input
int ReadInputs(Inputs* inputs, char** argv){

	//read solveType
	if(argv[2][0] == '0'){
		inputs->solveType = 0;
	}
	else{
		inputs->solveType = 1;
	}

	if(argv[2][0] == '0'){
		inputs->isLowerTriangular = 0;
	}
	else{
		inputs->isLowerTriangular = 1;
	}

	//read mtx file path
	char* mtxPath = (char*) malloc(sizeof(char)*256);

	if(mtxPath == NULL){
		return 1;
	}

	ReadCharPtrInput(argv[3], mtxPath);
	inputs->mtxPath = mtxPath;

	char buff[256];

	//read factorisation id
	ReadCharPtrInput(argv[4], buff);
	inputs->fid = atoi(buff);;


	//read solve id
	ReadCharPtrInput(argv[5], buff);
	inputs->sid = atoi(buff);;

	//read outputPath
	char* outputPath = (char*) malloc(sizeof(char)*256);
	if(outputPath == NULL){
		free(mtxPath);
		return 1;
	}


	//read the variable to enable the tests (it will put in a file the solutions)
	ReadCharPtrInput(argv[6], outputPath);
	inputs->outputPath = outputPath;

	if(argv[7][0] == '0'){
		inputs->doTest = 0;
	}
	else{
		inputs->doTest = 1;
	}
	
	return 0;
}

char* idToFilename(char mtxLetter, int id){

	char buff[9];
	itoa(id, buff, 10);

	char* fileName = (char*) malloc(sizeof(char)*14);

	char buff2[14] = "L00000000";
	strcpy(fileName, buff2);
	strcpy(&fileName[9-strlen(buff)], buff);
	strcpy(&fileName[9], ".mtx");

	fileName[13] = '\0';
	fileName[0] = mtxLetter;

	return fileName;
}

CondMatrix* arrayToCondMatrix(double* array, int64_t sizeArray){

	int64_t i, nnz = 0;
	for(i = 0; i < sizeArray; i++){
		if(array[i] != 0){
			nnz++;
		}
	}

	CondMatrix* arrayCondMatrix = allocateCondMatrix(nnz, 1, sizeArray, true);
	
	if(arrayCondMatrix == NULL){
		return NULL;
	}

	int64_t	j = 0;
	for(i = 0; i < sizeArray; i++){
		if(array[i] !=0 ){
			arrayCondMatrix->indices[j] = i;
			arrayCondMatrix->data[j] = array[i];
			j++;
		}
	}

	return arrayCondMatrix;
	
}


void fprintFeatures(Inputs* inputs, CondMatrix* triangularMatrix, CondMatrix* rhs, CondMatrix* sol){

	LARGE_INTEGER startTime, endTime, timer;

	//open the file where to write the features
	char* fileFeaturesPath = (char*) malloc(sizeof(char)*256);
	strcpy(fileFeaturesPath, inputs->outputPath);
	strcat(fileFeaturesPath, "/features.txt");

	FILE* fileFeatures = fopen(fileFeaturesPath, "a+");
	free(fileFeaturesPath);

	LARGE_INTEGER frequency;

	QueryPerformanceFrequency(&frequency);
	//write some important data
	fprintf(fileFeatures, "%d ", inputs->solveType);
	
	fprintf(fileFeatures, "%d ", inputs->fid);
	
	fprintf(fileFeatures, "%d ", inputs->sid);

	fprintf(fileFeatures, "%d ", inputs->isLowerTriangular);

	fprintf(fileFeatures, "%lld ", frequency.QuadPart);

	uint64_t res_feature;
	double res_f3;


	//Add the feature for the L solve	
	fprintf(fileFeatures, "%lld ", triangularMatrix->nRow);

	fprintf(fileFeatures, "%lld ", triangularMatrix->nnz);
	
	fprintf(fileFeatures, "%lld ", rhs->nnz);

	fprintf(fileFeatures, "%lld ", sol->nnz);

	//Compute the feature for the computation of L

	//f0
	QueryPerformanceCounter(&startTime);
	res_feature =  f0(triangularMatrix, rhs, 1, inputs->solveType, inputs->isLowerTriangular);
	QueryPerformanceCounter(&endTime);

	timer.QuadPart = endTime.QuadPart - startTime.QuadPart;

	fprintf(fileFeatures, "%llu ", res_feature);
	fprintf(fileFeatures, "%lld ", timer.QuadPart);


	//f1
	QueryPerformanceCounter(&startTime);
	res_feature = f1(triangularMatrix, rhs, 1, inputs->solveType, inputs->isLowerTriangular);
	QueryPerformanceCounter(&endTime);


	timer.QuadPart = endTime.QuadPart - startTime.QuadPart;

	fprintf(fileFeatures, "%llu ", res_feature);
	fprintf(fileFeatures, "%lld ", timer.QuadPart);


	//f2
	QueryPerformanceCounter(&startTime);
	res_feature = f2(triangularMatrix, rhs, 1, inputs->solveType, inputs->isLowerTriangular);
	QueryPerformanceCounter(&endTime);


	timer.QuadPart = endTime.QuadPart - startTime.QuadPart;

	fprintf(fileFeatures, "%llu ", res_feature);
	fprintf(fileFeatures, "%lld ", timer.QuadPart);
	

	//f3
	QueryPerformanceCounter(&startTime);
	res_f3 = f3(triangularMatrix, rhs, 1, inputs->solveType, inputs->isLowerTriangular);
	QueryPerformanceCounter(&endTime);


	timer.QuadPart = endTime.QuadPart - startTime.QuadPart;

	fprintf(fileFeatures, "%lf ", res_f3);
	fprintf(fileFeatures, "%lld ", timer.QuadPart);

	// will add a number which is not really a feature but have importance
	QueryPerformanceCounter(&startTime);
	res_feature =  f0(triangularMatrix, sol, 1, inputs->solveType, inputs->isLowerTriangular);
	QueryPerformanceCounter(&endTime);

	timer.QuadPart = endTime.QuadPart - startTime.QuadPart;

	printf(" sumnnz = %lld\n", res_feature);
	fprintf(fileFeatures, "%lld ", res_feature);
	fprintf(fileFeatures, "%lld\n", timer.QuadPart);



	fclose(fileFeatures);
}

void fprintTimers(Inputs* inputs, LARGE_INTEGER timerGeneral1, LARGE_INTEGER timerTwoPhases1, LARGE_INTEGER timerOnePhase1){
	
	//opent the file for the timers
	char* ftimersPath = (char*) malloc(sizeof(char)*256);
	strcpy(ftimersPath, inputs->outputPath);
	strcat(ftimersPath, "/timers.txt");

	FILE* ftimers = fopen(ftimersPath,"a+");
	free(ftimersPath);

	LARGE_INTEGER frequency;

	QueryPerformanceFrequency(&frequency);

	//write some data
	fprintf(ftimers, "%d ", inputs->solveType);
	
	fprintf(ftimers, "%d ", inputs->fid);
	
	fprintf(ftimers, "%d ", inputs->sid);

	fprintf(ftimers, "%d ", inputs->isLowerTriangular);

	fprintf(ftimers, "%lld ", frequency.QuadPart);

	//write all the timers

	fprintf(ftimers, "%lld ", timerGeneral1.QuadPart);
	fprintf(ftimers, "%lld ", timerTwoPhases1.QuadPart);
	fprintf(ftimers, "%lld\n", timerOnePhase1.QuadPart);
	
	
	fclose(ftimers);

}


int main(int argc, char *argv[]) {

	Inputs input;

	if(argc != 8){
		 fprintf(stderr, "Wrong number of inputs got %d but expected %d.\n",argc-1, 7);
		 exit(1);
	}

	if(ReadInputs(&input, argv)){
		fprintf(stderr, "Error when allocating memory for the inputs.\n");
		exit(1);
	}
	
	char matrix_path[256]= "";
	char bpath[256]= "";

	strcpy(matrix_path, input.mtxPath);
	strcat(matrix_path, "/");
	strcpy(bpath, matrix_path);

	char matrix_form;

	if(input.isLowerTriangular){
		matrix_form = 'L';
	}
	else{
		matrix_form = 'U';
	}

	char* tmpId = idToFilename(matrix_form, input.fid);
	strcat(matrix_path, tmpId);
	free(tmpId);

	tmpId = idToFilename('b', input.sid);
	strcat(bpath, tmpId);
	free(tmpId);

	printf("%s %s\n", matrix_path, bpath);
	
	CondMatrix* matrix = ReadMatrix(matrix_path, input.solveType, false);

	if(matrix == NULL){
		fprintf(stderr, "Failed to read triangular matrix.\n");
		return 1;
	}

	CondMatrix* b = ReadMatrix(bpath, true, false);

	if(b == NULL){
		fprintf(stderr, "Failed to read right-hand side.\n");
		return 1;
	}


	LARGE_INTEGER startTime, endTime;
	

	// Starting to run the different algorithm

	// general algorithm

	LARGE_INTEGER timerGeneral1;
	double* solution;

	QueryPerformanceCounter(&startTime);
	solution = generalAlgorithm(matrix, (void*) b, input.solveType, input.isLowerTriangular, true);
	QueryPerformanceCounter(&endTime);

	
	timerGeneral1.QuadPart = endTime.QuadPart - startTime.QuadPart;


	if(input.doTest){

		CondMatrix* solutionCondMatrix = arrayToCondMatrix(solution, b->nRow);
		
		fprintCondMatrix(solutionCondMatrix, "solGeneral.txt");
		freeCondMatrix(solutionCondMatrix);
	}
	
	free(solution);



	// 2phases algorithm
	LARGE_INTEGER timerTwoPhases1;
	CondMatrix* sol2phases1;


	QueryPerformanceCounter(&startTime);
	sol2phases1 = twoPhasesAlgorithm(matrix, b, input.solveType, input.isLowerTriangular);
	QueryPerformanceCounter(&endTime);

	timerTwoPhases1.QuadPart = endTime.QuadPart - startTime.QuadPart;
	
	if(input.doTest){
		fprintCondMatrix(sol2phases1, "sol2phases.txt");
	}

	freeCondMatrix(sol2phases1);


	// 1phase algorithm

	LARGE_INTEGER timerOnePhase1;

	CondMatrix* sol1phase1;

	QueryPerformanceCounter(&startTime);
	sol1phase1 = onePhaseAlgorithm(matrix, b, input.solveType, input.isLowerTriangular);
	QueryPerformanceCounter(&endTime);

	timerOnePhase1.QuadPart = endTime.QuadPart - startTime.QuadPart;

	if(input.doTest){
		fprintCondMatrix(sol1phase1, "sol1phase.txt");
	}

	//recomputing the first solution to compute the feature (since it may have been adjusted for the solve
	CondMatrix* sol;
	sol = onePhaseAlgorithm(matrix, b, input.solveType, input.isLowerTriangular);

	//writting all the timers in a file
	fprintTimers(&input, timerGeneral1, timerTwoPhases1, timerOnePhase1);
	
	//compute the features
	fprintFeatures(&input, matrix, b, sol);

	printf("The nnz of the sol is : %lld\n",sol->nnz);

	freeCondMatrix(sol1phase1);
	freeCondMatrix(sol);
	freeCondMatrix(b);
	freeCondMatrix(matrix);
	freeInputs(&input);

}