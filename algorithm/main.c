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
	if(argv[1][0] == '0'){
		inputs->solveType = 0;
	}
	else{
		inputs->solveType = 1;
	}

	//read mtx file path
	char* mtxPath = (char*) malloc(sizeof(char)*256);

	if(mtxPath == NULL){
		return 1;
	}

	ReadCharPtrInput(argv[2], mtxPath);
	inputs->mtxPath = mtxPath;

	char buff[256];

	//read factorisation id
	ReadCharPtrInput(argv[3], buff);
	inputs->fid = atoi(buff);;


	//read solve id
	ReadCharPtrInput(argv[4], buff);
	inputs->sid = atoi(buff);;

	//read outputPath
	char* outputPath = (char*) malloc(sizeof(char)*256);
	if(outputPath == NULL){
		free(mtxPath);
		return 1;
	}


	//read the variable to enable the tests (it will put in a file the solutions)
	ReadCharPtrInput(argv[5], outputPath);
	inputs->outputPath = outputPath;

	if(argv[6][0] == '0'){
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

double* adjustRhsArray(double* rhs, int64_t rhsNbRow, int64_t matrixNbRow){
	
	if(rhsNbRow < matrixNbRow){
		//adding zeros in the b to have a size of matrixNbRow

		rhs = (double*) realloc( rhs, (matrixNbRow)*sizeof(double));

		if(rhs == NULL){
			//failed realloc
			return NULL;
		}
		int64_t i;
		
		for(i = rhsNbRow; i < matrixNbRow; i++){
			rhs[i] = 0;
		}
	}
	return rhs;

}

void adjustRhsCondMatrix(CondMatrix* rhs, int64_t matrixNbRow){
	
	if(rhs->nRow < matrixNbRow){
		//adding zeros in the rhs to have a size of matrixNbRow
		rhs->nRow = matrixNbRow;
	}

	if(rhs->nRow > matrixNbRow){
		int64_t i, j = 0, newNnz = 0;

		for(i = 0; i < rhs->nnz; i++){

			if(rhs->indices[i] < matrixNbRow){
				rhs->indices[newNnz] = rhs->indices[i];
				rhs->data[newNnz] = rhs->data[i];
				newNnz++;
			}

		}

		rhs->nRow = matrixNbRow;
		rhs->nnz = newNnz;
		rhs->indPtr[1] = newNnz;
	}

}

void fprintFeatures(Inputs* inputs, CondMatrix* triangularMatrix, CondMatrix* triangularMatrix2, CondMatrix* rhs, CondMatrix* rhs2, CondMatrix* sol, CondMatrix* sol2){

	LARGE_INTEGER startTime, endTime, timer;

	//open the file where to write the features
	char* fileFeaturesPath = (char*) malloc(sizeof(char)*256);
	strcpy(fileFeaturesPath, inputs->outputPath);
	strcat(fileFeaturesPath, "/features.txt");

	FILE* fileFeatures = fopen(fileFeaturesPath, "a+");
	free(fileFeaturesPath);

	CondMatrix* L, *U, *LSolveRhs, *USolveRhs,*LSol,*USol;

	if(inputs->solveType){
		L = triangularMatrix;
		LSolveRhs = rhs;
		LSol = sol;

		U = triangularMatrix2;
		USolveRhs = rhs2;
		USol = sol2;
	}
	else{

		L = triangularMatrix2;
		LSolveRhs = rhs2;
		LSol = sol2;

		U = triangularMatrix;
		USolveRhs = rhs;
		USol = sol;
	}

	LARGE_INTEGER frequency;

	QueryPerformanceFrequency(&frequency);
	//write some important data
	fprintf(fileFeatures, "%d ", inputs->solveType);
	
	fprintf(fileFeatures, "%d ", inputs->fid);
	
	fprintf(fileFeatures, "%d ", inputs->sid);

	fprintf(fileFeatures, "%lld ", frequency.QuadPart);

	uint64_t res_feature;
	double res_f3;


	//Add the feature for the L solve	
	fprintf(fileFeatures, "%lld ", L->nRow);

	fprintf(fileFeatures, "%lld ", L->nnz);
	
	fprintf(fileFeatures, "%lld ", LSolveRhs->nnz);

	fprintf(fileFeatures, "%lld ", LSol->nnz);

	//Compute the feature for the computation of L

	//f0
	QueryPerformanceCounter(&startTime);
	res_feature =  f0(L, LSolveRhs, 1, inputs->solveType, true);
	QueryPerformanceCounter(&endTime);

	timer.QuadPart = endTime.QuadPart - startTime.QuadPart;

	fprintf(fileFeatures, "%llu ", res_feature);
	fprintf(fileFeatures, "%lld ", timer.QuadPart);


	//f1
	QueryPerformanceCounter(&startTime);
	res_feature = f1(L, LSolveRhs, 1, inputs->solveType, true);
	QueryPerformanceCounter(&endTime);


	timer.QuadPart = endTime.QuadPart - startTime.QuadPart;

	fprintf(fileFeatures, "%llu ", res_feature);
	fprintf(fileFeatures, "%lld ", timer.QuadPart);


	//f2
	QueryPerformanceCounter(&startTime);
	res_feature = f2(L, LSolveRhs, 1, inputs->solveType, true);
	QueryPerformanceCounter(&endTime);


	timer.QuadPart = endTime.QuadPart - startTime.QuadPart;

	fprintf(fileFeatures, "%llu ", res_feature);
	fprintf(fileFeatures, "%lld ", timer.QuadPart);
	

	//f3
	QueryPerformanceCounter(&startTime);
	res_f3 = f3(L, LSolveRhs, 1, inputs->solveType, true);
	QueryPerformanceCounter(&endTime);


	timer.QuadPart = endTime.QuadPart - startTime.QuadPart;

	fprintf(fileFeatures, "%lf ", res_f3);
	fprintf(fileFeatures, "%lld ", timer.QuadPart);


	//Add the feature for the U solve	
	fprintf(fileFeatures, "%lld ", U->nRow);

	fprintf(fileFeatures, "%lld ", U->nnz);
	
	fprintf(fileFeatures, "%lld ", USolveRhs->nnz);

	fprintf(fileFeatures, "%lld ", USol->nnz);

	//Compute the feature for the computation of U

	//f0
	QueryPerformanceCounter(&startTime);
	res_feature =  f0(U, USolveRhs, 1, inputs->solveType, false);
	QueryPerformanceCounter(&endTime);


	timer.QuadPart = endTime.QuadPart - startTime.QuadPart;

	fprintf(fileFeatures, "%llu ", res_feature);
	fprintf(fileFeatures, "%lld ", timer.QuadPart);


	//f1
	QueryPerformanceCounter(&startTime);
	res_feature = f1(U, USolveRhs, 1, inputs->solveType, false);
	QueryPerformanceCounter(&endTime);

	timer.QuadPart = endTime.QuadPart - startTime.QuadPart;

	fprintf(fileFeatures, "%llu ", res_feature);
	fprintf(fileFeatures, "%lld ", timer.QuadPart);


	//f2
	QueryPerformanceCounter(&startTime);
	res_feature = f2(U, USolveRhs, 1, inputs->solveType, false);
	QueryPerformanceCounter(&endTime);

	timer.QuadPart = endTime.QuadPart - startTime.QuadPart;

	fprintf(fileFeatures, "%llu ", res_feature);
	fprintf(fileFeatures, "%lld ", timer.QuadPart);
	

	//f3
	QueryPerformanceCounter(&startTime);
	res_f3 = f3(U, USolveRhs, 1, inputs->solveType, false);
	QueryPerformanceCounter(&endTime);

	timer.QuadPart = endTime.QuadPart - startTime.QuadPart;

	fprintf(fileFeatures, "%lf ", res_f3);
	fprintf(fileFeatures, "%lld\n", timer.QuadPart);



	fclose(fileFeatures);
}

void fprintTimers(Inputs* inputs, LARGE_INTEGER timerGeneral1, LARGE_INTEGER timerGeneral2, LARGE_INTEGER timerTwoPhases1, LARGE_INTEGER timerTwoPhases2, LARGE_INTEGER timerOnePhase1, LARGE_INTEGER timerOnePhase2){
	
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

	fprintf(ftimers, "%lld ", frequency.QuadPart);

	//write all the timers
	if(inputs->solveType){
		//L solve is the solve to find y
		fprintf(ftimers, "%lld ", timerGeneral1.QuadPart);
		fprintf(ftimers, "%lld ", timerTwoPhases1.QuadPart);
		fprintf(ftimers, "%lld ", timerOnePhase1.QuadPart);

		//U solve is the solve to find x
		fprintf(ftimers, "%lld ", timerGeneral2.QuadPart);
		fprintf(ftimers, "%lld ", timerTwoPhases2.QuadPart);
		fprintf(ftimers, "%lld\n", timerOnePhase2.QuadPart);

	}
	else{
		//L solve is the solve to find x
		fprintf(ftimers, "%lld ", timerGeneral2.QuadPart);
		fprintf(ftimers, "%lld ", timerTwoPhases2.QuadPart);
		fprintf(ftimers, "%lld ", timerOnePhase2.QuadPart);

		//U solve is the solve to find y
		fprintf(ftimers, "%lld ", timerGeneral1.QuadPart);
		fprintf(ftimers, "%lld ", timerTwoPhases1.QuadPart);
		fprintf(ftimers, "%lld\n", timerOnePhase1.QuadPart);
	}
	
	fclose(ftimers);

}



int main(int argc, char *argv[]) {

	Inputs input;

	if(argc != 7){
		 fprintf(stderr, "Wrong number of inputs got %d but expected %d.\n",argc-1, 6);
		 exit(1);
	}

	if(ReadInputs(&input, argv)){
		fprintf(stderr, "Error when allocating memory for the inputs.\n");
		exit(1);
	}
	
	char Lpath[256]= "";
	char Upath[256]= "";
	char bpath[256]= "";

	strcpy(Lpath, input.mtxPath);
	strcat(Lpath, "/");
	strcpy(Upath, Lpath);
	strcpy(bpath, Lpath);

	char* tmpId = idToFilename('L', input.fid);
	strcat(Lpath, tmpId);
	free(tmpId);

	tmpId = idToFilename('U', input.fid);
	strcat(Upath, tmpId);
	free(tmpId);

	tmpId = idToFilename('b', input.sid);
	strcat(bpath, tmpId);
	free(tmpId);

	printf("%s %s %s\n",Lpath, Upath, bpath);
	
	CondMatrix* L = ReadMatrix(Lpath, input.solveType, false);

	if(L == NULL){
		fprintf(stderr, "Failed to read triangular matrix.\n");
		return 1;
	}

	CondMatrix* U = ReadMatrix(Upath, input.solveType, false);

	if(U == NULL){
		fprintf(stderr, "Failed to read triangular matrix.\n");
		return 1;
	}

	CondMatrix* b = ReadMatrix(bpath, true, false);

	if(b == NULL){
		fprintf(stderr, "Failed to read right-hand side.\n");
		return 1;
	}

	CondMatrix* triangularMatrix,* triangularMatrix2;
	
	if(input.solveType == 0){
		triangularMatrix = U;
		triangularMatrix2 = L;
	}
	else{
		triangularMatrix = L;
		triangularMatrix2 = U;
	}


	LARGE_INTEGER startTime, endTime;
	
	//adjusting the rhs

	if(b->nRow != triangularMatrix->nRow){
		adjustRhsCondMatrix(b, triangularMatrix->nRow);
	}

	// Starting to run the different algorithm

	// general algorithm

	LARGE_INTEGER timerGeneral1, timerGeneral2;

	double* solution,* solution2;

	QueryPerformanceCounter(&startTime);
	solution = generalAlgorithm(triangularMatrix, (void*) b, input.solveType, input.solveType, true);
	QueryPerformanceCounter(&endTime);

	timerGeneral1.QuadPart = endTime.QuadPart - startTime.QuadPart;

	if(b->nRow != triangularMatrix2->nRow){
		
		solution = adjustRhsArray(solution, b->nRow, triangularMatrix2->nRow);

		if(solution == NULL){
			fprintf(stderr, "Error while adjusting the 1st solution of the general solver.\n");
			return 1;
		}
	}

	QueryPerformanceCounter(&startTime);
	solution2 = generalAlgorithm(triangularMatrix2, (void*) solution, input.solveType, !input.solveType, false);
	QueryPerformanceCounter(&endTime);
	
	timerGeneral2.QuadPart = endTime.QuadPart - startTime.QuadPart;

	if(input.doTest){

		CondMatrix* solutionCondMatrix = arrayToCondMatrix(solution, b->nRow);
		CondMatrix* solution2CondMatrix = arrayToCondMatrix(solution2, triangularMatrix2->nRow);
		
		fprintCondMatrix(solutionCondMatrix, "solGeneral.txt");
		fprintCondMatrix(solution2CondMatrix, "solGeneral2.txt");

		freeCondMatrix(solutionCondMatrix);
		freeCondMatrix(solution2CondMatrix);
	}
	
	free(solution);
	free(solution2);


	// 2phases algorithm
	LARGE_INTEGER timerTwoPhases1, timerTwoPhases2;
	CondMatrix* sol2phases1,* sol2phases2;


	QueryPerformanceCounter(&startTime);
	sol2phases1 = twoPhasesAlgorithm(triangularMatrix, b, input.solveType, input.solveType);
	QueryPerformanceCounter(&endTime);

	timerTwoPhases1.QuadPart = endTime.QuadPart - startTime.QuadPart;
	
	if(input.doTest){
		fprintCondMatrix(sol2phases1, "sol2phases.txt");
	}


	if(sol2phases1->nRow != triangularMatrix2->nRow){
		adjustRhsCondMatrix(sol2phases1, triangularMatrix2->nRow);

	}
	
	QueryPerformanceCounter(&startTime);
	sol2phases2 = twoPhasesAlgorithm(triangularMatrix2, sol2phases1, input.solveType, !input.solveType);
	QueryPerformanceCounter(&endTime);

	timerTwoPhases2.QuadPart = endTime.QuadPart - startTime.QuadPart;

	if(input.doTest){
		fprintCondMatrix(sol2phases2, "sol2phases2.txt");
	}

	freeCondMatrix(sol2phases1);
	freeCondMatrix(sol2phases2);



	// 1phase algorithm

	LARGE_INTEGER timerOnePhase1, timerOnePhase2;

	CondMatrix* sol1phase1,* sol1phase2;

	QueryPerformanceCounter(&startTime);
	sol1phase1 = onePhaseAlgorithm(triangularMatrix, b, input.solveType, input.solveType);
	QueryPerformanceCounter(&endTime);

	if(input.doTest){
		fprintCondMatrix(sol1phase1, "sol1phase.txt");
	}

	//adjusting the computed solution to the triangular matrix
	if(sol1phase1->nRow != triangularMatrix2->nRow){
		adjustRhsCondMatrix(sol1phase1, triangularMatrix2->nRow);
	}

	timerOnePhase1.QuadPart = endTime.QuadPart - startTime.QuadPart;

	QueryPerformanceCounter(&startTime);
	sol1phase2 = onePhaseAlgorithm(triangularMatrix2, sol1phase1, input.solveType, !input.solveType);
	QueryPerformanceCounter(&endTime);
	
	if(input.doTest){
		fprintCondMatrix(sol1phase2, "sol1phase2.txt");
	}

	timerOnePhase2.QuadPart = endTime.QuadPart - startTime.QuadPart;

	//recomputing the first solution to compute the feature (since it may have been adjusted for the solve
	CondMatrix* sol;
	sol = onePhaseAlgorithm(triangularMatrix, b, input.solveType, input.solveType);

	//writting all the timers in a file
	fprintTimers(&input, timerGeneral1, timerGeneral2, timerTwoPhases1, timerTwoPhases2, timerOnePhase1, timerOnePhase2);
	
	//compute the features
	fprintFeatures(&input, triangularMatrix, triangularMatrix2, b, sol1phase1, sol, sol1phase2);

	freeCondMatrix(sol1phase1);
	freeCondMatrix(sol1phase2);
	freeCondMatrix(sol);
	freeCondMatrix(b);
	freeCondMatrix(triangularMatrix);
	freeCondMatrix(triangularMatrix2);
	freeInputs(&input);

}