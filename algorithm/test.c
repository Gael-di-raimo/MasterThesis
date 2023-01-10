#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <stdbool.h>
#include <time.h>
#include "headers/mtx.h"
#include "headers/algorithm.h"
#include <processthreadsapi.h>



//code taken from site https://levelup.gitconnected.com/8-ways-to-measure-execution-time-in-c-c-48634458d0f9 26/09/2020 at 9 PM
/*
double get_cpu_time(){
    FILETIME a,b,c,d;
    if (GetProcessTimes(GetCurrentProcess(),&a,&b,&c,&d) != 0){
        //  Returns total user time.
        //  Can be tweaked to include kernel times as well.
        return
            (double)(d.dwLowDateTime |
            ((unsigned long long)d.dwHighDateTime << 32)) * 0.0000001;
    }else{
        //  Handle error
        return 0;
    }
}
*/
struct Inputs{
	bool triangularForm;
	bool solveType;
	char* matrixPath;
	char* rhsPath;
	char* outputPath;
	int testNb;
};
typedef struct Inputs Inputs;


void printInputs(Inputs* inputs){

	printf("Inputs:{\n");
	printf("%d,\n",inputs->triangularForm);
	printf("%d,\n",inputs->solveType);
	printf("%s,\n",inputs->matrixPath);
	printf("%s,\n",inputs->rhsPath);
	printf("%s }\n",inputs->outputPath);

}

void freeInputs(Inputs* inputs){

	free(inputs->matrixPath);
	free(inputs->matrixPath2);
	free(inputs->rhsPath);
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

	//read triangularForm
	if(argv[1][0] == '0'){
		inputs->triangularForm = 0;
	}
	else{
		inputs->triangularForm = 1;
	}

	//read solveType
	if(argv[2][0] == '0'){
		inputs->solveType = 0;
	}
	else{
		inputs->solveType = 1;
	}

	//read matrixPath
	char* matrixPath = (char*) malloc(sizeof(char)*256);

	if(matrixPath == NULL){
		return 1;
	}

	ReadCharPtrInput(argv[3], matrixPath);
	inputs->matrixPath = matrixPath;


	//read rhsPath
	char* rhsPath = (char*) malloc(sizeof(char)*256);

	if(rhsPath == NULL){
		free(matrixPath);
		return 1;
	}

	ReadCharPtrInput(argv[4], rhsPath);
	inputs->rhsPath = rhsPath;

	//read outputPath
	char* outputPath = (char*) malloc(sizeof(char)*256);
	if(outputPath == NULL){
		free(matrixPath);
		free(rhsPath);
		return 1;
	}

	ReadCharPtrInput(argv[5], outputPath);
	inputs->outputPath = outputPath;

	inputs->testNb = atoi(argv[6]);
	
	return 0;
}

int fprintFeatures(Inputs* inputs, clock_t f0_timer, clock_t f1_timer, clock_t f2_timer, clock_t f3_timer){

	FILE* fileFeatures;

	if(inputs->triangularForm){
		ftimers = fopen(input.outputPath+"/featuresL.txt","a+");
	}
	else{
		ftimers = fopen(input.outputPath+"/featuresU.txt","a+");
	}

	fprintf(ftimers, "%d ", input.solveType);

	fprintf(ftimers, "%ld ", CLOCKS_PER_SEC);

	
	fprintf(ftimers, "%ld ", f0_timer);

	fprintf(ftimers, "%ld ", f1_timer);

	fprintf(ftimers, "%ld ", f2_timer);
	
	fprintf(ftimers, "%ld ", f3_timer);

	fprintf(ftimers, "%ld\n", timerOnePhase);

	fclose(ftimers);
}

int fprintTimers(Inputs* inputs, clock_t timerGeneral, clock_t timerTwoPhases, clock_t timerOnePhase){
	
	FILE* ftimers;

	if(inputs->triangularForm){
		ftimers = fopen(input.outputPath+"/timersL.txt","a+");
	}
	else{
		ftimers = fopen(input.outputPath+"/timersU.txt","a+");
	}

	fprintf(ftimers, "%d ", input.solveType);

	fprintf(ftimers, "%ld ", CLOCKS_PER_SEC);

	
	fprintf(ftimers, "%ld ", timerGeneral);

	fprintf(ftimers, "%ld ", timerTwoPhases);

	fprintf(ftimers, "%ld\n", timerOnePhase);

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


	CondMatrix* triangularMatrix = ReadMatrix(input.matrixPath, input.solveType, false);

	if(triangularMatrix == NULL){
		fprintf(stderr, "Failed to read triangular matrix.\n");
		return 1;
	}

	CondMatrix* triangularMatrix = ReadMatrix(input.matrixPath2, input.solveType, false);

	if(triangularMatrix == NULL){
		fprintf(stderr, "Failed to read triangular matrix.\n");
		return 1;
	}

	CondMatrix* rhs = ReadMatrix(input.rhsPath, true, false);

	if(rhs == NULL){
		fprintf(stderr, "Failed to read right-hand side.\n");
		return 1;
	}

	// Starting to run the different algorithm
	clock_t start, end, tmpTimer;

	double timerGeneral = -1, timerTwoPhases = -1, timerOnePhase = -1;

	// general algorithm
	if(input.testNb == 0 || input.testNb == 1){

		start = clock();
		double* solution = generalAlgorithm(triangularMatrix, rhs, input.solveType, input.triangularForm);
		end = clock();

		timerGeneral = end - start;// / CLOCKS_PER_SEC;

		int64_t i = 0;


		if(input.rhsPath->testNb == 1){
		
			FILE* file = fopen("solGeneral.txt","w+");

			for(i = 0; i < rhs->nRow; i++){
				fprintf(file, "%.8lf\n", solution[i]);
			}

			fclose(file);
		}
		free(solution);
	}
	
	// 2phases algorithm
	if(input.testNb == 0 || input.testNb == 2){

		start = get_cpu_time();

		CondMatrix* sol2phases = twoPhasesAlgorithm(triangularMatrix, rhs, input.solveType, input.triangularForm);

		end = get_cpu_time();

		timerTwoPhases = end - start;// / CLOCKS_PER_SEC;
		
		if(input.rhsPath->testNb == 2){
			fprintCondMatrix(sol2phases, "sol2phases.txt");
		}

		freeCondMatrix(sol2phases);
	}

	// 1phase algorithm
	if(input.testNb == 0 || input.testNb == 3){
		
		CondMatrix* sol1phase;

		// 1phase algorithm
		int i;
		
	
		start = get_cpu_time();

		sol1phase = onePhaseAlgorithm(triangularMatrix, rhs, input.solveType, input.triangularForm);

		end = get_cpu_time();

		timerOnePhase = end - start;// / CLOCKS_PER_SEC;

		if(sol1phase == NULL){
			fprintf(stderr,"Got an error in the solving of the one onePhaseAlgorithm\n");
		}

		if(input.rhsPath->testNb == 3){
			fprintCondMatrix(sol1phase, "sol1phase.txt");
		}

		freeCondMatrix(sol1phase);
	}


	if(input.testNb == 0){
		fprintTimers(input.outputPath, timerGeneral, timerTwoPhases, timerOnePhase);
		fprintFeatures()
	}

	freeCondMatrix(rhs);
	freeCondMatrix(triangularMatrix);
	freeInputs(&input);

}