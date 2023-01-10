#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <stdbool.h>
#include "headers/mtx.h"

struct Inputs{
	bool triangularForm;//L = 1, U = 0
	bool solveType;
	char* matrixPath;
	char* rhsPath;
	char* outputPath;
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
	printf("%c", argv[0][0]);
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

	//read matrixPath
	char* rhsPath = (char*) malloc(sizeof(char)*256);

	if(rhsPath == NULL){
		free(matrixPath);
		return 1;
	}

	ReadCharPtrInput(argv[4], rhsPath);
	inputs->rhsPath = rhsPath;

	//read matrixPath
	char* outputPath = (char*) malloc(sizeof(char)*256);
	if(outputPath == NULL){
		free(matrixPath);
		free(rhsPath);
		return 1;
	}

	ReadCharPtrInput(argv[5], outputPath);
	inputs->outputPath = outputPath;

	return 0;
}

int main(int argc, char *argv[]) {

	Inputs input;

	if(argc != 6){
		 fprintf(stderr, "Wrong number of inputs got %d but expected %d.\n",argc,5);
		 exit(1);
	}

	if(ReadInputs(&input, argv)){
		fprintf(stderr, "Error when allocating memory for the inputs.\n");
		exit(1);
	}

	printInputs(&input);

	CondMatrix* myMatrix = ReadMatrix(input.matrixPath, true);
	if(myMatrix == NULL){
		fprintf(stderr, "Failed to read condense matrix.\n");
		return 1;
	}
	printCondMatrix(myMatrix);
	fprintCondMatrix(myMatrix,input.outputPath);
	freeCondMatrix(myMatrix);
	freeInputs(&input);

}