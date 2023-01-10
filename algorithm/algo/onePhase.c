#include "../headers/algorithm.h"
#include "../headers/heap.h"


CondMatrix* onePhaseAlgorithmLForm(CondMatrix* triangularMatrix, CondMatrix* rhs){
	int64_t i;
	bool debug = false;
	Heap* heap =  buildMinHeapFromArray(rhs->indices, rhs->data, rhs->nnz);

	if(heap == NULL){
		//error when the heap was built
		return NULL;
	}
	
	HeapNode* node = remTopMinHeapItem(heap);

	int64_t solutionCapacity = rhs->nnz;
	int64_t solutionNnz = 0;
	int64_t lastElementIndex;


	CondMatrix* solution = allocateCondMatrix(rhs->nnz, 1, rhs->nRow, 1);

	while(node != NULL){

		if(debug){
			printf("In the onePhaseLoop at index %lld, %lf\n", node->key, node->data);
		}

		if(solutionCapacity == solutionNnz){
			if(debug){
				printf("Allocating more memory %lld wasn't enough\n",solutionNnz);
			}
			//add more element in the sol with realloc
			solutionCapacity = solutionCapacity*2;
			
			//realloc indices
			solution->indices = realloc(solution->indices, solutionCapacity*sizeof(int64_t));

			if(solution->indices == NULL){
				//failed realloc
				return NULL;
			}

			//realloc data
			solution->data = realloc(solution->data, solutionCapacity*sizeof(double));
			
			if(solution->data == NULL){
				//failed realloc
				return NULL;
			}

			if(debug){
				printf("New capacity is %lld\n", solutionCapacity);
			}
		}

		//computing the final value of the solution at index: node->key
		solution->indices[solutionNnz] = node->key;
		solution->data[solutionNnz] = 0;

		lastElementIndex = node->key;
		
		while(node != NULL && node->key == lastElementIndex){
			solution->data[solutionNnz] = solution->data[solutionNnz] + node->data;
			node = remTopMinHeapItem(heap);

			if(debug && node != NULL){
				printf("In the second loop onePhase node have: %lld, %lf\n", node->key, node->data);
			}
			
		}


		for(i = triangularMatrix->indPtr[lastElementIndex]; i < triangularMatrix->indPtr[lastElementIndex+1]; i++){
			
			if(triangularMatrix->indices[i] == lastElementIndex){
				solution->data[solutionNnz] = solution->data[solutionNnz]/triangularMatrix->data[i];
				break;
			}

		}

		//adding the value to remove to index i of the solution on the heap
		for(i = triangularMatrix->indPtr[lastElementIndex];i < triangularMatrix->indPtr[lastElementIndex+1]; i++){
			
			if(triangularMatrix->indices[i] != lastElementIndex){
				addMinHeapItem(heap, triangularMatrix->indices[i], -triangularMatrix->data[i]*solution->data[solutionNnz]);
				
				if(debug){
					printf("Adding index with val: %lld, %lf\n", triangularMatrix->indices[i],-triangularMatrix->data[i]*solution->data[solutionNnz]);
				}

			}

		}

		if(node == NULL){
			node = remTopMinHeapItem(heap);
		}

		if(debug)
			printf("Sol[%lld]= %lf\n",lastElementIndex, solution->data[solutionNnz]);
		solutionNnz++;
	}

	solution->indPtr[0] = 0;
	solution->indPtr[1] = solutionNnz;
	solution->nnz = solutionNnz;
	if(debug)
		printf("Finished solution onePhase.\n");
	freeHeap(heap);

	return solution;
}

CondMatrix* onePhaseAlgorithmUForm(CondMatrix* triangularMatrix, CondMatrix* rhs){

	int64_t i;
	bool debug = false;

	Heap* heap =  buildMaxHeapFromArray(rhs->indices, rhs->data, rhs->nnz);

	if(heap == NULL){
		//error when the heap was built
		return NULL;
	}

	HeapNode* node = remTopMaxHeapItem(heap);

	int64_t solutionCapacity = rhs->nnz;
	int64_t solutionNnz = 0;
	int64_t lastElementIndex;


	CondMatrix* solution = allocateCondMatrix(rhs->nnz, 1, rhs->nRow, 1);

	while(node != NULL){

		if(debug){
			printf("In the onePhaseLoop at index %lld, %lf\n", node->key, node->data);
		}

		if(solutionCapacity == solutionNnz){
			if(debug){
				printf("Allocating more memory %lld wasn't enough\n",solutionNnz);
			}
			//add more element in the sol with realloc
			solutionCapacity = solutionCapacity*2;
			
			//realloc indices
			solution->indices = realloc(solution->indices, solutionCapacity*sizeof(int64_t));

			if(solution->indices == NULL){
				//failed realloc
				return NULL;
			}

			//realloc data
			solution->data = realloc(solution->data, solutionCapacity*sizeof(double));
			
			if(solution->data == NULL){
				//failed realloc
				return NULL;
			}

			if(debug){
				printf("New capacity is %lld\n", solutionCapacity);
			}
		}

		//computing the final value of the solution at index: node->key
		solution->indices[solutionNnz] = node->key;
		solution->data[solutionNnz] = 0;

		lastElementIndex = node->key;
		
		while(node != NULL && node->key == lastElementIndex){
			solution->data[solutionNnz] = solution->data[solutionNnz] + node->data;
			node = remTopMaxHeapItem(heap);

			if(debug && node != NULL){
				printf("In the second loop onePhase node have: %lld, %lf\n", node->key, node->data);
			}
			
		}


		for(i = triangularMatrix->indPtr[lastElementIndex]; i < triangularMatrix->indPtr[lastElementIndex+1]; i++){
			
			if(triangularMatrix->indices[i] == lastElementIndex){
				solution->data[solutionNnz] = solution->data[solutionNnz]/triangularMatrix->data[i];
				break;
			}

		}

		//adding the value to remove to index i of the solution on the heap
		for(i = triangularMatrix->indPtr[lastElementIndex];i < triangularMatrix->indPtr[lastElementIndex+1]; i++){
			
			if(triangularMatrix->indices[i] != lastElementIndex){
				addMaxHeapItem(heap, triangularMatrix->indices[i], -triangularMatrix->data[i]*solution->data[solutionNnz]);
				
				if(debug){
					printf("Adding index with val: %lld, %lf\n", triangularMatrix->indices[i],-triangularMatrix->data[i]*solution->data[solutionNnz]);
				}

			}

		}

		if(node == NULL){
			node = remTopMaxHeapItem(heap);
		}

		if(debug)
			printf("Sol[%lld]= %lf\n",lastElementIndex, solution->data[solutionNnz]);

		solutionNnz++;
	}

	solution->indPtr[0] = 0;
	solution->indPtr[1] = solutionNnz;
	solution->nnz = solutionNnz;

	if(debug)
		printf("Finished solution onePhase.\n");

	freeHeap(heap);

	return solution;
}

CondMatrix* onePhaseAlgorithm(CondMatrix* triangularMatrix, CondMatrix* rhs, int solveType, int triangularForm){

	//Call the right form for the solve
	CondMatrix* solution;

	if(solveType == triangularForm){
		solution = onePhaseAlgorithmLForm(triangularMatrix, rhs);
	}
	else{
		solution = onePhaseAlgorithmUForm(triangularMatrix, rhs);
	}

	return solution;
}
