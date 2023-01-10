#include "../headers/algorithm.h"

struct ListArray{
	ListElement* array;
	struct ListArray* next;
};

typedef struct ListArray ListArray;

static void printList(ListElement* listNode){
	ListElement* currNode = listNode;
	int64_t i = 0;
	while(currNode != NULL){
		printf("Node %lld index %lld, data %lf\n", i, currNode->index, currNode->data);
		currNode = currNode->next;

	}
}

static void freeListArray(ListArray* list){
	ListArray* prevNode = list;

	while(prevNode != NULL){
		list = list->next;
		free(prevNode->array);
		free(prevNode);
		prevNode = list;
	}
}

static int compElementUForm(const void * a, const void * b){
	return ((ListElement*)b)->index - ((ListElement*)a)->index;
}

static int compElementLForm(const void * a, const void * b){
	return ((ListElement*)a)->index - ((ListElement*)b)->index;
}

static int addElementListArray(ListArray* list, ListElement* array){

	ListArray* newNode = (ListArray*) malloc(sizeof(ListArray));

	if(newNode == NULL){
		return 1;
	}

	newNode->array = array;
	newNode->next = NULL;
	list->next = newNode;

	return 0;
}

static ListElement* createSortedListElementFromRhs(int64_t* indicesArray, double* dataArray, int64_t nElements, bool isLForm){

	ListElement* arrayElements = (ListElement*) malloc(sizeof(ListElement)*nElements);
	int64_t i;
	
	for(i = 0; i < nElements; i++){
		arrayElements[i].index = indicesArray[i];
		arrayElements[i].data = dataArray[i];
	}

	//sort it
	if(isLForm){
		qsort(arrayElements, nElements, sizeof(ListElement), compElementLForm);
	}
	else{
		qsort(arrayElements, nElements, sizeof(ListElement), compElementUForm);		
	}
	
	//make the list with the sorted array

	for(i = 0; i < nElements - 1; i++){
		arrayElements[i].next = &arrayElements[i+1];
	}

	arrayElements[nElements - 1].next = NULL;

	return arrayElements;

}

static ListElement* createSortedListElement(int64_t* indicesArray, int64_t nElements, bool isLForm){

	ListElement* arrayElements = (ListElement*) malloc(sizeof(ListElement)*nElements);
	int64_t i;
	
	for(i = 0; i < nElements; i++){
		arrayElements[i].index = indicesArray[i];
		arrayElements[i].data = 0;
	}

	//sort it
	if(isLForm){
		qsort(arrayElements, nElements, sizeof(ListElement), compElementLForm);
	}
	else{
		qsort(arrayElements, nElements, sizeof(ListElement), compElementUForm);		
	}
	
	//make the list with the sorted array

	for(i = 0; i < nElements - 1; i++){
		arrayElements[i].next = &arrayElements[i+1];
	}

	arrayElements[nElements - 1].next = NULL;

	return arrayElements;

}

//need to have a new fct for the unsorted algorithm remembering which array were allocated and only free these arrays


void freeListElement(ListElement* listNode){
	ListElement* prevNode = listNode;

	while(prevNode != NULL){
		listNode = listNode->next;
		free(prevNode);
		prevNode = listNode;
	}
}

/*
CondMatrix* twoPhasesAlgorithmUForm(CondMatrix* triangularMatrix, CondMatrix* rhs){
	
	// First Phase : compute the "maximum" size of the solution

	//create a list
	bool debug = false;
	ListElement* listNz = (ListElement*) malloc(sizeof(ListElement));

	if(listNz == NULL){
		fprintf(stderr, "Error while allocating the list in the two phases algorithm.");
		return NULL;
	}

	//travel the element of rhs
	int64_t i, k, index, sol_nnz = 0;
	ListElement* nodePrevRhsAdded = listNz;
	ListElement* newNode,*prevNode,*nextNode;
	
	//adding the first element avoiding putting an if in the loop
	
	prevNode = listNz;
	
	if(rhs->nnz != 0){
		sol_nnz++;
		listNz->index = rhs->indices[rhs->nnz-1];
		listNz->data = rhs->data[rhs->nnz-1];
	}

	listNz->next = NULL;
	//adding all rhs in the list

	if(debug){
		printf("rhs->nnz = %lld\n", rhs->nnz);
	}

	for(i = rhs->nnz - 2; i >= 0; i --){

		newNode = (ListElement*) malloc(sizeof(ListElement));
		sol_nnz++;

		if(newNode == NULL){
			fprintf(stderr, "Error while allocating a new node for the list in the two phases algorithm.");
			freeListElement(listNz);
			return NULL;
		}

		newNode->index = rhs->indices[i];
		newNode->data = rhs->data[i];
		newNode->next = prevNode->next;

		prevNode->next = newNode;
		prevNode = newNode;
	}


	if(debug){
		printf("\nAdded the nz of the rhs\n");
		if(sol_nnz == 0){
			printf("ListNz is empty.\n");
		}
		else{
			printList(listNz);			
		}

	}
	
	ListElement* travelerNode = listNz;

	if(debug && travelerNode == NULL){

		printf("\nNever go in the loop.\n");
	}

	if(sol_nnz > 0){

		while(travelerNode != NULL){
		
			if(debug){
				printf("trav index: %lld, %lld\n", triangularMatrix->indPtr[travelerNode->index + 1] - 2, triangularMatrix->indPtr[travelerNode->index]);
			}

			for(k = triangularMatrix->indPtr[travelerNode->index + 1] - 2 ; k >= triangularMatrix->indPtr[travelerNode->index]; k--){
				
				index = triangularMatrix->indices[k];
				
				if(debug){
					printf("trav index: %lld\n",index);
				}

				newNode = travelerNode->next;
				prevNode  = travelerNode;

				//find the place where to put the new value (must be ordered)
				if(newNode !=NULL){
					while(newNode->index >= index){
						prevNode = newNode;
						newNode = newNode->next;
						if(newNode == NULL){
							break;
						}

					}
				}
				
				

				//adding only if the value does not exist in the list
				if (prevNode->index != index){//(newNode == NULL  (prevNode->index != index)){
					//printf("PrevNode:= %lld, newNode:= %lld\n", prevNode->index,index);
					sol_nnz++;
					//adding a new node on the list
					nextNode = newNode;
					newNode = (ListElement*) malloc(sizeof(ListElement));

					if(newNode == NULL){
						fprintf(stderr, "Error while allocating a new node for the list in the two phases algorithm.");
						freeListElement(listNz);
						return NULL;
					}

					newNode->index = index;
					newNode->data = 0;
					newNode->next = nextNode;
					prevNode->next = newNode;
				}
			}
			travelerNode = travelerNode->next;
		}
	}


	if(debug){
		printf("\nAdded all the nz of the sol, sol_nnz = %lld\n",sol_nnz);
		if(sol_nnz == 0){
			printf("ListNz is empty.\n");
		}
		else{
			printList(listNz);			
		}

	}
	

	CondMatrix* solution = allocateCondMatrix(sol_nnz, 1, rhs->nRow, true);
	if(solution == NULL){
		fprintf(stderr, "Error while allocating a compressed matrix for the solution in the two phases algorithm.");
		freeListElement(listNz);

	}

	int64_t m_index;
	nextNode = listNz;

	for(i = 0; i < sol_nnz; i++){
		index = nextNode->index;

		//compute final node value

		nextNode->data = nextNode->data/triangularMatrix->data[triangularMatrix->indPtr[index+1] - 1];


		if(debug){
			printf("index: %lld add done: div %lf / %lf\n", nextNode->index, nextNode->data,triangularMatrix->data[triangularMatrix->indPtr[index]]);
			printf("%lld:= %lf \n",nextNode->index, nextNode->data);
		}
		//read the element of col L
		for(k = triangularMatrix->indPtr[index + 1] - 2; k >= triangularMatrix->indPtr[index]; k--){

			//find the index in the list
			m_index = triangularMatrix->indices[k];
			travelerNode = nextNode;

			//traveling the list to find element with index == m_index
			while(travelerNode->index != m_index){

				travelerNode = travelerNode->next;
				//this check is avoidable if well implemented (left it for testes)
				if(travelerNode == NULL){
					break;
				}

			}
			//this check is avoidable if well implemented (left it for testes)
			if(travelerNode == NULL){
				printf("Shouldn't had a null travelerNode\n");
			}
			//printf("diaVal =  %lf, L_val = %lf\n", nextNode->data, triangularMatrix->data[k]);
			travelerNode->data = travelerNode->data - nextNode->data*triangularMatrix->data[k];

		}
		nextNode = nextNode->next;
	}

	if(debug){
		printf("Found Sol\n");
	}
	//from list to compmatrix
	solution->indPtr[0] = 0;
	nextNode = listNz;

	for(i = 0; i < sol_nnz; i++){

		solution->indices[i] = nextNode->index;
		solution->data[i] = nextNode->data;
		nextNode = nextNode->next;

	}

	freeListElement(listNz);

	return solution;
}


CondMatrix* twoPhasesAlgorithmLForm(CondMatrix* triangularMatrix, CondMatrix* rhs){

	// First Phase : compute the "maximum" size of the solution

	//create a list
	bool debug = false;
	ListElement* listNz = (ListElement*) malloc(sizeof(ListElement));

	if(listNz == NULL){
		fprintf(stderr, "Error while allocating the list in the two phases algorithm.");
		return NULL;
	}

	//travel the element of rhs
	int64_t i, k, index, sol_nnz = 0,start_;
	ListElement* nodePrevRhsAdded = listNz;
	ListElement* newNode,*prevNode,*nextNode;
	
	//adding the first element avoiding putting an if in the loop
	
	prevNode = listNz;
	
	if(rhs->nnz != 0){
		sol_nnz++;
		listNz->index = rhs->indices[0];
		listNz->data = rhs->data[0];
	}

	listNz->next = NULL;
	//adding all rhs in the list

	//printf("rhs->nnz = %lld\n", rhs->nnz);
	for(i = 1; i < rhs->nnz; i++){

		newNode = (ListElement*) malloc(sizeof(ListElement));
		sol_nnz++;

		if(newNode == NULL){
			fprintf(stderr, "Error while allocating a new node for the list in the two phases algorithm.");
			freeListElement(listNz);
			return NULL;
		}

		//printf("curr index: %lld\n", rhs->indices[i]);
		newNode->index = rhs->indices[i];
		newNode->data = rhs->data[i];
		newNode->next = prevNode->next;

		prevNode->next = newNode;
		prevNode = newNode;
	}

	if(debug){
		printf("\nAdded the nz of the rhs\n");
		if(sol_nnz == 0){
			printf("ListNz is empty.\n");
		}
		else{
			printList(listNz);			
		}
	}
	
	ListElement* travelerNode = listNz;
	
	if(sol_nnz > 0){
		while(travelerNode!= NULL){

			for(k = triangularMatrix->indPtr[travelerNode->index] + 1; k < triangularMatrix->indPtr[travelerNode->index + 1]; k++){
				
				index = triangularMatrix->indices[k];
				
				if(debug){
					printf("trav index: %lld\n",index);
				}

				newNode = travelerNode->next;
				prevNode  = travelerNode;

				//find the place where to put the new value (must be ordered)
				if(newNode !=NULL){
					while(newNode->index <= index){
						prevNode = newNode;
						newNode = newNode->next;
						if(newNode == NULL){
							break;
						}

					}
				}
				
				

				//adding only if the value does not exist in the list
				if (prevNode->index != index){//(newNode == NULL  (prevNode->index != index)){
					//printf("PrevNode:= %lld, newNode:= %lld\n", prevNode->index,index);
					sol_nnz++;
					//adding a new node on the list
					nextNode = newNode;
					newNode = (ListElement*) malloc(sizeof(ListElement));

					if(newNode == NULL){
						fprintf(stderr, "Error while allocating a new node for the list in the two phases algorithm.");
						freeListElement(listNz);
						return NULL;
					}

					newNode->index = index;
					newNode->data = 0;
					newNode->next = nextNode;
					prevNode->next = newNode;
				}
			}
			travelerNode = travelerNode->next;
		}
	}

	if(debug){
		printf("\nAdded all the nz of the sol, sol_nnz = %lld\n",sol_nnz);
		if(sol_nnz == 0){
			printf("ListNz is empty.\n");
		}
		else{
			printList(listNz);			
		}

	}
	

	CondMatrix* solution = allocateCondMatrix(sol_nnz, 1, rhs->nRow, true);
	if(solution == NULL){
		fprintf(stderr, "Error while allocating a compressed matrix for the solution in the two phases algorithm.");
		freeListElement(listNz);

	}

	int64_t m_index;
	nextNode = listNz;

	for(i = 0; i < sol_nnz; i++){
		index = nextNode->index;

		//compute final node value
		nextNode->data = nextNode->data/triangularMatrix->data[triangularMatrix->indPtr[index]];
		if(debug){
			printf("index: %lld add done: div %lf / %lf\n", nextNode->index, nextNode->data,triangularMatrix->data[triangularMatrix->indPtr[index]]);
			printf("%lld:= %lf \n",nextNode->index, nextNode->data);
		}
		//read the element of col L
		for(k = triangularMatrix->indPtr[index] + 1; k < triangularMatrix->indPtr[index + 1]; k++){

			//find the index in the list
			m_index = triangularMatrix->indices[k];
			travelerNode = nextNode;

			//traveling the list to find element with index == m_index
			while(travelerNode->index != m_index){

				travelerNode = travelerNode->next;
				//this check is avoidable if well implemented (left it for testes)
				if(travelerNode == NULL){
					break;
				}

			}
			//this check is avoidable if well implemented (left it for testes)
			if(travelerNode == NULL){
				printf("Shouldn't had a null travelerNode\n");
			}
			//printf("diaVal =  %lf, L_val = %lf\n", nextNode->data, triangularMatrix->data[k]);
			travelerNode->data = travelerNode->data - nextNode->data*triangularMatrix->data[k];

		}
		nextNode = nextNode->next;
	}
	if(debug){
		printf("Found Sol\n");
	}
	//from list to compmatrix
	solution->indPtr[0] = 0;
	nextNode = listNz;

	for(i = 0; i < sol_nnz; i++){

		solution->indices[i] = nextNode->index;
		solution->data[i] = nextNode->data;
		nextNode = nextNode->next;

	}

	freeListElement(listNz);

	return solution;
}
*/

CondMatrix* twoPhasesAlgorithmUFormSorted(CondMatrix* triangularMatrix, CondMatrix* rhs){
	
	// First Phase : compute the "maximum" size of the solution

	//create a list
	bool debug = false;
	ListElement* listNz = (ListElement*) malloc(sizeof(ListElement));

	if(listNz == NULL){
		fprintf(stderr, "Error while allocating the list in the two phases algorithm.");
		return NULL;
	}

	//travel the element of rhs
	int64_t i, k, index, sol_nnz = 0;
	ListElement* nodePrevRhsAdded = listNz;
	ListElement* newNode,*prevNode,*nextNode;
	
	//adding the first element of the rhs to a list (avoiding putting an if in the loop)
	
	prevNode = listNz;
	
	if(rhs->nnz != 0){
		sol_nnz++;
		listNz->index = rhs->indices[rhs->nnz-1];
		listNz->data = rhs->data[rhs->nnz-1];
	}

	listNz->next = NULL;
	//adding the remaining elements of the rhs to a list (the first one already in)

	if(debug){
		printf("rhs->nnz = %lld\n", rhs->nnz);
	}

	for(i = rhs->nnz - 2; i >= 0; i --){

		newNode = (ListElement*) malloc(sizeof(ListElement));
		sol_nnz++;

		if(newNode == NULL){
			fprintf(stderr, "Error while allocating a new node for the list in the two phases algorithm.");
			freeListElement(listNz);
			return NULL;
		}

		newNode->index = rhs->indices[i];
		newNode->data = rhs->data[i];
		newNode->next = prevNode->next;

		prevNode->next = newNode;
		prevNode = newNode;
	}


	if(debug){
		printf("\nAdded the nz of the rhs\n");
		if(sol_nnz == 0){
			printf("ListNz is empty.\n");
		}
		else{
			printList(listNz);			
		}

	}
	
	ListElement* travelerNode = listNz;

	if(debug && travelerNode == NULL){

		printf("\nNever go in the loop.\n");
	}

	if(sol_nnz > 0){

		while(travelerNode != NULL){
		
			if(debug){
				printf("trav index: %lld, %lld\n", triangularMatrix->indPtr[travelerNode->index + 1] - 2, triangularMatrix->indPtr[travelerNode->index]);
			}

			for(k = triangularMatrix->indPtr[travelerNode->index + 1] - 2 ; k >= triangularMatrix->indPtr[travelerNode->index]; k--){
				
				index = triangularMatrix->indices[k];
				
				if(debug){
					printf("trav index: %lld\n",index);
				}

				newNode = travelerNode->next;
				prevNode  = travelerNode;

				//find the place where to put the new value (must be ordered)
				if(newNode !=NULL){
					while(newNode->index >= index){
						prevNode = newNode;
						newNode = newNode->next;
						if(newNode == NULL){
							break;
						}

					}
				}
				
				

				//adding only if the value does not exist in the list
				if (prevNode->index != index){//(newNode == NULL  (prevNode->index != index)){
					//printf("PrevNode:= %lld, newNode:= %lld\n", prevNode->index,index);
					sol_nnz++;
					//adding a new node on the list
					nextNode = newNode;
					newNode = (ListElement*) malloc(sizeof(ListElement));

					if(newNode == NULL){
						fprintf(stderr, "Error while allocating a new node for the list in the two phases algorithm.");
						freeListElement(listNz);
						return NULL;
					}

					newNode->index = index;
					newNode->data = 0;
					newNode->next = nextNode;
					prevNode->next = newNode;
				}
			}
			travelerNode = travelerNode->next;
		}
	}


	if(debug){
		printf("\nAdded all the nz of the sol, sol_nnz = %lld\n",sol_nnz);
		if(sol_nnz == 0){
			printf("ListNz is empty.\n");
		}
		else{
			printList(listNz);			
		}

	}
	

	CondMatrix* solution = allocateCondMatrix(sol_nnz, 1, rhs->nRow, true);
	if(solution == NULL){
		fprintf(stderr, "Error while allocating a compressed matrix for the solution in the two phases algorithm.");
		freeListElement(listNz);

	}

	int64_t m_index;
	nextNode = listNz;

	for(i = 0; i < sol_nnz; i++){
		index = nextNode->index;

		//compute final node value

		nextNode->data = nextNode->data/triangularMatrix->data[triangularMatrix->indPtr[index+1] - 1];


		if(debug){
			printf("index: %lld add done: div %lf / %lf\n", nextNode->index, nextNode->data,triangularMatrix->data[triangularMatrix->indPtr[index]]);
			printf("%lld:= %lf \n",nextNode->index, nextNode->data);
		}
		//read the element of col L
		for(k = triangularMatrix->indPtr[index + 1] - 2; k >= triangularMatrix->indPtr[index]; k--){

			//find the index in the list
			m_index = triangularMatrix->indices[k];
			travelerNode = nextNode;

			//traveling the list to find element with index == m_index
			while(travelerNode->index != m_index){

				travelerNode = travelerNode->next;
				//this check is avoidable if well implemented (left it for testes)
				if(travelerNode == NULL){
					break;
				}

			}
			//this check is avoidable if well implemented (left it for testes)
			if(travelerNode == NULL){
				printf("Shouldn't had a null travelerNode\n");
			}
			//printf("diaVal =  %lf, L_val = %lf\n", nextNode->data, triangularMatrix->data[k]);
			travelerNode->data = travelerNode->data - nextNode->data*triangularMatrix->data[k];

		}
		nextNode = nextNode->next;
	}

	if(debug){
		printf("Found Sol\n");
	}
	//from list to compmatrix
	solution->indPtr[0] = 0;
	nextNode = listNz;

	for(i = 0; i < sol_nnz; i++){

		solution->indices[i] = nextNode->index;
		solution->data[i] = nextNode->data;
		nextNode = nextNode->next;

	}

	freeListElement(listNz);

	return solution;
}

CondMatrix* twoPhasesAlgorithmLFormSorted(CondMatrix* triangularMatrix, CondMatrix* rhs){

	// First Phase : compute the "maximum" size of the solution

	//create a list
	bool debug = false;
	ListElement* listNz = (ListElement*) malloc(sizeof(ListElement));

	if(listNz == NULL){
		fprintf(stderr, "Error while allocating the list in the two phases algorithm.");
		return NULL;
	}

	//travel the element of rhs
	int64_t i, k, index, sol_nnz = 0,start_;
	ListElement* nodePrevRhsAdded = listNz;
	ListElement* newNode,*prevNode,*nextNode;
	
	//adding the first element avoiding putting an if in the loop
	
	prevNode = listNz;
	
	if(rhs->nnz != 0){
		sol_nnz++;
		listNz->index = rhs->indices[0];
		listNz->data = rhs->data[0];
	}

	listNz->next = NULL;
	//adding all rhs in the list

	//printf("rhs->nnz = %lld\n", rhs->nnz);
	for(i = 1; i < rhs->nnz; i++){

		newNode = (ListElement*) malloc(sizeof(ListElement));
		sol_nnz++;

		if(newNode == NULL){
			fprintf(stderr, "Error while allocating a new node for the list in the two phases algorithm.");
			freeListElement(listNz);
			return NULL;
		}

		//printf("curr index: %lld\n", rhs->indices[i]);
		newNode->index = rhs->indices[i];
		newNode->data = rhs->data[i];
		newNode->next = prevNode->next;

		prevNode->next = newNode;
		prevNode = newNode;
	}

	if(debug){
		printf("\nAdded the nz of the rhs\n");
		if(sol_nnz == 0){
			printf("ListNz is empty.\n");
		}
		else{
			printList(listNz);			
		}
	}
	
	ListElement* travelerNode = listNz;
	
	if(sol_nnz > 0){
		while(travelerNode!= NULL){

			for(k = triangularMatrix->indPtr[travelerNode->index] + 1; k < triangularMatrix->indPtr[travelerNode->index + 1]; k++){
				
				index = triangularMatrix->indices[k];
				
				if(debug){
					printf("trav index: %lld\n",index);
				}

				newNode = travelerNode->next;
				prevNode  = travelerNode;

				//find the place where to put the new value (must be ordered)
				if(newNode !=NULL){
					while(newNode->index <= index){
						prevNode = newNode;
						newNode = newNode->next;
						if(newNode == NULL){
							break;
						}

					}
				}
				
				

				//adding only if the value does not exist in the list
				if (prevNode->index != index){//(newNode == NULL  (prevNode->index != index)){
					//printf("PrevNode:= %lld, newNode:= %lld\n", prevNode->index,index);
					sol_nnz++;
					//adding a new node on the list
					nextNode = newNode;
					newNode = (ListElement*) malloc(sizeof(ListElement));

					if(newNode == NULL){
						fprintf(stderr, "Error while allocating a new node for the list in the two phases algorithm.");
						freeListElement(listNz);
						return NULL;
					}

					newNode->index = index;
					newNode->data = 0;
					newNode->next = nextNode;
					prevNode->next = newNode;
				}
			}
			travelerNode = travelerNode->next;
		}
	}

	if(debug){
		printf("\nAdded all the nz of the sol, sol_nnz = %lld\n",sol_nnz);
		if(sol_nnz == 0){
			printf("ListNz is empty.\n");
		}
		else{
			printList(listNz);			
		}

	}
	

	CondMatrix* solution = allocateCondMatrix(sol_nnz, 1, rhs->nRow, true);
	if(solution == NULL){
		fprintf(stderr, "Error while allocating a compressed matrix for the solution in the two phases algorithm.");
		freeListElement(listNz);

	}

	int64_t m_index;
	nextNode = listNz;

	for(i = 0; i < sol_nnz; i++){
		index = nextNode->index;

		//compute final node value
		nextNode->data = nextNode->data/triangularMatrix->data[triangularMatrix->indPtr[index]];
		if(debug){
			printf("index: %lld add done: div %lf / %lf\n", nextNode->index, nextNode->data,triangularMatrix->data[triangularMatrix->indPtr[index]]);
			printf("%lld:= %lf \n",nextNode->index, nextNode->data);
		}
		//read the element of col L
		for(k = triangularMatrix->indPtr[index] + 1; k < triangularMatrix->indPtr[index + 1]; k++){

			//find the index in the list
			m_index = triangularMatrix->indices[k];
			travelerNode = nextNode;

			//traveling the list to find element with index == m_index
			while(travelerNode->index != m_index){

				travelerNode = travelerNode->next;
				//this check is avoidable if well implemented (left it for testes)
				if(travelerNode == NULL){
					break;
				}

			}
			//this check is avoidable if well implemented (left it for testes)
			if(travelerNode == NULL){
				printf("Shouldn't had a null travelerNode\n");
			}
			//printf("diaVal =  %lf, L_val = %lf\n", nextNode->data, triangularMatrix->data[k]);
			travelerNode->data = travelerNode->data - nextNode->data*triangularMatrix->data[k];

		}
		nextNode = nextNode->next;
	}
	if(debug){
		printf("Found Sol\n");
	}
	//from list to compmatrix
	solution->indPtr[0] = 0;
	nextNode = listNz;

	for(i = 0; i < sol_nnz; i++){

		solution->indices[i] = nextNode->index;
		solution->data[i] = nextNode->data;
		nextNode = nextNode->next;

	}

	freeListElement(listNz);

	return solution;
}


CondMatrix* twoPhasesAlgorithmUForm(CondMatrix* triangularMatrix, CondMatrix* rhs){
	
	// First Phase : compute the "maximum" size of the solution

	//create a list
	bool debug = false;

	//travel the element of rhs
	int64_t i, k, index, sol_nnz = 0, start, end;

	ListElement* newNode,*prevNode,*nextNode, *newElements;

	//allocate the array with the rhs
	ListElement* listNz = createSortedListElementFromRhs(rhs->indices, rhs->data, rhs->nnz, false);

	//should push listNz in an array to free
	ListArray* listToFreeArray = (ListArray*) malloc(sizeof(ListArray));

	if(listToFreeArray == NULL){
		free(listNz);
		fprintf(stderr,"Failed allocation of the free array in the 2phases algorithm.\n");
		return NULL;
	}

	listToFreeArray->array = listNz;
	listToFreeArray->next = NULL;

	sol_nnz = rhs->nnz;

	if(debug){
		printf("rhs->nnz = %lld\n", rhs->nnz);

		printf("\nAdded the nz of the rhs\n");

		if(sol_nnz == 0){
			printf("ListNz is empty.\n");
		}
		else{
			printList(listNz);			
		}

	}
	
	ListElement* travelerNode = listNz;

	if(debug && travelerNode == NULL){

		printf("\nNever go in the loop.\n");
	}

	if(sol_nnz > 0){

		//travel the element added in the list (at first it is the nnz of the rhs)
		while(travelerNode != NULL){
		
			if(debug){
				printf("trav index: %lld, %lld\n", triangularMatrix->indPtr[travelerNode->index + 1] - 2, triangularMatrix->indPtr[travelerNode->index]);
			}

			end = triangularMatrix->indPtr[travelerNode->index + 1];
			start = triangularMatrix->indPtr[travelerNode->index];

			newElements  = createSortedListElement(&triangularMatrix->indices[start], end-start, false);
			
			if(addElementListArray(listToFreeArray, newElements)){
				fprintf(stderr, "Failed to add new array to free in listToFreeArray");
				freeListArray(listToFreeArray);
				return NULL;
			}

			for(k = 1 ; k < end-start; k ++){
				
				index = newElements[k].index;
				
				if(debug){
					printf("trav index: %lld\n",index);
				}

				newNode = travelerNode->next;
				prevNode  = travelerNode;

				//find the place where to put the new value (must be ordered)
				if(newNode !=NULL){
					while(newNode->index >= index){
						prevNode = newNode;
						newNode = newNode->next;
						if(newNode == NULL){
							break;
						}

					}
				}
				
				

				//adding only if the value does not exist in the list
				if (prevNode->index != index){
					//(newNode == NULL  (prevNode->index != index)){
					//printf("PrevNode:= %lld, newNode:= %lld\n", prevNode->index,index);
					
					sol_nnz++;
					//adding a new node on the list
					newElements[k].next = newNode;
					prevNode->next = &newElements[k];
				}
			}
			travelerNode = travelerNode->next;
		}
	}


	if(debug){
		printf("\nAdded all the nz of the sol, sol_nnz = %lld\n",sol_nnz);
		if(sol_nnz == 0){
			printf("ListNz is empty.\n");
		}
		else{
			printList(listNz);			
		}

	}
	

	CondMatrix* solution = allocateCondMatrix(sol_nnz, 1, rhs->nRow, true);
	if(solution == NULL){
		fprintf(stderr, "Error while allocating a compressed matrix for the solution in the two phases algorithm.");
		freeListElement(listNz);

	}

	int64_t m_index, good_index;
	nextNode = listNz;

	for(i = 0; i < sol_nnz; i++){
		index = nextNode->index;

		//compute final node value

		//need to find the element in the matrix with the good index
		for(k = triangularMatrix->indPtr[index]; k < triangularMatrix->indPtr[index+1]; k++){
			
			if(triangularMatrix->indices[k] == index){
				good_index = k;
			}

		}

		nextNode->data = nextNode->data/triangularMatrix->data[good_index];


		if(debug){
			printf("index: %lld add done: div %lf / %lf\n", nextNode->index, nextNode->data,triangularMatrix->data[triangularMatrix->indPtr[index]]);
			printf("%lld:= %lf \n",nextNode->index, nextNode->data);
		}

		//read the element of col L
		for(k = triangularMatrix->indPtr[index]; k < good_index; k++){

			//find the index in the list
			m_index = triangularMatrix->indices[k];
			travelerNode = nextNode;

			//traveling the list to find element with index == m_index
			while(travelerNode->index != m_index){

				travelerNode = travelerNode->next;
				//this check is avoidable if well implemented (left it for testes)
				if(travelerNode == NULL){
					break;
				}

			}
			//this check is avoidable if well implemented (left it for testes)
			if(travelerNode == NULL){
				printf("Shouldn't had a null travelerNode\n");
			}
			//printf("diaVal =  %lf, L_val = %lf\n", nextNode->data, triangularMatrix->data[k]);
			travelerNode->data = travelerNode->data - nextNode->data*triangularMatrix->data[k];
		}

		for(k = good_index + 1; k < triangularMatrix->indPtr[index+1]; k++){

			//find the index in the list
			m_index = triangularMatrix->indices[k];
			travelerNode = nextNode;

			//traveling the list to find element with index == m_index
			while(travelerNode->index != m_index){

				travelerNode = travelerNode->next;
				//this check is avoidable if well implemented (left it for testes)
				if(travelerNode == NULL){
					break;
				}

			}
			//this check is avoidable if well implemented (left it for testes)
			if(travelerNode == NULL){
				printf("Shouldn't had a null travelerNode\n");
			}
			//printf("diaVal =  %lf, L_val = %lf\n", nextNode->data, triangularMatrix->data[k]);
			travelerNode->data = travelerNode->data - nextNode->data*triangularMatrix->data[k];
		}

		nextNode = nextNode->next;
	}

	if(debug){
		printf("Found Sol\n");
	}
	//from list to compmatrix
	solution->indPtr[0] = 0;
	nextNode = listNz;

	for(i = 0; i < sol_nnz; i++){

		solution->indices[i] = nextNode->index;
		solution->data[i] = nextNode->data;
		nextNode = nextNode->next;

	}

	if(sol_nnz > 0){
		freeListArray(listToFreeArray);
	}
	

	if(debug){
		printf("After free the listArray\n");
	}

	return solution;
}

CondMatrix* twoPhasesAlgorithmLForm(CondMatrix* triangularMatrix, CondMatrix* rhs){
	
	// First Phase : compute the "maximum" size of the solution

	//create a list
	bool debug = false;

	//travel the element of rhs
	int64_t i, k, index, sol_nnz = 0, start, end;

	ListElement* newNode,*prevNode,*nextNode, *newElements;

	//allocate the array with the rhs
	ListElement* listNz = createSortedListElementFromRhs(rhs->indices, rhs->data, rhs->nnz, true);

	//should push listNz in an array to free
	ListArray* listToFreeArray = (ListArray*) malloc(sizeof(ListArray));

	if(listToFreeArray == NULL){
		free(listNz);
		fprintf(stderr,"Failed allocation of the free array in the 2phases algorithm.\n");
		return NULL;
	}

	listToFreeArray->array = listNz;
	listToFreeArray->next = NULL;
	
	sol_nnz = rhs->nnz;
	
	if(debug){
		printf("rhs->nnz = %lld\n", rhs->nnz);

		printf("\nAdded the nz of the rhs\n");

		if(sol_nnz == 0){
			printf("ListNz is empty.\n");
		}
		else{
			printList(listNz);			
		}

	}
	
	ListElement* travelerNode = listNz;

	if(debug && travelerNode == NULL){

		printf("\nNever go in the loop.\n");
	}

	if(sol_nnz > 0){

		//travel the element added in the list (at first it is the nnz of the rhs)
		while(travelerNode != NULL){
		
			if(debug){
				printf("trav index: %lld, %lld\n", triangularMatrix->indPtr[travelerNode->index + 1] - 2, triangularMatrix->indPtr[travelerNode->index]);
			}

			end = triangularMatrix->indPtr[travelerNode->index + 1];
			start = triangularMatrix->indPtr[travelerNode->index];

			newElements  = createSortedListElement(&triangularMatrix->indices[start], end-start, true);
			
			if(addElementListArray(listToFreeArray, newElements)){
				fprintf(stderr, "Failed to add new array to free in listToFreeArray");
				freeListArray(listToFreeArray);
				return NULL;
			}

			for(k = 1 ; k < end-start; k ++){
				
				index = newElements[k].index;
				
				if(debug){
					printf("trav index: %lld\n",index);
				}

				newNode = travelerNode->next;
				prevNode  = travelerNode;

				//find the place where to put the new value (must be ordered)
				if(newNode != NULL){
					while(newNode->index <= index){
						prevNode = newNode;
						newNode = newNode->next;
						if(newNode == NULL){
							break;
						}

					}
				}
				
				

				//adding only if the value does not exist in the list
				if (prevNode->index != index){
					//(newNode == NULL  (prevNode->index != index)){
					//printf("PrevNode:= %lld, newNode:= %lld\n", prevNode->index,index);
					
					sol_nnz++;
					//adding a new node on the list
					newElements[k].next = newNode;
					prevNode->next = &newElements[k];
				}
			}
			travelerNode = travelerNode->next;
		}
	}


	if(debug){
		printf("\nAdded all the nz of the sol, sol_nnz = %lld\n",sol_nnz);
		if(sol_nnz == 0){
			printf("ListNz is empty.\n");
		}
		else{
			printList(listNz);			
		}

	}
	

	CondMatrix* solution = allocateCondMatrix(sol_nnz, 1, rhs->nRow, true);
	if(solution == NULL){
		fprintf(stderr, "Error while allocating a compressed matrix for the solution in the two phases algorithm.");
		freeListElement(listNz);

	}

	int64_t m_index, good_index;
	nextNode = listNz;

	for(i = 0; i < sol_nnz; i++){
		index = nextNode->index;

		//compute final node value

		//need to find the element in the matrix with the good index
		for(k = triangularMatrix->indPtr[index]; k < triangularMatrix->indPtr[index+1]; k++){
			
			if(triangularMatrix->indices[k] == index){
				good_index = k;
			}

		}

		nextNode->data = nextNode->data/triangularMatrix->data[good_index];


		if(debug){
			printf("index: %lld add done: div %lf / %lf\n", nextNode->index, nextNode->data,triangularMatrix->data[triangularMatrix->indPtr[index]]);
			printf("%lld:= %lf \n", nextNode->index, nextNode->data);
		}

		//read the element of col L
		for(k = triangularMatrix->indPtr[index]; k < good_index; k++){

			//find the index in the list
			m_index = triangularMatrix->indices[k];
			travelerNode = nextNode;

			//traveling the list to find element with index == m_index
			while(travelerNode->index != m_index){

				travelerNode = travelerNode->next;
				//this check is avoidable if well implemented (left it for testes)
				if(travelerNode == NULL){
					break;
				}

			}
			//this check is avoidable if well implemented (left it for testes)
			if(travelerNode == NULL){
				printf("Shouldn't had a null travelerNode\n");
			}
			//printf("diaVal =  %lf, L_val = %lf\n", nextNode->data, triangularMatrix->data[k]);
			travelerNode->data = travelerNode->data - nextNode->data*triangularMatrix->data[k];
		}

		for(k = good_index + 1; k < triangularMatrix->indPtr[index+1]; k++){

			//find the index in the list
			m_index = triangularMatrix->indices[k];
			travelerNode = nextNode;

			//traveling the list to find element with index == m_index
			while(travelerNode->index != m_index){

				travelerNode = travelerNode->next;
				//this check is avoidable if well implemented (left it for testes)
				if(travelerNode == NULL){
					break;
				}

			}
			//this check is avoidable if well implemented (left it for testes)
			if(travelerNode == NULL){
				printf("Shouldn't had a null travelerNode\n");
			}
			//printf("diaVal =  %lf, L_val = %lf\n", nextNode->data, triangularMatrix->data[k]);
			travelerNode->data = travelerNode->data - nextNode->data*triangularMatrix->data[k];
		}

		nextNode = nextNode->next;
	}

	if(debug){
		printf("Found Sol\n");
	}
	//from list to compmatrix
	solution->indPtr[0] = 0;
	nextNode = listNz;

	for(i = 0; i < sol_nnz; i++){

		solution->indices[i] = nextNode->index;
		solution->data[i] = nextNode->data;
		nextNode = nextNode->next;

	}

	if(sol_nnz > 0){
		freeListArray(listToFreeArray);
	}
	if(debug){
		printf("After free the listArray\n");
	}
	return solution;
}

CondMatrix* twoPhasesAlgorithm(CondMatrix* triangularMatrix, CondMatrix* rhs, int solveType, int triangularForm){

	//Call the right form for the solve
	CondMatrix* solution;

	if(solveType == triangularForm){

		if(triangularMatrix->indicesSorted && rhs->indicesSorted){
			//indices sorted so the algorithm can be faster
			solution = twoPhasesAlgorithmLFormSorted(triangularMatrix, rhs);
		}
		else{
			//slower because indices are not sorted
			solution = twoPhasesAlgorithmLForm(triangularMatrix, rhs);
		}

	}
	else{
		if(triangularMatrix->indicesSorted && rhs->indicesSorted){
			//indices sorted so the algorithm can be faster
			solution = twoPhasesAlgorithmUFormSorted(triangularMatrix, rhs);
		}
		else{
			//slower because indices are not sorted
			solution = twoPhasesAlgorithmUForm(triangularMatrix, rhs);
		}
	}

	return solution;
}
