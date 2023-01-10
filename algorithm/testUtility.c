#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <stdbool.h>
#include <time.h>
#include "headers/dynamic_array.h"

int main(int argc, char *argv[]) {
	
	DynamicArray* array = allocDynamicArray(20);

	if(array == NULL){
		fprintf(stderr, "Error while allocating the DynamicArray\n");
		return 1;
	}

	int64_t i;
	ItemArray item;

	for(i = 0; i < 20; i++){
		item.index = i;
		item.data = 1;
		DynamicArraySetAt(array, i, item);
	}

	for(i = 0; i < 20; i++){
		item = DynamicArrayGetAt(array, i);
		printf("item %lld: %lld, %lf\n", i, item.index, item.data);
	}

	printf("capa: %lld\n", array->capacity);

	for(i = 20; i < 40; i++){
		item.index = i;
		item.data = 2;
		DynamicArraySetAt(array, i, item);
	}

	for(i = 0; i < 40; i++){
		item = DynamicArrayGetAt(array, i);
		printf("item %lld: %lld, %lf\n", i, item.index, item.data);
	}
	
	printf("capa: %lld\n", array->capacity);
	item.index = 40;
	item.data = 3;

	DynamicArraySetAt(array, 40, item);
	printf("capa: %lld\n", array->capacity);
	return 0;

}