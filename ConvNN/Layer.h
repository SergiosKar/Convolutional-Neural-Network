#pragma once
//
// Created by sergios on 24/12/2016.
//

#ifndef LAYER_H
#define LAYER_H

#include "include.h"



typedef struct Node {

	int numberOfWeights;
	float weights[1024];
	float output;
	float delta;

}Node;

typedef struct Filter {

	float weights[25];
	float bias;

}Filter;


typedef struct Layer {

	int numOfNodes;
	Node nodes[1024];

}Layer;

typedef struct ConvLayer {

	int numOfFilters;
	Filter filters[10];

}ConvLayer;






Layer* layer(int numberOfNodes, int numberOfWeights);
ConvLayer* convlayer(int numberOfFilters, int filtdim);



float inline sigmoid(float x)
{
	//To deal with overflow rounding errors and the such
	if (x < -100)
		return 0;
	if (x > 100)
		return 1;
	return 1 / (1 + exp(-x));
}

#endif //LAYER_H
