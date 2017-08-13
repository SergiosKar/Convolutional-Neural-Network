

#include "Layer.h"


float getRandom(float lowerbound, float upperbound)
{
	float f = (float)rand() / RAND_MAX;
	f = lowerbound + f * (upperbound - lowerbound);
	return f;
}




///Creates a hidden or output layer
Layer* layer(int numberOfNodes, int numberOfWeights)
{
	Layer* hidlayer = new Layer();
	hidlayer->numOfNodes = numberOfNodes;

	for (int i = 0; i != numberOfNodes; ++i)
	{
		Node node;
		node.numberOfWeights = numberOfWeights;


		node.output = 0.0;
		for (int j = 0; j < numberOfWeights; j++)
			node.weights[j] =  getRandom(-0.2, 0.2); 

		hidlayer->nodes[i] = node;
	}
	return hidlayer;
}

ConvLayer* convlayer(int numberOfFilters, int filtdim)
{
	ConvLayer* layer = new ConvLayer();
	layer->numOfFilters = numberOfFilters;

	for (int i = 0; i != numberOfFilters; ++i)
	{
		Filter filter;
		for (int j = 0; j != filtdim; ++j) {

			for (int k = 0; k != filtdim; ++k)
				filter.weights[k*filtdim + j] = getRandom(-0.2, 0.2);


		}
		filter.bias = 0.1;//getRandom(-0.3, 0.3);
		

		layer->filters[i] = filter;
	}
	return layer;
}