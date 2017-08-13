
#define actflag 0// 0:sigmoid , 1:tanh ,2:relu

typedef struct Node {

	int numberOfWeights;
	float weights[1200];
	float output;
	float delta;

}Node;

typedef struct Filter {

	float weights[49];
	float bias;

}Filter;


typedef struct Layer {

	int numOfNodes;
	Node nodes[1200];

}Layer;

typedef struct ConvLayer {

	int numOfFilters;
	Filter filters[10];

}ConvLayer;


float inline sigmoid(float x)
{
	//To deal with overflow rounding errors and the such
	if (x < -100)
		return 0;
	if (x > 100)
		return 1;
	return 1 / (1 + exp(-x));
}

float inline devsigmoid(float x)
{
	
	return (x*(1-x));
}

float inline mtanh(float x)
{
	
	return tanh(x);
}
float inline devtanh(float x)
{
	
	return (1-x*x);
}

float inline relu(float x)
{
	if(x<0)
		return 0;
	else
		return x;
	
}
float inline devrelu(float x)
{
	if(x<0)
		return 0;
	else
		return 1;
	
}








