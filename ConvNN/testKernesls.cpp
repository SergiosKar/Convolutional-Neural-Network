//------------------------------------------------------------------------------
//
// kernel:  computeOutput
//
// Purpose: Compute the output of layer
//

//#include "kernelheader.h"

/*
 void compOut(

	 Layer*  layers, int index_layer)
{
	 const int n=0;// = get_global_size(0);
	 const int i=0;// = get_global_id(0);

	float t = 0;
	for ( int j = 0; j < layers[index_layer].nodes[i].numberOfWeights; j++)
		t += layers[index_layer].nodes[i].weights[j] * layers[index_layer - 1].nodes[i].weights[j];

	//layers[index_layer].nodes[i].output = sigmoid(t);
}
*/
/*
__kernel void backprophid(

__global Layer*  layers,int index_layer)
{
const int n = get_global_size(0);
const int i = get_global_id(0);



delta = 0;
for (int j = 0; j != numberOfNodes_NextLayer; j++)
delta += layers[index_layer+1].nodes[j].delta * layers[index_layer+1].nodes[j].weights[nodeNumber];
delta *= layers[index_layer].nodes[i].output*(1-layers[index_layer].nodes[i].output);


layers[index_layer].nodes[i].delta = delta;


for (int j = 0; j != numberOfWeights; j++)
layers[index_layer].nodes[i].weights[j] += a*delta*layers[index_layer-1].nodes[j].output;


}


__kernel void backpropout(__global Layer*  layers,int index_layer,__global float* targets )
{
const int n = get_global_size(0);
const int i = get_global_id(0);



layers[index_layer].nodes[i].delta = (targets[i] - layers[index_layer].nodes[i].output)*layers[index_layer].nodes[i].output*(1-layers[index_layer].nodes[i].output);;


for (int j = 0; j != numberOfWeights; j++)
layers[index_layer].nodes[i].weights[j] += a*layers[index_layer].nodes[i].delta*layers[index_layer-1].nodes[j].output;


}




*/