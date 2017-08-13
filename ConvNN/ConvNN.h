#pragma once



#ifndef THESIS_CONVNN_H
#define THESIS_CONVNN_H



#include "include.h"
#include "Layer.h"

class ConvNN {


public:



	void createConvNN(int numoffilters, int filtdim, int inpdim);
	void createFullyConnectedNN(std::vector<cl_int> &newNetVec, bool onlyFCNN, int inpdim);


	void train(std::vector<std::vector<float>> &inputs, std::vector<std::vector<float>> &targets, std::vector<std::vector<float>> &testinputs, std::vector<float> &testtargets, int epoches);
	void forward(std::vector<float> &input);

	void forwardFCNN(std::vector<float> &input);
	void trainFCNN(std::vector<std::vector<float>> &inputs, std::vector<std::vector<float>> &targets, std::vector<std::vector<float>> &testinputs, std::vector<float> &testtargets, int epoches);


	void trainingAccuracy(std::vector<std::vector<float>> &testinputs, std::vector<float> &testtargets, int num, bool onlyfcnn);

	void calculateError(std::vector<float> desiredout);

	float lr = 0.001;

	int softflag = 0;


private:
	

	///cnn
	cl::Kernel convKern;
	cl::Kernel  poolKern;
	cl::Kernel  reluKern;
	cl::Kernel  deltasKern;
	cl::Kernel  backpropcnnKern;


	cl::Buffer d_InputBuffer;
	cl::Buffer d_FiltersBuffer;
	cl::Buffer d_FeatMapBuffer;
	cl::Buffer d_PoolBuffer;
	cl::Buffer d_PoolIndexBuffer;
	cl::Buffer d_targetBuffer;
	cl::Buffer d_deltasBuffer;
	cl::Buffer d_rotatedImgBuffer;


	ConvLayer convLayer;
	int filterdim;
	int pooldim;
	int featmapdim;
	int inputdim;


	void computeConvolution();
	void pooling();
	void cnntoFcnn();

	///fcnn
	cl::Kernel compoutKern;
	cl::Kernel backpropoutKern;
	cl::Kernel bakckprophidKern;
	cl::Kernel cnnToFcnnKern;
	cl::Kernel rotate180Kern;
	cl::Kernel  softmaxKern;;


	std::vector<int> h_netVec;
	std::vector<Layer> h_layers;
	std::vector<cl::Buffer> d_layersBuffers;


	void computeOutputofNN();


	

	cl_int err;




};


#endif //THESIS_CONVNN_H
