

#include "ConvNN.h"







///Create the neural net as a vector of layers
void ConvNN::createConvNN(int numoffilters, int filtdim,int inpdim)


{
	///Create the input layer

	convLayer = *convlayer(numoffilters, filtdim);


	///Create memory buffers
	inputdim = inpdim;
	filterdim = filtdim;
	featmapdim = inputdim - filterdim + 1;
	pooldim = ((featmapdim - 2) / 2) + 1;

	d_InputBuffer = cl::Buffer(OpenCL::clcontext, CL_MEM_READ_WRITE, sizeof(float)*inputdim*inputdim);


	d_FiltersBuffer = cl::Buffer(OpenCL::clcontext, CL_MEM_READ_WRITE, sizeof(Filter)*convLayer.numOfFilters);
	(OpenCL::clqueue).enqueueWriteBuffer(d_FiltersBuffer, CL_TRUE,0, sizeof(Filter)*convLayer.numOfFilters, convLayer.filters);


	d_FeatMapBuffer = cl::Buffer(OpenCL::clcontext, CL_MEM_READ_WRITE, sizeof(float)*featmapdim*featmapdim*convLayer.numOfFilters);

	d_PoolBuffer = cl::Buffer(OpenCL::clcontext, CL_MEM_READ_WRITE, sizeof(float)*pooldim*pooldim*convLayer.numOfFilters);


	d_PoolIndexBuffer=cl::Buffer(OpenCL::clcontext, CL_MEM_READ_WRITE, sizeof(float)*pooldim*pooldim*convLayer.numOfFilters);

	d_deltasBuffer = cl::Buffer(OpenCL::clcontext, CL_MEM_READ_WRITE, sizeof(float)*featmapdim*featmapdim*convLayer.numOfFilters);
	std::vector<float> del(featmapdim*featmapdim*convLayer.numOfFilters, 0.0);
	OpenCL::clqueue.enqueueWriteBuffer(d_deltasBuffer, CL_TRUE, 0, sizeof(float)*featmapdim*featmapdim*convLayer.numOfFilters,del.data());
	
	d_rotatedImgBuffer= cl::Buffer(OpenCL::clcontext, CL_MEM_READ_WRITE, sizeof(float)*featmapdim*featmapdim);
		///kernels

	convKern = cl::Kernel(OpenCL::clprogram, "convolve");
	poolKern = cl::Kernel(OpenCL::clprogram, "pooling");
	deltasKern = cl::Kernel(OpenCL::clprogram, "deltas");
	backpropcnnKern = cl::Kernel(OpenCL::clprogram, "backpropcnn");
	cnnToFcnnKern = cl::Kernel(OpenCL::clprogram, "cnntoFcnn");
	rotate180Kern = cl::Kernel(OpenCL::clprogram, "rotatemat");
	softmaxKern= cl::Kernel(OpenCL::clprogram, "softmax");

	




}


//Create the neural net as a vector of layers
void ConvNN::createFullyConnectedNN(std::vector<cl_int> &newNetVec, bool onlyFCNN, int inpdim)
{
	///Create the input layer
	h_netVec = newNetVec;
	Layer *inputLayer = layer(h_netVec[0], 0);
	h_layers.push_back(*inputLayer);

	///Create the other layers
	for (unsigned int i = 1; i <h_netVec.size(); i++)
	{
		Layer *hidlayer = layer(h_netVec[i], h_netVec[i - 1]);
		h_layers.push_back(*hidlayer);

	}


	///Create memory buffers

	///Create memory buffers
	if (onlyFCNN == 1) {
		d_InputBuffer = cl::Buffer(OpenCL::clcontext, CL_MEM_READ_WRITE, sizeof(float)*inpdim*inpdim);
		inputdim = inpdim;
		cnnToFcnnKern = cl::Kernel(OpenCL::clprogram, "cnntoFcnn");
		softmaxKern = cl::Kernel(OpenCL::clprogram, "softmax");

	}


	cl::Buffer  tempbuf;

	tempbuf = cl::Buffer(OpenCL::clcontext, CL_MEM_READ_WRITE, sizeof(Node)*h_layers[0].numOfNodes);
	(OpenCL::clqueue).enqueueWriteBuffer(tempbuf,CL_TRUE,0,sizeof(Node)*h_layers[0].numOfNodes,h_layers[0].nodes);
	d_layersBuffers.push_back(tempbuf);

	for (int i = 1; i<h_layers.size(); i++) {
		tempbuf = cl::Buffer(OpenCL::clcontext, CL_MEM_READ_WRITE, sizeof(Node)*h_layers[i].numOfNodes);
		(OpenCL::clqueue).enqueueWriteBuffer(tempbuf, CL_TRUE,0, sizeof(Node)*h_layers[i].numOfNodes, h_layers[i].nodes);
		d_layersBuffers.push_back(tempbuf);

	}



	d_targetBuffer = cl::Buffer(OpenCL::clcontext, CL_MEM_READ_WRITE, sizeof(float)*h_netVec.back());
	///kernels

	compoutKern = cl::Kernel(OpenCL::clprogram, "compout");
	backpropoutKern = cl::Kernel(OpenCL::clprogram, "backpropout");
	bakckprophidKern = cl::Kernel(OpenCL::clprogram, "backprophid");

	


}


void ConvNN::forward(std::vector<float> &input) {

	


	(OpenCL::clqueue).enqueueWriteBuffer(d_InputBuffer, CL_TRUE,0, sizeof(float)*inputdim*inputdim, input.data());



	//std::cout << "convolve" << std::endl;
	computeConvolution();
	
	//std::cout << "pool" << std::endl;
	pooling();
	//std::cout << "cnntofcnn" << std::endl;
	cnntoFcnn();
	//std::cout << "fcnn" << std::endl;
	computeOutputofNN();




}

void ConvNN::forwardFCNN(std::vector<float> &input) {



	(OpenCL::clqueue).enqueueWriteBuffer(d_InputBuffer, CL_TRUE, 0, sizeof(float)*inputdim*inputdim, input.data());

	cnnToFcnnKern.setArg(0, d_InputBuffer);
	cnnToFcnnKern.setArg(1, d_layersBuffers[0]);
	cnnToFcnnKern.setArg(2, inputdim);
	cnnToFcnnKern.setArg(3, 0);
	err = (OpenCL::clqueue).enqueueNDRangeKernel(cnnToFcnnKern, cl::NullRange,
		cl::NDRange(inputdim, inputdim, 1),
		cl::NullRange);

	

	computeOutputofNN();
}





void ConvNN::train(std::vector<std::vector<float>> &inputs, std::vector<std::vector<float>> &targets, std::vector<std::vector<float>> &testinputs, std::vector<float> &testtargets ,int epoches) {


	


	int i = 0;

	for (int e = 0; e<epoches; e++) {
		
		
		if (e < inputs.size()) { i = e; }
		else if ( e%inputs.size()==0) { i = 0; }
		else if (e > inputs.size()) { i++; }

		(OpenCL::clqueue).enqueueWriteBuffer(d_targetBuffer, CL_TRUE,0, sizeof(float)*h_netVec.back(), targets[i].data());

		///input
		


		///forward
		(OpenCL::clqueue).enqueueWriteBuffer(d_InputBuffer, CL_TRUE, 0, sizeof(float)*inputdim*inputdim, inputs[i].data());

		//std::cout << "convolve" << std::endl;
		computeConvolution();
		
		//std::cout << "pool" << std::endl;
		pooling();
		//std::cout << "cnntofcnn" << std::endl;
		cnntoFcnn();
		//std::cout << "fcnn" << std::endl;
		computeOutputofNN();
		
		if (e % 1000 == 0) {
			std::cout << e << std::endl;
		}



		//calculateError(targets[i]);

		///backward
		//std::cout << "back" << std::endl;
		for (int l = h_layers.size() - 1; l>0; l--) {
			
			

			if (l == h_layers.size() - 1) {
			
				backpropoutKern.setArg(0, d_layersBuffers[l]);
				backpropoutKern.setArg(1, d_layersBuffers[l-1]);
				backpropoutKern.setArg(2, d_targetBuffer);
				backpropoutKern.setArg(3, lr);
				backpropoutKern.setArg(4, softflag);

				err = (OpenCL::clqueue).enqueueNDRangeKernel(backpropoutKern, cl::NullRange,
					cl::NDRange(h_netVec[l]),
					cl::NullRange);



			
			}

			else {

				
				bakckprophidKern.setArg(0, d_layersBuffers[l]);
				bakckprophidKern.setArg(1, d_layersBuffers[l - 1]);
				bakckprophidKern.setArg(2, d_layersBuffers[l + 1]);
				bakckprophidKern.setArg(3,h_netVec[l+1]);
				bakckprophidKern.setArg(4, lr);
				err = (OpenCL::clqueue).enqueueNDRangeKernel(bakckprophidKern, cl::NullRange,
					cl::NDRange(h_netVec[l]),
					cl::NullRange);

				

				
			}

			
		}

	

		deltasKern.setArg(0, d_layersBuffers[0]);
		deltasKern.setArg(1, d_layersBuffers[1]);
		deltasKern.setArg(2, d_deltasBuffer);
		deltasKern.setArg(3, d_PoolIndexBuffer);
		deltasKern.setArg(4, featmapdim);
		deltasKern.setArg(5, h_netVec[1]);
		deltasKern.setArg(6, pooldim);
		err = (OpenCL::clqueue).enqueueNDRangeKernel(deltasKern, cl::NullRange,
			cl::NDRange(pooldim, pooldim,convLayer.numOfFilters),
			cl::NullRange);

		
		
		
		//rotate image for backprop
		
		rotate180Kern.setArg(0, d_deltasBuffer);
		rotate180Kern.setArg(1, d_rotatedImgBuffer);
		rotate180Kern.setArg(2, featmapdim);
		
		err = (OpenCL::clqueue).enqueueNDRangeKernel(rotate180Kern, cl::NullRange,
			cl::NDRange(featmapdim, featmapdim),
			cl::NullRange);

		

		
		
	

		backpropcnnKern.setArg(0, d_FeatMapBuffer);
		backpropcnnKern.setArg(1, d_rotatedImgBuffer);//deltas
		backpropcnnKern.setArg(2, d_FiltersBuffer);
		backpropcnnKern.setArg(3, featmapdim);
		backpropcnnKern.setArg(4, inputdim);
		backpropcnnKern.setArg(5, filterdim);
		backpropcnnKern.setArg(6, lr);
		backpropcnnKern.setArg(7, d_InputBuffer);
		err = (OpenCL::clqueue).enqueueNDRangeKernel(backpropcnnKern, cl::NullRange,
			cl::NDRange(filterdim, filterdim,convLayer.numOfFilters),
			cl::NullRange);

		

		if (e % 50000 == 0 && e!=0) {
			trainingAccuracy(testinputs, testtargets, 2000,0);
				
		}

		
		
	}


	
}

void ConvNN::trainFCNN(std::vector<std::vector<float>> &inputs, std::vector<std::vector<float>> &targets, std::vector<std::vector<float>> &testinputs, std::vector<float> &testtargets, int epoches) {


	int i = 0;

	for (int e = 0; e < epoches; e++) {


		if (e % 1000 == 0)
			std::cout << e << std::endl;

		if (e < inputs.size()) { i = e; }
		else if (e%inputs.size() == 0) { i = 0; }
		else if (e > inputs.size()) { i++; }

		(OpenCL::clqueue).enqueueWriteBuffer(d_targetBuffer, CL_TRUE, 0, sizeof(float)*h_netVec.back(), targets[i].data());

		///input

		///forward
		(OpenCL::clqueue).enqueueWriteBuffer(d_InputBuffer, CL_TRUE, 0, sizeof(float)*inputdim*inputdim, inputs[i].data());


		cnnToFcnnKern.setArg(0, d_InputBuffer);
		cnnToFcnnKern.setArg(1, d_layersBuffers[0]);
		cnnToFcnnKern.setArg(2, inputdim);
		cnnToFcnnKern.setArg(3, 0);
		err = (OpenCL::clqueue).enqueueNDRangeKernel(cnnToFcnnKern, cl::NullRange,
			cl::NDRange(inputdim, inputdim, 1),
			cl::NullRange);




		

		computeOutputofNN();



		///backward
		
		for (int l = h_layers.size() - 1; l>0; l--) {



			if (l == h_layers.size() - 1) {

				backpropoutKern.setArg(0, d_layersBuffers[l]);
				backpropoutKern.setArg(1, d_layersBuffers[l - 1]);
				backpropoutKern.setArg(2, d_targetBuffer);
				backpropoutKern.setArg(3, lr);
				backpropoutKern.setArg(4, softflag);

				err = (OpenCL::clqueue).enqueueNDRangeKernel(backpropoutKern, cl::NullRange,
					cl::NDRange(h_netVec[l]),
					cl::NullRange);


			}

			else {


				bakckprophidKern.setArg(0, d_layersBuffers[l]);
				bakckprophidKern.setArg(1, d_layersBuffers[l - 1]);
				bakckprophidKern.setArg(2, d_layersBuffers[l + 1]);
				bakckprophidKern.setArg(3, h_netVec[l + 1]);
				bakckprophidKern.setArg(4, lr);
				err = (OpenCL::clqueue).enqueueNDRangeKernel(bakckprophidKern, cl::NullRange,
					cl::NDRange(h_netVec[l]),
					cl::NullRange);


			}


		}

		if (e % 50000 == 0 && e!=0) {
			trainingAccuracy(testinputs, testtargets, 2000, 1);

		}



	}



}





//Computes the output of the net given an array of inputs
void ConvNN::computeConvolution() {



	convKern.setArg(0, d_InputBuffer);
	convKern.setArg(1, d_FiltersBuffer);
	convKern.setArg(2, d_FeatMapBuffer);
	convKern.setArg(3, filterdim);
	convKern.setArg(4, inputdim);
	convKern.setArg(5, featmapdim);


	err = (OpenCL::clqueue).enqueueNDRangeKernel(convKern, cl::NullRange,
		cl::NDRange(featmapdim, featmapdim, convLayer.numOfFilters),
		cl::NullRange);

	
}




void ConvNN::pooling() {



	poolKern.setArg(0, d_FeatMapBuffer);
	poolKern.setArg(1, d_PoolBuffer);
	poolKern.setArg(2, d_PoolIndexBuffer);
	poolKern.setArg(3, featmapdim);
	poolKern.setArg(4, pooldim);


	err = (OpenCL::clqueue).enqueueNDRangeKernel(poolKern, cl::NullRange,
		cl::NDRange(pooldim, pooldim, convLayer.numOfFilters),
		cl::NullRange);

	
}

void ConvNN::cnntoFcnn() {

	//pass output of cnn to fcnn
	for (int i = 0; i < convLayer.numOfFilters; i++) {
		cnnToFcnnKern.setArg(0, d_PoolBuffer);
		cnnToFcnnKern.setArg(1, d_layersBuffers[0]);
		cnnToFcnnKern.setArg(2, pooldim);
		cnnToFcnnKern.setArg(3, i);
		err = (OpenCL::clqueue).enqueueNDRangeKernel(cnnToFcnnKern, cl::NullRange,
			cl::NDRange(pooldim, pooldim,convLayer.numOfFilters),
			cl::NullRange);
	}

	


}








//Computes the output of the net given an array of inputs
void ConvNN::computeOutputofNN() {

	
	

	for (int i = 1; i<h_layers.size(); i++) {
		
		
		int sf = 0;
		if ((i == h_layers.size() - 1) && (softflag == 1))
			sf = 1;
		

		compoutKern.setArg(0, d_layersBuffers[i]);
		compoutKern.setArg(1, d_layersBuffers[i-1]);
		compoutKern.setArg(2, sf);

		err = (OpenCL::clqueue).enqueueNDRangeKernel(compoutKern, cl::NullRange,
			cl::NDRange(h_netVec[i]),
			cl::NullRange);

		if ((i == h_layers.size() - 1) && (softflag==1) ) {
			
			softmaxKern.setArg(0, d_layersBuffers[i]);
			softmaxKern.setArg(1, h_layers.back().numOfNodes);

			err = (OpenCL::clqueue).enqueueNDRangeKernel(softmaxKern, cl::NullRange,
				cl::NDRange(h_netVec[i]),
				cl::NullRange);
		}


		


		
	}

	


}






void ConvNN::trainingAccuracy(std::vector<std::vector<float>> &testinputs, std::vector<float> &testtargets,int num,bool onlyfcnn){


	float testerrors = 0;


	for (int i = 0; i < num; i++) {

		if (onlyfcnn == 0)
			forward(testinputs[i]);
		else
			forwardFCNN(testinputs[i]);

		Node bufdata[10];

		(OpenCL::clqueue).enqueueReadBuffer(d_layersBuffers.back(), CL_TRUE, 0, sizeof(Node)*h_layers.back().numOfNodes, bufdata);

		//findmax
		float max = 0;
		int maxindex = 0;
		for (int j = 0; j < h_netVec.back(); j++) {
			if (bufdata[j].output > max) {
				max = bufdata[j].output;
				maxindex = j;
			}
			
		}

		
		if (maxindex ==(int) testtargets[i])
			testerrors++;

		
		
		
		


	}

	std::cout << "Net:" << std::endl;
	if (onlyfcnn == 0) {
		std::cout << "Num of filters: " << convLayer.numOfFilters << std::endl;
		std::cout << "Filter dimension: " << filterdim << std::endl;
	}
	std::cout << "Fullconnected:";
	for (int i = 0; i < h_netVec.size();i++) {
		std::cout << h_netVec[i] <<" ";
	}
	std::cout << std::endl;
	std::cout << "Learning rate: " << lr << std::endl;
	



	//std::cout << "    Completed in " << difftime(end ,start) << " seconds" << endl;
	std::cout << "NUMBER OF CORRECT: " << testerrors << " CORRECT RATE: " << 100 * (testerrors / num) << "%" << std::endl;


	


}

void ConvNN::calculateError(std::vector<float> desiredout) {

	

	Node bufdata[10];

	(OpenCL::clqueue).enqueueReadBuffer(d_layersBuffers.back(), CL_TRUE, 0, sizeof(Node)*h_layers.back().numOfNodes, bufdata);

	float error=0;
	


}















