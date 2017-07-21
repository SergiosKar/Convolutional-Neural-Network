// Thesis.cpp : Defines the entry point for the console application.
//



#include "OpenCL.h"
#include "ConvNN.h"
#include "util.h"



void read_Mnist(std::string filename, std::vector<std::vector<float>> &vec);
void read_Mnist_Label(std::string filename, std::vector<std::vector<float>> &vec, std::vector<float> &testtargets, bool testflag);
void printInput(std::vector<float> &inputs);

int main(void)
{


	OpenCL::initialize_OpenCL();


	util::Timer timer;

	timer.reset();




	std::vector<std::vector<float> > inputs, targets;
	std::vector<std::vector<float> > testinputs;
	std::vector<float> testtargets;



	/*//////////////////////////////////
	std::vector<float> intemp(28*28);

	for (int i = 0; i < 28*28; i++) {
		intemp.at(i)=0.5;
	}

	for (int j = 0; j < 10000; j++)
		inputs.push_back(intemp);


	std::vector<float> temp(10);

	for(int i=0;i<1;i++)
		temp.at(i)=0;
	temp.at(1) = 1;

	for (int j = 0; j < 10000; j++)
		targets.push_back(temp);

	testinputs = inputs;
	for (int i = 0; i < 10000; i++)
		testtargets.push_back(1);
	//for (int i = 0; i < 64; i++) { std::cout << inputs[0].at(i); }
	///////////////////////////////////////////////*/



	/*/////////////////////////////////////////////////
	std::vector<float> intemp(28 * 28);



	for (int j = 0; j < 1000; j++) {
		for (int i = 0; i < 28 * 28; i++) {
			if (j % 2 == 0)
				intemp.at(i) = 0.2;
			else
				intemp.at(i) = 0.7;
		}
		inputs.push_back(intemp);
	}

	std::vector<float> temp(2);



	for (int j = 0; j < 1000; j++){
		if (j % 2 == 0) {
			temp.at(0) = 1;
			temp.at(1) = 0;

		}
		else {
			temp.at(0) = 0;
			temp.at(1) = 1;

		}
		targets.push_back(temp);
	}
	
	testinputs = inputs;
	for (int i = 0; i < 1000; i++) {
		if (i % 2 == 0)
			testtargets.push_back(0);
		else
			testtargets.push_back(1);
	}

	////////////////////////////////////////////////////////*/
	
	///////////////////////////////////////////////////
	read_Mnist("train-images.idx3-ubyte", inputs);
	read_Mnist_Label("train-labels.idx1-ubyte", targets,testtargets,0);


	std::cout << "MNIST loaded in: " <<timer.getTimeMilliseconds()/1000.0 <<" s"<<std::endl;

	timer.reset();
	read_Mnist("t10k-images.idx3-ubyte", testinputs);
	read_Mnist_Label("t10k-labels.idx1-ubyte", targets, testtargets, 1);

	//for (int i = 0; i < 30; i++)
		//std::cout << " " <<testtargets[i];
	std::cout << "MNIST test loaded in: " << timer.getTimeMilliseconds() / 1000.0 << " s" << std::endl;
	
	//printInput(inputs[54]);
	
	////////////////////////////////////////////////////*/
	
	ConvNN m_nn;
	m_nn.createConvNN(7,5,28);//num of filters,filterdim,imagedim


	//todo::many filters  3d kernel
	std::vector<int> netVec;
	netVec = { 144*7,10 };
	m_nn.createFullyConnectedNN(netVec);

	//todo::relu to fcnn instead of sigmoid
    //m_nn.forward(inputs[0]);

	
	m_nn.train(inputs, targets,60000);
	
	std::cout << "trained in : " << timer.getTimeMilliseconds()/1000.0 << " s"<<std::endl;



	m_nn.trainingAccuracy(testinputs, testtargets, 2000);
	
	 
}