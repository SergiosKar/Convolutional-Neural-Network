// Thesis.cpp : Defines the entry point for the console application.
//



#include "OpenCL.h"
#include "ConvNN.h"
#include "util.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"



void read_Mnist(std::string filename, std::vector<std::vector<float>> &vec);
void read_Mnist_Label(std::string filename, std::vector<std::vector<float>> &vec, std::vector<float> &testtargets, bool testflag);
void printInput(std::vector<float> &inputs);
void read_CIFAR10(cv::Mat &trainX, cv::Mat &testX, cv::Mat &trainY, cv::Mat &testY);

int main(void)
{


	try {

		OpenCL::initialize_OpenCL();



		util::Timer timer;

		timer.reset();




		std::vector<std::vector<float> > inputs, targets;
		std::vector<std::vector<float> > testinputs;
		std::vector<float> testtargets;



		/*//////////////////////////////////
		std::vector<float> intemp(28 * 28);

		for (int i = 0; i < 28 * 28; i++) {
			intemp.at(i) = 0.5;
		}

		for (int j = 0; j < 10000; j++)
			inputs.push_back(intemp);


		std::vector<float> temp(10);

		for (int i = 0; i < 1; i++)
			temp.at(i) = 0;
		temp.at(1) = 1;

		for (int j = 0; j < 10000; j++)
			targets.push_back(temp);

		testinputs = inputs;
		for (int i = 0; i < 10000; i++)
			testtargets.push_back(1);
		

		////////////////////////////////////////////////////////*/

		///MNIST
		/*//////////////////////////////////////////////////
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


		///CIFAR10
		/////////////////////////////////////////////////////////
		cv::Mat trainX, testX;

		cv::Mat trainY, testY;

		trainX = cv::Mat::zeros(1024, 50000, CV_32FC1);

		testX = cv::Mat::zeros(1024, 10000, CV_32FC1);

		trainY = cv::Mat::zeros(1, 50000, CV_32FC1);

		testY = cv::Mat::zeros(1, 10000, CV_32FC1);


		read_CIFAR10(trainX, testX, trainY, testY);



		std::cout << "Cifar10 loaded in: " << timer.getTimeMilliseconds() / 1000.0 << " s" << std::endl;

		


		timer.reset();


		for (int i = 0; i < 50000; i++) {
			inputs.push_back(trainX.col(i));

			std::vector<float> tempvec(10);

			for (int j = 0; j < 10; j++) {
				if (j == trainY.col(i).at<float>(0))
					tempvec[j] = (float)1.0;
				else
					tempvec[j] = (float) 0.0;
			}
			targets.push_back(tempvec);

		}
		for (int i = 0; i < 10000; i++) {
			testinputs.push_back(testX.col(i));
			testtargets.push_back(testY.col(i).at<float>(0));

		}



		std::cout << "Cifar10 converted in: " << timer.getTimeMilliseconds() / 1000.0 << " s" << std::endl;

		timer.reset();




		////////////////////////////////////////////////////////*/

		///CNN
		//////////////////////////////////////////////////////////

		ConvNN m_nn;
		m_nn.createConvNN(7, 7, 32);//num of filters,filterdim,imagedim


		//todo::many filters  3d kernel
		std::vector<int> netVec;
		netVec = { 169 * 7,10 };
		m_nn.createFullyConnectedNN(netVec, 0, 32);

	

		m_nn.train(inputs, targets, testinputs, testtargets, 1000000);

		std::cout << "trained in : " << timer.getTimeMilliseconds() / 1000.0 << " s" << std::endl;



		//////////////////////////////////////////////////////////////////////////////*/

		/// FCNN
		/*////////////////////////////////////////////////////

	   ConvNN m_nn;

	   std::vector<int> netVec;
	   netVec = { 1024,10 };
	   m_nn.createFullyConnectedNN(netVec, 1, 32);


	   //m_nn.forwardFCNN(inputs[0]);


	   m_nn.trainFCNN(inputs, targets, testinputs, testtargets, 50000);

	   std::cout << "trained in : " << timer.getTimeMilliseconds() / 1000.0 << " s" << std::endl;

	   m_nn.trainingAccuracy(testinputs, testtargets, 2000, 1);
	   /////////////////////////////////////////////////////////////*/

	}
	

	catch (cl::Error e) 
	{
		std::cout << "opencl error: " << e.what() << std::endl;
		std::cout << "error number: " << e.err() << std::endl;
	}
	catch (int e)
	{
		std::cout << "An exception occurred. Exception Nr. " << e << '\n';
	}

	
}