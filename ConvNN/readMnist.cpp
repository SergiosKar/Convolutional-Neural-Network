#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <math.h>
#include <iostream>
#include <fstream>


using namespace cv;






int ReverseInt(int i)
{

	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;

}



void read_Mnist(std::string filename, std::vector< std::vector<float>> &vec)

{

	std::ifstream file(filename, std::ios::binary);
	if (file.is_open())

	{

		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;

		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);

		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);

		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);

		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols); 
			for (int i = 0; i < number_of_images; ++i)
			{

				std::vector<float> tp;

				for (int r = 0; r < n_rows; ++r)
				{
					for (int c = 0; c < n_cols; ++c)
					{

						unsigned char temp = 0;
						file.read((char*)&temp, sizeof(temp));
						//tp.push_back(((float)temp/255.0)*2-1); // x=(x/255)*range =min
						tp.push_back(((float)temp / 255.0));

					}

				}

				vec.push_back(tp);

			}

	}

}



void read_Mnist(std::string filename, std::vector<cv::Mat> &vec) {

	std::ifstream file(filename, std::ios::binary);
	if (file.is_open())

	{

		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;

		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number); 
			file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);

		for (int i = 0; i < number_of_images; ++i)
		{

			cv::Mat tp = Mat::zeros(n_rows, n_cols, CV_8UC1);

			for (int r = 0; r < n_rows; ++r)
			{
				for (int c = 0; c < n_cols; ++c)

				{

					unsigned char temp = 0;
					file.read((char*)&temp, sizeof(temp));
					tp.at<uchar>(r, c) = (float)temp/255.0;

				}

			}

			vec.push_back(tp);

		}

	}

}




void read_Mnist_Label(std::string filename, std::vector<std::vector<float>> &vec,std::vector<float> &testtargets,bool testflag)

{

	std::ifstream file(filename, std::ios::binary);
	if (file.is_open())

	{

		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;

		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);

		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);


		if (testflag == 0) {
			std::vector<float> tempvec(10);
			for (int i = 0; i < number_of_images; ++i)
			{

				unsigned char temp = 0;
				file.read((char*)&temp, sizeof(temp));


				
				for (int j = 0; j < 10; j++) {
					if (j == (int)temp)
						tempvec[j] = 1.0;
					else
						tempvec[j] = 0.0;
				}
				

				vec.push_back(tempvec);

			}
		}
		else {
			
			for (int i = 0; i < number_of_images; ++i)
			{

				unsigned char temp = 0;
				file.read((char*)&temp, sizeof(temp));

				testtargets.push_back((float) temp);
				//testtargets.push_back((float)temp>5 ? 1 : 0);

			}
		
		
		}

	}

}


void printInput(std::vector<float> &inputs)
{
	std::cout << "BELOW IS AN IMAGE" << std::endl;
	int c = 0;
	for (int i = 0; i != 32; ++i)
	{
		std::cout << "    " << std::endl;
		for (int j = 0; j != 32; ++j)
		{
			if (inputs[c] > 0)
				std::cout << 1;
			else
				std::cout << 0;
			++c;
		}
		std::cout << std::endl;
	}
}
