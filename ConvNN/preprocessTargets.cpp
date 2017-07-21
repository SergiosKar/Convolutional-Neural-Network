#include "include.h"


void preprocessTargets(std::vector<float> &prevtargets , std::vector<std::vector<float>> &newtargets){

	std::vector<float> temp(10);
	for (int i = 0; i < prevtargets.size(); i++) {
		for (int j = 0; i < 10; i++) {
			if (i == prevtargets[i])
				temp[i] = 1;
			else
				temp[i] = 0;
		}

		newtargets.push_back(temp);
	
	}


}