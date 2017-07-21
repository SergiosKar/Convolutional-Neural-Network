#pragma once
#include "include.h"
#include "util.h"


class OpenCL {


public:

	static cl::Program clprogram;
	static cl::CommandQueue clqueue;
	static cl::Context clcontext;

	static void initialize_OpenCL();



};



