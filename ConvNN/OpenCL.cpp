

#include "OpenCL.h"
#include "util.h"



cl::Program OpenCL::clprogram;
cl::Context OpenCL::clcontext;
cl::CommandQueue OpenCL::clqueue;

void OpenCL::initialize_OpenCL() {
	// get all platforms (drivers), e.g. NVIDIA
	std::vector<cl::Platform> all_platforms;
	cl::Platform::get(&all_platforms);

	if (all_platforms.size() == 0) {
		std::cout << " No platforms found. Check OpenCL installation!\n";
		exit(1);
	}
	cl::Platform default_platform = all_platforms[0];
	std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

	// get default device (CPUs, GPUs) of the default platform
	std::vector<cl::Device> all_devices;
	default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
	if (all_devices.size() == 0) {
		std::cout << " No devices found. Check OpenCL installation!\n";
		exit(1);
	}

	
	cl::Device default_device = all_devices[0];
	std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";
	/*////////////////////////////////////////////////////
	std::cout << "\t-------------------------" << std::endl;

	std::string s;
	default_device.getInfo(CL_DEVICE_NAME, &s);
	std::cout << "\t\tName: " << s << std::endl;

	//default_device.getInfo(CL_DEVICE_OPENCL_C_VERSION, &s);
	//std::cout << "\t\tVersion: " << s << std::endl;

	int i;
	default_device.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &i);
	std::cout << "\t\tMax. Compute Units: " << i << std::endl;

	size_t size;
	default_device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &size);
	std::cout << "\t\tLocal Memory Size: " << size / 1024 << " KB" << std::endl;

	default_device.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &size);
	std::cout << "\t\tGlobal Memory Size: " << size / (1024 * 1024) << " MB" << std::endl;

	default_device.getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &size);
	std::cout << "\t\tMax Alloc Size: " << size / (1024 * 1024) << " MB" << std::endl;

	default_device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &size);
	std::cout << "\t\tMax Work-group Total Size: " << size << std::endl;

	std::vector<size_t> d;
	default_device.getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &d);
	std::cout << "\t\tMax Work-group Dims: (";
	for (std::vector<size_t>::iterator st = d.begin(); st != d.end(); st++)
		std::cout << *st << " ";
	std::cout << "\x08)" << std::endl;

	std::cout << "\t-------------------------" << std::endl;
	////////////////////////////////////////////////////////////*/

	// a context is like a "runtime link" to the device and platform;
	// i.e. communication is possible
	OpenCL::clcontext=cl::Context({ default_device });

	// create the program that we want to execute on the device
	cl::Program::Sources sources;

	
	std::string src,src2,src3;
	
	src = util::loadProgram("kernelheader.cl");
	src2 = util::loadProgram("fcnn_kernels.cl");
	src3 = util::loadProgram("conv_kernels.cl");

	src = src + src2+src3;
	sources.push_back({ src.c_str(), src.length() });

	OpenCL::clprogram=cl::Program(OpenCL::clcontext, sources);
	try {
		OpenCL::clprogram.build({ default_device });
	}
	catch (...) {

		cl_int buildErr = CL_SUCCESS;
		auto buildInfo = OpenCL::clprogram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&buildErr);
		for (auto &pair : buildInfo) {
			std::cerr << pair.second << std::endl << std::endl;
		}
		
	}

	OpenCL::clqueue=cl::CommandQueue(OpenCL::clcontext, default_device);
	

	


}




