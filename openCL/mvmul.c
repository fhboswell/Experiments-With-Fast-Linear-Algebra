

////////////////////////////////////////////////////////////////////////////////

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <OpenCL/opencl.h>

#include <sys/time.h>
#include <sys/resource.h>
//gcc clexp.c  -framework OpenCL
//gcc clexp.c -O3 -framework OpenCL
////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////

// Simple compute kernel which computes the square of an input array
//
const char *KernelSource1 = "\n" \
"__kernel void square(                                                       \n" \
"   __global double* input,                                              \n" \
"   __global double* output,                                             \n" \
"   const unsigned int count, const int M, const int N)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count)                                                       \n" \
"       output[i] = input[i] + M + N;                                \n" \
"}                                                                      \n" \
"\n";

const char *KernelSource2 = "\n" \
"__kernel void square(                                                       \n" \
"   __global double* input,                                              \n" \
"   __global double* input2,                                                \n" \
"   __global double* output,                                             \n" \
"   const unsigned int count, const int M, const int N)                                            \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   double acc = 0.0;                                                   \n" \
"   for(int j = 0; j < N; j++){                                           \n" \
"       acc += input[M * j + i] * input2[j];                        \n" \
"   }                                                               \n" \
"   output[i] = acc;                                                \n" \
"}                                                                      \n" \
"\n";
////////////////////////////////////////////////////////////////////////////////

double get_time() //https://stackoverflow.com/questions/2349776/how-can-i-benchmark-c-code-easily
{
    struct timeval t;
    struct timezone tzp;
    gettimeofday(&t, &tzp);
    return t.tv_sec + t.tv_usec*1e-6;
}
int add(double * data, double * data2, int M, int N, double * results, int DATA_SIZE);
int main(int argc, char** argv)
{
    // results returned from device
    unsigned int M = 256;
    unsigned int N = 1024;
    unsigned int count = 256;
    
    
    double *X = (double *)calloc(M * N, sizeof(double));
    double *Y = (double *)calloc(N * 1, sizeof(double));
    double *results = (double *)calloc(N * 1, sizeof(double));
    
    *(X) = 1.0;
    *(X + 1) = 2.0;
    *(X + 256 ) = 3.0;
    *(X + 257) = 4.0;
    
    *(Y) = 1.0;
    *(Y + 1) = 2.0;
    *(Y + 2) = 0;
    *(Y + 3) = 0;
    add(X, Y, M, N, results,M);
    printf("vale  %f\n", results[1]);

}

int add(double * data, double * data2, int M, int N, double * results, int DATA_SIZE){
    int err;
    int i = 0;// error code returned from api calls
    
    // results returned from device
    unsigned int correct;               // number of correct results returned
    
    size_t global;                      // global domain size for our calculation
    size_t local;                       // local domain size for our calculation
    
    cl_device_id device_id;             // compute device id
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel
    
    cl_mem input;                       // device memory used for the input array
    cl_mem input2;
    cl_mem output;                      // device memory used for the output array
    
    
    cl_uint deviceCount;
    cl_device_id* devices;
    
    
    
    unsigned int count = DATA_SIZE;
    char* value;
    size_t valueSize;
    clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
    devices = (cl_device_id*) malloc(sizeof(cl_device_id) * deviceCount);
    clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);
    
    //clGetDeviceInfo(devices[2], CL_DEVICE_NAME, 0, NULL, &valueSize);
    //value = (char*) malloc(valueSize);
    //clGetDeviceInfo(devices[2], CL_DEVICE_NAME, valueSize, value, NULL);
    //printf("%d. Device: %s\n", 2+1, value);
    device_id = devices[2];
    
    
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, NULL);
    commands = clCreateCommandQueue(context, device_id, 0, NULL);
    cl_event event = NULL;
    
    
    
    
    
    if (!commands)
    {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }
    
    // Create the compute program from the source buffer
    //
    program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource2, NULL, &err);
    if (!program)
    {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }
    
    // Build the program executable
    //
    
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];
        
        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }
    
    // Create the compute kernel in the program we wish to run
    //
    kernel = clCreateKernel(program, "square", &err);
    if (!kernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }
    
    // Create the input and output arrays in device memory for our calculation
    //
    input = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(double) * M * N, NULL, NULL);
    input2 = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(double) * N, NULL, NULL);
    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double) * N, NULL, NULL);
    if (!input || !output)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }
    
    // Write our data set into the input array in device memory
    //
    err = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(double) *  M * N, data, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(commands, input2, CL_TRUE, 0, sizeof(double) *  N, data2, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        exit(1);
    }
    
    // Set the arguments to our compute kernel
    //
    err = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    err  = clSetKernelArg(kernel, 1, sizeof(cl_mem), &input2);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &count);
    err |= clSetKernelArg(kernel, 4, sizeof(int), &M);
    err |= clSetKernelArg(kernel, 5, sizeof(int), &N);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }
    
    // Get the maximum work group size for executing the kernel on the device
    //
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }
    
    
    
    
    
    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    //
    global = count;
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err)
    {
        printf("Error: Failed to execute kernel!\n");
        return EXIT_FAILURE;
    }
    
    // Wait for the command commands to get serviced before reading back results
    //
    clFinish(commands);
    err = clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(double) * M, results, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }
    
    clReleaseMemObject(input);
    clReleaseMemObject(output);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
    
    return 0;
}

