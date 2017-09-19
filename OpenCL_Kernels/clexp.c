

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
////////////////////////////////////////////////////////////////////////////////

// Use a static data size for simplicity
//
#define DATA_SIZE (204800)

////////////////////////////////////////////////////////////////////////////////

// Simple compute kernel which computes the square of an input array
//
const char *KernelSource = "\n" \
"__kernel void square(                                                       \n" \
"   __global float* input,                                              \n" \
"   __global float* output,                                             \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count)                                                       \n" \
"       output[i] = native_exp(input[i]);                                \n" \
"}                                                                      \n" \
"\n";

////////////////////////////////////////////////////////////////////////////////

#define MEM_SIZE (32)
#define MAX_BINARY_SIZE (0x100000)


double get_time() //https://stackoverflow.com/questions/2349776/how-can-i-benchmark-c-code-easily
{
    struct timeval t;
    struct timezone tzp;
    gettimeofday(&t, &tzp);
    return t.tv_sec + t.tv_usec*1e-6;
}

int main(int argc, char** argv)
{
    int err;                            // error code returned from api calls
    
    float data[DATA_SIZE];              // original data set given to device
    float results[DATA_SIZE];           // results returned from device
    unsigned int correct;               // number of correct results returned
    
    size_t global;                      // global domain size for our calculation
    size_t local;                       // local domain size for our calculation
    
    cl_device_id device_id;             // compute device id
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel
    
    cl_mem input;                       // device memory used for the input array
    cl_mem output;                      // device memory used for the output array
    
    
    cl_uint deviceCount;
    cl_device_id* devices;
    // Fill our data set with random float values
    //
    int i = 0;
    unsigned int count = DATA_SIZE;
    for(i = 0; i < count; i++)
        data[i] = rand() / (float)RAND_MAX;
    
    
    char* value;
    size_t valueSize;
    clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
    devices = (cl_device_id*) malloc(sizeof(cl_device_id) * deviceCount);
    clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);

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
    
    
   
    float mem[MEM_SIZE];
    
    FILE *fp;
    char fileName[] = "./kernel.bc";
    size_t binary_size;
    char *binary_buf;
    cl_int binary_status;
    //cl_int i;
    
    /* Load kernel binary */
    fp = fopen(fileName, "r");
    if (!fp) {
        
        
    }
    binary_buf = (char *)malloc(MAX_BINARY_SIZE);
    binary_size = fread(binary_buf, 1, MAX_BINARY_SIZE, fp);
    fclose(fp);
    
    
    //char bitcode_path[100] = "kernel.bc\0";
    
    //size_t len = 1;
    //program = clCreateProgramWithBinary(context, 1, &device_id, &len,
    //                                    (const unsigned char**)&bitcode_path, NULL, &err);
    //check_status("clCreateProgramWithBinary", err);
    
    // The above tells OpenCL how to locate the intermediate bitcode, but we
    // still must build the program to produce executable bits for our
    // *specific* device.  This transforms gpu32 bitcode into actual executable
    // bits for an AMD or Intel compute device (for example).
    
    //err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    
    
     double start =get_time();
    program = clCreateProgramWithBinary(context, 1, &device_id, (const size_t *)&binary_size,
                                        (const unsigned char **)&binary_buf, &binary_status, &err);
    
    fprintf(stdout, "clCreateProgramWithBinary err: %d\n", err);
    fprintf(stdout, "binary_status: %d\n", binary_status);
    
    err = clBuildProgram( program, 1, &device_id, NULL, NULL, NULL );
    
   

    //0=check_status("clBuildProgram", err);
    
    // And that's it -- we have a fully-compiled program created from the
    // bitcode.  Let's ask OpenCL for the test kernel.
    
    
    
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
    input = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * count, NULL, NULL);
    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, NULL);
    if (!input || !output)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }
    
    // Write our data set into the input array in device memory
    //
    err = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(float) * count, data, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        exit(1);
    }
    
    // Set the arguments to our compute kernel
    //
    err = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &count);
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
    
    double bench = (get_time() - start);
      printf("Time = %f \n", bench*1000);

    // Read back the results from the device to verify the output
    //
    err = clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(float) * count, results, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }
    
    // Validate our results
    //
    /*
    correct = 0;
    for(i = 0; i < count; i++)
    {
        printf("vale %f, %f,  %f\n", results[i], data[i], exp(data[i]));
        if(results[i] == exp(data[i]))
            correct++;
    }
    
    // Print a brief summary detailing the results
    //
     
     printf("Computed '%d/%d' correct values!\n", correct, count);
     */
     printf("vale %f, %f,  %f\n", results[1], data[1], exp(data[1]));
    
    // Shutdown and cleanup
    //
    clReleaseMemObject(input);
    clReleaseMemObject(output);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
    
    return 0;
}
