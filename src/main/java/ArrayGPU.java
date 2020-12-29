import org.jocl.*;

import java.util.Arrays;

// from https://blogs.oracle.com/javamagazine/programming-the-gpu-in-java
public class ArrayGPU {
    /**
     * The source code of the OpenCL program
     */
    private static String programSource =
            "__kernel void "+
                    "sampleKernel(__global const float *a,"+
                    "             __global const float *b,"+
                    "             __global float *c)"+
                    "{"+
                    "    int gid = get_global_id(0);"+
                    "    c[gid] = a[gid] + b[gid];"+
                    "}";

    public static void main(String args[])
    {
        int n = 10;
        float srcArrayA[] = new float[n];
        float srcArrayB[] = new float[n];
        float dstArray[] = new float[n];
        for (int i=0; i<n; i++)
        {
            srcArrayA[i] = i;
            srcArrayB[i] = i + 1;
        }
        Pointer srcA = Pointer.to(srcArrayA);
        Pointer srcB = Pointer.to(srcArrayB);
        Pointer dst = Pointer.to(dstArray);


        // The platform, device type and device number
        // that will be used
        final int platformIndex = 0;
        final long deviceType = CL.CL_DEVICE_TYPE_GPU;
        final int deviceIndex = 0;

        // Enable exceptions and subsequently omit error checks in this sample
        CL.setExceptionsEnabled(true);

        // Obtain the number of platforms
        int numPlatformsArray[] = new int[1];
        CL.clGetPlatformIDs(0, null, numPlatformsArray);
        int numPlatforms = numPlatformsArray[0];
        System.out.println("Number of platforms: " + numPlatforms);

        // Obtain a platform ID
        cl_platform_id platforms[] = new cl_platform_id[numPlatforms];
        CL.clGetPlatformIDs(platforms.length, platforms, null);
        cl_platform_id platform = platforms[platformIndex];

        // Initialize the context properties
        cl_context_properties contextProperties = new cl_context_properties();
        contextProperties.addProperty(CL.CL_CONTEXT_PLATFORM, platform);

        // Obtain the number of devices for the platform
        int numDevicesArray[] = new int[1];
        CL.clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
        int numDevices = numDevicesArray[0];
        System.out.println("Number of devices: " + numDevices);

        // Obtain a device ID
        cl_device_id devices[] = new cl_device_id[numDevices];
        CL.clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
        cl_device_id device = devices[deviceIndex];
        for (int i = 0 ; i < numDevices ; i++) {
            System.out.println(devices[i].toString());
        }

        // Create a context for the selected device
        cl_context context = CL.clCreateContext(
                contextProperties, 1, new cl_device_id[]{device},
                null, null, null);

        // Create a command-queue for the selected device
        cl_command_queue commandQueue =
                CL.clCreateCommandQueue(context, device, 0, null);

        // Allocate the memory objects for the input and output data
        cl_mem memObjects[] = new cl_mem[3];
        memObjects[0] = CL.clCreateBuffer(context,
                CL.CL_MEM_READ_ONLY | CL.CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * n, srcA, null);
        memObjects[1] = CL.clCreateBuffer(context,
                CL.CL_MEM_READ_ONLY | CL.CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * n, srcB, null);
        memObjects[2] = CL.clCreateBuffer(context,
                CL.CL_MEM_READ_WRITE,
                Sizeof.cl_float * n, null, null);

        // Create the program from the source code
        cl_program program = CL.clCreateProgramWithSource(context,
                1, new String[]{ programSource }, null, null);

        // Build the program
        CL.clBuildProgram(program, 0, null, null, null, null);

        // Create the kernel
        cl_kernel kernel = CL.clCreateKernel(program, "sampleKernel", null);

        // Set the arguments for the kernel
        CL.clSetKernelArg(kernel, 0,
                Sizeof.cl_mem, Pointer.to(memObjects[0]));
        CL.clSetKernelArg(kernel, 1,
                Sizeof.cl_mem, Pointer.to(memObjects[1]));
        CL.clSetKernelArg(kernel, 2,
                Sizeof.cl_mem, Pointer.to(memObjects[2]));

        // Set the work-item dimensions
        long global_work_size[] = new long[]{n};
        long local_work_size[] = new long[]{1};

        // Execute the kernel
        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 1, null,
                global_work_size, local_work_size, 0, null, null);

        // Read the output data
        CL.clEnqueueReadBuffer(commandQueue, memObjects[2], CL.CL_TRUE, 0,
                n * Sizeof.cl_float, dst, 0, null, null);

        // Release kernel, program, and memory objects
        CL.clReleaseMemObject(memObjects[0]);
        CL.clReleaseMemObject(memObjects[1]);
        CL.clReleaseMemObject(memObjects[2]);
        CL.clReleaseKernel(kernel);
        CL.clReleaseProgram(program);
        CL.clReleaseCommandQueue(commandQueue);
        CL.clReleaseContext(context);

        System.out.println(Arrays.toString(srcArrayA));
        System.out.println(Arrays.toString(srcArrayB));
        System.out.println(Arrays.toString(dstArray));
    }

    private static String getString(cl_device_id device, int paramName) {
        // Obtain the length of the string that will be queried
        long size[] = new long[1];
        CL.clGetDeviceInfo(device, paramName, 0, null, size);

        // Create a buffer of the appropriate size and fill it with the info
        byte buffer[] = new byte[(int)size[0]];
        CL.clGetDeviceInfo(device, paramName, buffer.length, Pointer.to(buffer), null);

        // Create a string from the buffer (excluding the trailing \0 byte)
        return new String(buffer, 0, buffer.length-1);
    }
}
