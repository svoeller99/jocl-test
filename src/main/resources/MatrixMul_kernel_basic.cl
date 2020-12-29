__kernel void MatrixMul_kernel_basic(int dim,
                  __global float *A,
                  __global float *B,
                  __global float *C){

    int iCol = get_global_id(0);
    int iRow = get_global_id(1);
    float result = 0.0;
    for(int i=0; i< dim; ++i)
    {
          result +=
          A[iRow*dim + i]*B[i*dim + iCol];
    }
    C[iRow*dim + iCol] = result;
}