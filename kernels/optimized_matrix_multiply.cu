for (int m = 0; m < (N + 31) / 32; m++) {
    if (row < N && m * 32 + threadIdx.x < N)
        tileA[threadIdx.y][threadIdx.x] = A[row * N + m * 32 + threadIdx.x];
    else
        tileA[threadIdx.y][threadIdx.x] = 0.0f;
    
    if (col < N && m * 32 + threadIdx.y < N)
        tileB[threadIdx.y][threadIdx.x] = B[(m * 32 + threadIdx.y) * N + col];
    else
        tileB[threadIdx.y][threadIdx.x] = 0.0f;
    
    __syncthreads();
    
    for (int k = 0; k < 32; k++)
        sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
    
    __syncthreads();
}

if (row < N && col < N)
    C[row * N + col] = sum;
