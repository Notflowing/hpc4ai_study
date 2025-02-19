__global__ void warp_shuffle_hs_scan(uint* d_Src, uint* d_Dst, const uint size, 
                                     uint* partial_sums = NULL)
{
    const int num_warps = blockDim.x / kWarpSize;
    extern __shared__ uint sums[];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane_id = tid % kWarpSize;
    const int warp_id = tid / kWarpSize;

    uint value = bid < size ? d_Src[bid] : 0;
    uint origin = value;
    // using a shfl instruction for a scan. (Hillis Steele scan)
    for (int i = 1; i < kWarpSize; i *= 2)
    {
        uint mask = 0xffffffff;
        uint n = __shfl_up_sync(mask, value, i, kWarpSize);
        if (lane_id >= i) value += n;  // inclusive scan
    }

    // write the sum of the warp to smem
    if (lane_id == kWarpSize - 1)
    {
        sums[warp_id] = value;
    }
    __syncthreads();

    // scan sum the warp sums
    if (warp_id == 0 && lane_id < num_warps)
    {
        uint warp_sum = sums[lane_id];
        uint mask = (0x01 << num_warps) - 1;
        for (int i = 1; i < num_warps; i *= 2)
        {
            uint n = __shfl_up_sync(mask, warp_sum, i, num_warps);
            if (lane_id >= i) warp_sum += n;  // inclusive scan
        }

        sums[lane_id] = warp_sum;
    }
    __syncthreads();

    // perform a uniform add across warps in the block
    // read neighbouring warp's sum and add it to threads value
    int block_sum = 0;
    if (warp_id > 0) block_sum = sums[warp_id - 1];
    value += block_sum - origin;  // exclusive scan
    d_Dst[bid] = value;

    // last thread has sum, write out the block's sum
    if (partial_sums != NULL && tid == blockDim.x - 1)
    {
        partial_sums[blockIdx.x] = value + origin;
        // note: 这整个文件是用于计算exclusive scan，核函数也是用于计算exclusive scan，
        // 但在partial_sums的计算中，我们需要计算inclusive scan，用于下一步的scan计算，以确保最终结果的正确性
    }

}

__global__ void uniform_add(uint* d_Dst, uint* partial_sums, const uint size)
{
    __shared__ uint buf;
    const int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid > size) return;

    if (threadIdx.x == 0) buf = partial_sums[blockIdx.x];
    __syncthreads();

    d_Dst[bid] += buf;

}

{
    int block_size = THREADBLOCK_SIZE;
    int grid_size = N / block_size;

    int n_partialSums = N / block_size;
    int p_blockSize = n_partialSums > block_size ? block_size : n_partialSums;
    int p_gridSize = (n_partialSums + p_blockSize - 1) / p_blockSize;

    const int num_warps = block_size / kWarpSize;
    warp_shuffle_hs_scan<<<grid_size, block_size, num_warps * sizeof(type_t), stream>>>
                        (d_input_warp_shfl, d_output_warp_shfl, N, d_partial_sums);
    warp_shuffle_hs_scan<<<p_gridSize, p_blockSize, num_warps * sizeof(type_t), stream>>>
                        (d_partial_sums, d_partial_sums, n_partialSums);
    // 只进行两步的scan，对于序列长度仍有限制
    // print_partial_sums<<<1, 1, 0, stream>>>(d_partial_sums, n_partialSums);
    uniform_add<<<grid_size, block_size, 0, stream>>>
               (d_output_warp_shfl, d_partial_sums, N);

}