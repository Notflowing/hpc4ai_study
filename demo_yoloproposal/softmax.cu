
// pack_size: 每个线程处理的数据包pack的大小
// cols_per_thread: 每个线程处理的列数
template <typename Dtype, int pack_size, int cols_per_thread, bool padding, int thread_group_width = kWarpSize>
__global__ void softmax_warp_kernel(const Dtype* input, Dtype* output, int rows, int cols)
{
    // static_assert(cols_per_thread % pack_size == 0, "");    // 确保每个线程处理的列数是每个线程处理的数据包大小的倍数
    // static_assert(thread_group_width <= kWarpSize, "");     // 处理元素的线程组的宽度需要小于等于kWarpSize
    // static_assert(kWarpSize % thread_group_width == 0, ""); // 线程组的宽度需要被kWarpSize整除
    assert(cols_per_thread % pack_size == 0);               // 确保每个线程处理的列数是每个线程处理的数据包大小的倍数
    assert(cols <= cols_per_thread * thread_group_width);   // 需要保证cols <= 每个线程处理的元素个数 * 处理元素的线程组的宽度(这里是warpSize(32))
    constexpr int num_packs = (cols_per_thread + pack_size - 1) / pack_size; // 每个线程处理的pack后的元素个数
    Dtype buf[cols_per_thread]; // 开一块寄存器，长度为每个线程处理的元素个数
    const int row_idx = blockIdx.x * blockDim.y + threadIdx.y; // 获得全局warp的id
    const int lane_id = threadIdx.x; // 获得当前warp中的通道号


    // for循环的开始为当前行号，结束为总行数
    for (int row = row_idx; row < rows; row += gridDim.x * blockDim.y)
    {
        const int row_offset = row * cols;
        const Dtype* row_input = input + row_offset;
        Dtype* row_output = output + row_offset;
        Dtype thread_max = -FLT_MAX; // 记录当前warp处理的行的最大值
        // 循环处理num_packs
        for (int pack_id = 0; pack_id < num_packs; ++pack_id)
        {
            const int pack_offset = pack_id * pack_size; // pack的偏移量
            const int col = (pack_id * thread_group_width + lane_id) * pack_size; // 计算当前线程处理的列索引
            if (!padding || col < cols) // 如果不使用填充或者列在有效范围内
            { // 将输入数据加入到缓冲区
                for (int i = 0; i < pack_size; ++i)
                {
                    buf[pack_offset + i] = row_input[col + i];
                    thread_max = max(thread_max, buf[pack_offset + i]);
                }
            }
            else // 否则，设为负无穷，初始化buf数组中的一部分将被设为负无穷
            {
                for (int i = 0; i < pack_size; ++i)
                {
                    buf[pack_offset + i] = -FLT_MAX;
                }
            }
        }

        // 计算一组warp的最大值， MaxOp定义了如何比较和选择最大值(warp shuffle)
        const Dtype warp_max = WarpAllReduce<MaxOp, Dtype, thread_group_width>(thread_max);
        Dtype thread_sum = 0;
        // 对单个线程，计算softmax并求和
        for (int i = 0; i < cols_per_thread; ++i)
        {
            buf[i] = exp(buf[i] - warp_max);
            thread_sum += buf[i];
        }
        // 计算一组warp的累加和
        const Dtype warp_sum = WarpAllReduce<SumOp, Dtype, thread_group_width>(thread_sum);
        // 归一化，使得和为1，呈概率分布
        for (int i = 0; i < cols_per_thread; ++i)
        {
            buf[i] = buf[i] / warp_sum;
        }

        // 将结果写回output
        for (int pack_id = 0; pack_id < num_packs; ++pack_id)
        {
            const int col = (pack_id * thread_group_width + lane_id) * pack_size; // 计算当前线程处理的列索引
            if (!padding || col < cols) // 如果不使用填充或者列在有效范围内
            {
                for (int i = 0; i < pack_size; ++i)
                {
                    row_output[col + i] = buf[pack_id * pack_size + i];
                }
            }
        }

    } // loop for row
}

int main()
{
    dim3 threads(32, 4);
    dim3 blocks;
    blocks.x = (rows + threads.y - 1) / threads.y;

    constexpr int pack_size = 1;
    constexpr int iter = 1;

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    if (cols_padding < kWarpSize)
    {
        constexpr int cols_per_thread = 1;
        TICK(softmax_warp);
        for (int it = 0; it < iter; ++it)
        {
            softmax_warp_kernel<type_t, pack_size, cols_per_thread, padding, cols_padding>
                   <<<blocks, threads, 0, stream>>>(d_input, d_output, rows, cols);
        }
        TOCK(softmax_warp, rows, cols);
        cudaStreamSynchronize(stream);
    }
    else
    {
        constexpr int cols_per_thread = cols_padding / kWarpSize;
        TICK(softmax_warp);
        for (int it = 0; it < iter; ++it)
        {
            softmax_warp_kernel<type_t, pack_size, cols_per_thread, padding, kWarpSize>
                   <<<blocks, threads, 0, stream>>>(d_input, d_output, rows, cols);
        }
        TOCK(softmax_warp, rows, cols);
        cudaStreamSynchronize(stream);
    }

}