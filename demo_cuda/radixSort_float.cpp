void radixSort_float(float* arr, size_t n)
{
    unsigned int* arr_cast = reinterpret_cast<unsigned int*>(arr);
    // positive: sign bit set 0 to 1
    // negative: all bit trans
    // 32bits split into 4 bytes, sort 1 byte 1 time
    for (size_t i = 0; i < n; ++i)
    {
        arr_cast[i] = ((arr_cast[i] >> 31 & 0x1) ? (~arr_cast[i]) : arr_cast[i] | 0x80000000);
    }

    unsigned int* tmp = new unsigned int[n];

    for (int b = 0; b < 4; ++b)
    {
        int count[256] = {0};
        // 统计某一位出现相同数字的元素个数
        for (size_t i = 0; i < n; ++i)
        {
            int index = (arr_cast[i] >> (b * 8)) & 0xff;
            count[index]++;
        }

        int start[256] = {0};
        // 统计个位相同的元素在数组arr中出现的起始位置
        for (size_t i = 0; i < 256; ++i)
        {
            start[i] = count[i-1] + start[i-1]; // prefix sum -> scan
        }

        // 从桶中重新排列数据
        for (size_t i = 0; i < n; ++i)
        {
            int index = (arr_cast[i] >> (b * 8)) & 0xff;
            tmp[start[index]++] = arr_cast[i];
        }

        memcpy(arr_cast, tmp, n * sizeof(unsigned int));

    }

    // after sort, recover
    for (size_t i = 0; i < n; ++i)
    {
        arr_cast[i] = ((arr_cast[i] >> 31 & 0x1) ? (arr_cast[i] & 0x7fffffff) : ~arr_cast[i]);
    }

    delete[] tmp;
}