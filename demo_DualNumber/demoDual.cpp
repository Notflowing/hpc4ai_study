#include <iostream>
#include <vector>
#include "DualNumber.hpp"

// template<typename DType>
// DType testFunction(DType x) {
//     return 3.0f * x * x + 2.0f * x;      // 这里必须写成3.0f的类型必须和dualnumber的值类型一致，不是一个好的写法
// }

// template<typename DType>                 // PASS，但不是最好的选择
// DType testFunction(DType x) {
//     return static_cast<typename DType::value_type>(3) * x * x + static_cast<typename DType::value_type>(2) * x;
// }

// ** 以下这种写法是错误的，正确的写法应该是再下面那个
// ** 只需指定模版参数的名称 (typename T),而不应该再次包装它们
// template<typename dualnumbers::Dual<T>>  // ERROR
// dualnumbers::Dual<T> testFunction(dualnumbers::Dual<T> x) {
//     return static_cast<T>(3) * x * x + static_cast<T>(2) * x;
// }

template<typename T>                        // 好的选择，PASS
dualnumbers::Dual<T> testFunction(dualnumbers::Dual<T> x) {
    return static_cast<T>(3) * x * x + static_cast<T>(2) * x;
}

// template <typename std::vector<T>>       // ERROR
// T test(std::vector<T> const& arr) {
//     return *arr.begin();
// }

template <typename T>                       // PASS
T test(std::vector<T> const& arr) {
    return *arr.begin();
}

int main() {
    constexpr dualnumbers::Dual a{1.0, 2.0};
    constexpr dualnumbers::Dual b{3.0, 4.0};
    constexpr dualnumbers::Dual c = a + b;

    std::cout << c.a() << " " << c.b() << std::endl;
    std::cout << c << std::endl;
    std::cout << (c == dualnumbers::Dual{4.0, 6.0}) << std::endl;

    constexpr dualnumbers::Dual<float> d{2.0, 1.0};
    float dfdx = testFunction(d).b();
    std::cout << dfdx << std::endl;

    std::vector<int> arr{1, 2, 3, 4, 5};
    std::cout << test(arr) << std::endl;

    return 0;
}