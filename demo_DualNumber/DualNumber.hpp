#pragma once

#include <cmath>
#include <limits>
#include <complex>
#include <iostream>

namespace dualnumbers {

/*
    @brief traits of dual number
    @tparam T: arbitrary type
*/
template<typename T>
struct dual_number_traits {
    static constexpr auto a(T const& val) {
        return val.a();
    }
    static constexpr auto b(T const& val) {
        return val.b();
    }
};

/*
    @brief traits of std::complex
*/
template<typename T>
struct dual_number_traits<std::complex<T>> {
    static constexpr auto a(std::complex<T> const& val) {
        return val.real();
    }
    static constexpr auto b(std::complex<T> const& val) {
        return val.imag();
    }
};

/*
    @breif implement of dual number
    @tparam T type
*/
template<typename T>
struct Dual {
    using this_type = Dual<T>;
    using value_type = T;

    // 对偶数零元, return 0 + 0ε
    static constexpr this_type zero() {
        return this_type{T(0.0), T(0.0)};
    }

    // 对偶数加法单位元, return 0 + 0ε
    static constexpr this_type ID_add() {
        return zero();
    }

    // 对偶数乘法单位元, return 1 + 0ε
    static constexpr this_type ID_mul() {
        return this_type{T(1.0), T(0.0)};
    }

    // Default constructor
    constexpr Dual() = default;
    // Parameterized constructor
    constexpr Dual(T a, T b = T(0.0)): m_a{a}, m_b{b}
    {}
    // Copy constructor
    // 需要适合其类型dual_number_traits<T>的特化, other: 任意的对偶数
    template<typename OtherDual>
    constexpr Dual(OtherDual const& other):
        m_a{T{dual_number_traits<OtherDual>::a(other)}},
        m_b{T{dual_number_traits<OtherDual>::b(other)}}
    {}
    constexpr Dual(this_type const& other) = default;
    // move constructor
    constexpr Dual(this_type&& other) = default;

    // operator=
    constexpr this_type& operator=(this_type const& other) = default;
    constexpr this_type& operator=(this_type&& other) = default;

    // Cast operator overload
    constexpr operator T() const {
        return m_a;
    }

    constexpr this_type operator+() const {
        return *this;
    }

    constexpr this_type operator-() const {
        return this_type{-m_a, -m_b};
    }

    constexpr bool operator==(this_type const& rhs) const {
        // return m_a == rhs.m_a && m_b == rhs.m_b;
        return std::abs(m_a - rhs.m_a) <= std::numeric_limits<value_type>::epsilon() && 
               std::abs(m_b - rhs.m_b) <= std::numeric_limits<value_type>::epsilon();
    }

    constexpr bool operator!=(this_type const& rhs) const {
        return !(*this == rhs);
    }

    constexpr this_type& operator++() {                     // 前置++
        ++m_a;
        return *this;
    }

    constexpr this_type& operator--() {
        --m_a;
        return *this;
    }

    constexpr this_type operator++(const int ignored) {     // 后置++
        this_type const ret{*this};
        ++m_a;
        return ret;
    }

    constexpr this_type operator--(const int ignored) {
        this_type const ret{*this};
        --m_a;
        return ret;
    }

    // dual number and dual number operator
    constexpr this_type& operator+=(this_type const& rhs) {
        m_a += rhs.m_a;
        m_b += rhs.m_b;
        return *this;
    }

    constexpr this_type& operator-=(this_type const& rhs) {
        m_a -= rhs.m_a;
        m_b -= rhs.m_b;
        return *this;
    }

    constexpr this_type& operator*=(this_type const& rhs) {
		//(a+bε)*(c+dε) = ac + (ad + bc)ε
		m_b *= rhs.m_a;
		m_b += m_a * rhs.m_b;
		m_a *= rhs.m_a;
		return *this;
	}

    constexpr this_type& operator/=(this_type const& rhs) {
        // (a+bε)/(c+dε) = a/c + (-ad + bc)ε/c^2
        m_b *= rhs.m_a;
        m_b -= m_a * rhs.m_b;
        m_b /= rhs.m_a * rhs.m_a;
        m_a /= rhs.m_a;
        return *this;
    }

    // dual number and scalar
    constexpr this_type& operator+=(value_type const& rhs) {
        m_a += rhs;
        return *this;
    }

    constexpr this_type& operator-=(value_type const& rhs) {
        m_a -= rhs;
        return *this;
    }

    constexpr this_type& operator*=(value_type const& rhs) {
		m_a *= rhs;
		m_b *= rhs;
		return *this;
	}

    constexpr this_type& operator/=(value_type const& rhs) {
		m_a /= rhs;
		m_b /= rhs;
        return *this;
    }

    // dual number and dual number operator
    constexpr this_type operator+(this_type const& rhs) const {
        return this_type{*this} += rhs;
    }

    constexpr this_type operator-(this_type const& rhs) const {
        return this_type{*this} -= rhs;
    }

    constexpr this_type operator*(this_type const& rhs) const {
        return this_type{*this} *= rhs;
    }

    constexpr this_type operator/(this_type const& rhs) const {
        return this_type{*this} /= rhs;
    }

    // dual number and scalar
    constexpr this_type operator+(value_type const& rhs) const {
        return this_type{*this} += rhs;
    }

    constexpr this_type operator-(value_type const& rhs) const {
        return this_type{*this} -= rhs;
    }

    constexpr this_type operator*(value_type const& rhs) const {
        return this_type{*this} *= rhs;
    }

    constexpr this_type operator/(value_type const& rhs) const {
        return this_type{*this} /= rhs;
    }

    // 使当前对偶数变为其倒数
    constexpr void inverse() {
        // d^-1 = 1/a - (b/(a^2))ε
        m_a = value_type(1.0) / m_a;
        m_b /= (-m_a * m_a);
    }

    // 得到当前对偶数的倒数（原对偶数不变）
    constexpr this_type inverted() const {
        this_type copy = *this;
        copy.inverse();
        return copy;
    }

    // 使当前对偶数变为其共轭
    constexpr void conjugate() {
        // a + bε -> a - bε
        m_b = -m_b;
    }

    // 得到当前对偶数的共轭（原对偶数不变）
    constexpr this_type conjugated() const {
        this_type copy = *this;
        copy.conjugate();
        return copy;
    }

    // return m_a(real)
    constexpr value_type a() const {
        return m_a;
    }
    // return m_b(imag)
    constexpr value_type b() const {
        return m_b;
    }

private:
    value_type m_a{0.0};    // in-class initializers
    value_type m_b{0.0};

};

template<typename T>
std::ostream& operator<<(std::ostream& ostream, Dual<T> const& rhs) {
    ostream << rhs.a() << " + " << rhs.b() << "e";
    return ostream;
}

// dual number and scalar
template<typename T>
constexpr Dual<T> operator+(T const& lhs, Dual<T> const& rhs) {
    return Dual<T>{rhs} += lhs;
}

template<typename T>
constexpr Dual<T> operator-(T const& lhs, Dual<T> const& rhs) {
    return Dual<T>{rhs} -= lhs;
}

template<typename T>
constexpr Dual<T> operator*(T const& lhs, Dual<T> const& rhs) {
    return Dual<T>{rhs} *= lhs;
}

template<typename T>
constexpr Dual<T> operator/(T const& lhs, Dual<T> const& rhs) {
    return Dual<T>{rhs} /= lhs;
}

// TODO: 关于基本初等函数的对偶数扩展，eg: sin, cos, exp, log, etc.

}