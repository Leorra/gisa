#pragma once
#include <cmath>
#include <algorithm>
#include <array>

namespace gisa {

struct SymmetricTensor2 {
    T g00, g01, g11;
    static constexpr SymmetricTensor2 identity() noexcept { return {1.0, 0.0, 1.0}; }
    T det() const noexcept { return g00*g11 - g01*g01; }
    SymmetricTensor2 inverse() const {
        T d = det();
        if (std::abs(d) < 1e-12) return identity();
        return {g11/d, -g01/d, g00/d};
    }
    std::array<T,2> apply(const std::array<T,2>& v) const {
        return {g00*v[0]+g01*v[1], g01*v[0]+g11*v[1]};
    }
};

struct LorentzMetric {
    T c;
    SymmetricTensor2 tensor() const noexcept { return {-c*c, 0.0, 1.0}; }

    static inline T interval(T dlnV, T dlnP, T c) noexcept {
        return c*c*dlnV*dlnV - dlnP*dlnP;
    }
    static inline T gamma(T v, T c) noexcept {
        if (std::abs(c) < 1e-12) return 1.0;
        T r2 = (v/c)*(v/c);
        if (r2 >= 1.0) r2 = 0.9999*0.9999;
        return 1.0/std::sqrt(1.0 - r2);
    }
    static inline T ln_price_from_geodesic(T lnA, T v, T c) {
        return lnA + std::log(std::max(gamma(v,c), 1e-12));
    }
    static inline T price_from_geodesic(T A, T v, T c) {
        return A * gamma(v, c);
    }
    static bool is_lorentzian(const SymmetricTensor2& g) {
        return (g.g00 < 0.0) && (g.g11 > 0.0) && (g.det() < 0.0);
    }
};

} // namespace gisa
