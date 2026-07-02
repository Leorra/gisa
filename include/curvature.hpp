#pragma once
#include "metric.hpp"
#include <vector>
#include <cmath>
#include <deque>
#include <algorithm>
#include <tuple>

namespace gisa {

	struct Derivatives {
		T d0_c, d1_c, d00_c, d11_c, d01_c;
	};

	// ============================================================================
	// Savitzky-Golay filter — curved space-time derivative estimation
	// ============================================================================
	struct SavitzkyGolay {
		// 5-point quadratic first derivative: [-2,-1,0,1,2]/(10h)
		static T derivative_1(const std::vector<T>& y,
			const std::vector<T>& x, size_t idx)
		{
			const size_t n = y.size();
			if (idx < 2 || idx >= n - 2) return finite_diff_1(y, x, idx);
			T h = (x[idx + 1] - x[idx - 1]) / 2.0;
			if (std::abs(h) < 1e-12) return 0.0;
			return (-2.0 * y[idx - 2] - y[idx - 1] + y[idx + 1] + 2.0 * y[idx + 2]) / (10.0 * h);
		}

		// 5-point quadratic second derivative: [2,-1,-2,-1,2]/(7h^2)
		static T derivative_2(const std::vector<T>& y,
			const std::vector<T>& x, size_t idx)
		{
			const size_t n = y.size();
			if (idx < 2 || idx >= n - 2) return finite_diff_2(y, x, idx);
			T h = (x[idx + 1] - x[idx - 1]) / 2.0;
			if (std::abs(h) < 1e-12) return 0.0;
			return (2.0 * y[idx - 2] - y[idx - 1] - 2.0 * y[idx]
				- y[idx + 1] + 2.0 * y[idx + 2]) / (7.0 * h * h);
		}

	private:
		static T finite_diff_1(const std::vector<T>& y,
			const std::vector<T>& x, size_t idx)
		{
			if (idx == 0 || idx >= y.size() - 1) return 0.0;
			T h = (x[idx + 1] - x[idx - 1]) / 2.0;
			return (std::abs(h) > 1e-12) ? (y[idx + 1] - y[idx - 1]) / (2.0 * h) : 0.0;
		}
		static T finite_diff_2(const std::vector<T>& y,
			const std::vector<T>& x, size_t idx)
		{
			if (idx < 1 || idx >= y.size() - 1) return 0.0;
			T h = (x[idx + 1] - x[idx - 1]) / 2.0;
			return (std::abs(h) > 1e-12)
				? (y[idx + 1] - 2.0 * y[idx] + y[idx - 1]) / (h * h) : 0.0;
		}
	};

	// ============================================================================
	// Christoffel Symbols
	// ============================================================================
	struct ChristoffelSymbols {
		// Non-zero components for g_μν = diag(-c²(x), 1),
		// x⁰ = ln Ṽ, x¹ = ln P̃  (dimensionless log coordinates).
		//
		// Full derivation (thesis §5.3, verified 2026-07-01):
		//   Γ⁰₀₀ =  ∂₀c / c          (from ∂₀g₀₀ = -2c·∂₀c)
		//   Γ⁰₀₁ = Γ⁰₁₀ = ∂₁c / c   (from ∂₁g₀₀ = -2c·∂₁c, via g⁰⁰ = -1/c²)
		//   Γ¹₀₀ =  c · ∂₁c          (from ∂₁g₀₀, via g¹¹ = 1)
		//
		// All other components vanish identically:
		//   Γ⁰₁₁ = 0  (g₁₀ = 0, g₁₁ = 1 = const  →  no ∂₀g₁₁ contribution)
		//   Γ¹₀₁ = Γ¹₁₀ = 0  (g₁₁ = 1 = const  →  ∂₀g₁₁ = 0)
		//   Γ¹₁₁ = 0  (g₁₁ = 1 = const)
		//
		// Geodesic equation for x¹ = ln P̃ collapses to:
		//   ẍ¹ + Γ¹₀₀ (ẋ⁰)² = 0   i.e.   d²(lnP)/dτ² + c·∂₁c·(dlnV/dτ)² = 0
		T G0_00;   // ∂₀c / c
		T G0_01;   // ∂₁c / c   (= G0_10 by symmetry)
		T G1_00;   // c · ∂₁c
		// G0_11, G1_01, G1_10, G1_11 are identically zero — not stored.

		static ChristoffelSymbols compute(T c, T dc_dlnV, T dc_dlnP) noexcept {
			if (std::abs(c) < 1e-12) return { 0.0, 0.0, 0.0 };
			return {
				dc_dlnV / c,   // G0_00
				dc_dlnP / c,   // G0_01
				c * dc_dlnP    // G1_00
			};
		}
	};

	// ============================================================================
	// Ricci Scalar — flat and curved space-time
	// ============================================================================
	struct RicciScalar {
		// R = (2/c^2)(d00_c - d11_c) + (2/c^3)((d1_c)^2 - (d0_c)^2)
		static inline T compute(T c, const Derivatives& d) noexcept {
			if (std::abs(c) < 1e-12) return 0.0;
			T c2 = c * c, c3 = c2 * c;
			T result = (2.0 / c2) * (d.d00_c - d.d11_c)
				+ (2.0 / c3) * (d.d1_c * d.d1_c - d.d0_c * d.d0_c);
			return (std::abs(result) < 1e-8) ? 0.0 : result;
		}

		static T finite_diff(const std::vector<T>& y,
			const std::vector<T>& x,
			size_t idx, int order)
		{
			const size_t n = y.size();
			if (idx < 1 || idx >= n - 1) return 0.0;
			T hf = x[idx + 1] - x[idx], hb = x[idx] - x[idx - 1];
			if (std::abs(hf) < 1e-12 || std::abs(hb) < 1e-12) return 0.0;
			if (order == 1)
				return (y[idx + 1] - y[idx - 1]) / (hf + hb);
			return 2.0 * (hb * y[idx + 1] - (hb + hf) * y[idx] + hf * y[idx - 1])
				/ (hf * hb * (hf + hb));
		}

		static T compute_numeric(const std::vector<T>& c_vals,
			const std::vector<T>& x0,
			const std::vector<T>& x1,
			size_t idx)
		{
			const size_t n = c_vals.size();
			if (n < 3 || idx < 1 || idx >= n - 1) return 0.0;
			Derivatives d {
				finite_diff(c_vals, x0, idx, 1),
				finite_diff(c_vals, x1, idx, 1),
				finite_diff(c_vals, x0, idx, 2),
				finite_diff(c_vals, x1, idx, 2),
				0.0
			};
			return compute(c_vals[idx], d);
		}

		// 10-tuple history: {c,lnV,lnP,R,ds2,v,gamma,tau_L,xi,R_raw}
		static T compute_from_history(
			const std::deque<std::tuple<T, T, T, T, T, T, T, T, T, T>>& history,
			size_t window = 5)
		{
			return compute_impl(history, window,
				[](const auto& e) { return std::get<0>(e); },
				[](const auto& e) { return std::get<1>(e); },
				[](const auto& e) { return std::get<2>(e); });
		}

		// 9-tuple history (backward compat)
		static T compute_from_history(
			const std::deque<std::tuple<T, T, T, T, T, T, T, T, T>>& history,
			size_t window = 5)
		{
			return compute_impl(history, window,
				[](const auto& e) { return std::get<0>(e); },
				[](const auto& e) { return std::get<1>(e); },
				[](const auto& e) { return std::get<2>(e); });
		}

	private:
		template<typename Deque, typename FC, typename FV, typename FP>
		static T compute_impl(const Deque& history, size_t window,
			FC get_c, FV get_lnV, FP get_lnP)
		{
			const size_t n = history.size();
			if (n < 3) return 0.0;
			window = std::min(window, n);

			std::vector<T> cv, vv, pv;
			cv.reserve(window); vv.reserve(window); pv.reserve(window);
			for (size_t i = n - window; i < n; ++i) {
				cv.push_back(get_c(history[i]));
				vv.push_back(get_lnV(history[i]));
				pv.push_back(get_lnP(history[i]));
			}

			size_t mid = (window >= 3) ? window - 2 : 1;
			T result = compute_numeric(cv, vv, pv, mid);

			// NOTE: c-acceleration fallback removed. When the formal SG/finite-
			// difference estimate is near-zero it means the window sees no
			// significant curvature — returning 0 is correct. The fallback
			// substituted raw dc/d2c values which, on non-monotone c series,
			// produced spurious multi-thousand R spikes.
			return result;
		}
	};

} // namespace gisa