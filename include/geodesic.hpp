#pragma once
#include "manifold.hpp"
#include <cmath>
#include <algorithm>

namespace gisa {

enum class ExitSignal : int {
    HOLD        = 0,
    PREPARE_EXIT = 1,
    SELL        = 2
};

struct TrackerStats {
    T A_drift;       // log-space: |lnP - lnA_entry|
    T R_growth;      // current Ricci scalar
    T xi_norm;       // |xi| (flat-space)
    T xi_curved_norm;// |xi_curved|
    T tau_remaining;
    T v_ratio;
    T price_drift;   // price-space: |exp(lnP) - A_entry|
};

class GeodesicTracker {
public:
    explicit GeodesicTracker(T entry_lnA = 0.0, T entry_A = 0.0)
        : entry_lnA_(entry_lnA), entry_A_(entry_A),
          entry_lnV_(0), tau_elapsed_(0), max_xi_(0),
          initialized_(false), entry_set_(false) {}

    void set_entry(T lnV, T lnA, T A) {
        entry_lnV_   = lnV;
        entry_lnA_   = lnA;
        entry_A_     = A;
        initialized_ = true;
        entry_set_   = true;
        tau_elapsed_ = 0;
        max_xi_      = 0;
    }

    void set_entry_lnA(T lnA, T A) {
        entry_lnA_   = lnA;
        entry_A_     = A;
        initialized_ = true;
    }

    ExitSignal evaluate(const ManifoldPoint& pt) {
        if (!initialized_) {
            entry_lnA_ = pt.lnA_estimate;
            entry_A_   = pt.A_estimate;
            initialized_ = true;
        }

        const T A_drift  = std::abs(pt.lnP - entry_lnA_);
        const T v_ratio  = (std::abs(pt.c) > 1e-12)
                         ? std::abs(pt.v / pt.c) : 0.0;
        const T xi_norm  = std::abs(pt.xi);
        const T xic_norm = std::abs(pt.xi_curved);
        const bool R_active = (std::abs(pt.R) > 1e-3);

        if (xi_norm > max_xi_) max_xi_ = xi_norm;

        tau_elapsed_ += 1.0;
        const T tau_rem =
            14.0 * std::sqrt(std::max(0.0, 1.0 - v_ratio*v_ratio))
            - tau_elapsed_;

        const bool sing     = (pt.state == IsomorphicState::SINGULARITY);
        const bool breaking = (pt.state == IsomorphicState::BREAKING);

        // ---- Immediate SELL ----
        if (sing || breaking)                       return ExitSignal::SELL;
        if (R_active && v_ratio > 0.80)             return ExitSignal::SELL;
        if (xi_norm  > 0.50 && v_ratio > 0.60)     return ExitSignal::SELL;
        if (xic_norm > 0.40 && v_ratio > 0.60)     return ExitSignal::SELL;
        if (A_drift  > 0.08)                        return ExitSignal::SELL;

        // ---- Prepare exit ----
        if (R_active && v_ratio > 0.50)             return ExitSignal::PREPARE_EXIT;
        if (A_drift  > 0.03)                        return ExitSignal::PREPARE_EXIT;
        if (tau_rem  < 3.0)                         return ExitSignal::PREPARE_EXIT;
        if (v_ratio  > 0.70)                        return ExitSignal::PREPARE_EXIT;
        if (xic_norm > 0.20)                        return ExitSignal::PREPARE_EXIT;

        return ExitSignal::HOLD;
    }

    TrackerStats get_stats(const ManifoldPoint& pt) const {
        const T v_ratio = (std::abs(pt.c) > 1e-12)
                        ? std::abs(pt.v / pt.c) : 0.0;
        const T tau_rem = (v_ratio < 1.0)
            ? 14.0*std::sqrt(std::max(0.0, 1.0-v_ratio*v_ratio)) - tau_elapsed_
            : 0.0;
        return {
            std::abs(pt.lnP - entry_lnA_),
            pt.R,
            std::abs(pt.xi),
            std::abs(pt.xi_curved),
            tau_rem,
            v_ratio,
            std::abs(std::exp(pt.lnP) - entry_A_)
        };
    }

    void reset() {
        tau_elapsed_ = max_xi_ = 0;
        initialized_ = entry_set_ = false;
    }

    T max_xi()    const { return max_xi_;    }
    T entry_lnA() const { return entry_lnA_; }
    T entry_A()   const { return entry_A_;   }

private:
    T    entry_lnA_, entry_A_, entry_lnV_;
    T    tau_elapsed_, max_xi_;
    bool initialized_, entry_set_;
};

} // namespace gisa
