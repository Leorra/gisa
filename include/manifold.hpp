#pragma once
#include "metric.hpp"
#include "curvature.hpp"
#include <deque>
#include <tuple>
#include <cmath>
#include <algorithm>
#include <string>
#include <ostream>
#include <limits>
#include <vector>

namespace gisa {

enum class IsomorphicState : int {
    FLAT          = 0,
    CURVED_ATTRACT = 1,
    CURVED_REPULS  = 2,
    SINGULARITY    = 3,
    BREAKING       = 4
};

// ============================================================================
// ManifoldPoint
// Units contract:
//   lnV, lnP     — dimensionless natural logs
//   v             — dlnP/dlnV (dimensionless)
//   c             — local liquidity limit (dimensionless)
//   xi            — flat-space geodesic deviation: lnP_obs - lnP_geo
//   xi_curved     — curved-space geodesic deviation (Christoffel-corrected)
//   A_estimate    — rest-frame PRICE = exp(lnP)/γ
//   lnA_estimate  — ln(A_estimate)
// ============================================================================
struct ManifoldPoint {
    T lnV, lnP, v, c, ds2;
    T R, R_raw;
    T gamma, tau_L;
    IsomorphicState state;
    T xi;
    T xi_curved;
    T A_estimate;
    T lnA_estimate;
    T timestamp;
    size_t bar_index;

    ManifoldPoint()
        : lnV(0), lnP(0), v(0), c(1), ds2(0),
          R(0), R_raw(0), gamma(1), tau_L(14),
          state(IsomorphicState::FLAT), xi(0), xi_curved(0),
          A_estimate(1), lnA_estimate(0), timestamp(0), bar_index(0) {}

    ManifoldPoint(T lnV_, T lnP_, T v_, T c_, T ds2_,
                  T R_, T R_raw_, T gamma_, T tau_L_,
                  IsomorphicState st, T xi_, T xi_curved_,
                  T A_est, T lnA_est, T ts, size_t idx)
        : lnV(lnV_), lnP(lnP_), v(v_), c(c_), ds2(ds2_),
          R(R_), R_raw(R_raw_), gamma(gamma_), tau_L(tau_L_),
          state(st), xi(xi_), xi_curved(xi_curved_),
          A_estimate(A_est), lnA_estimate(lnA_est),
          timestamp(ts), bar_index(idx) {}

    bool is_on_geodesic(T thr = 0.01) const { return std::abs(xi) < thr; }
    bool is_singular()   const { return state == IsomorphicState::SINGULARITY; }
    bool is_breaking()   const { return state == IsomorphicState::BREAKING; }
    bool is_isomorphic() const {
        return state == IsomorphicState::FLAT      ||
               state == IsomorphicState::CURVED_ATTRACT ||
               state == IsomorphicState::CURVED_REPULS;
    }
};

// ============================================================================
// ManifoldTracker
// ============================================================================
class ManifoldTracker {
public:
    // History: {c, lnV, lnP, R, ds2, v, gamma, tau_L, xi, R_raw}
    using HistoryEntry = std::tuple<T,T,T,T,T,T,T,T,T,T>;

    explicit ManifoldTracker(
        size_t max_history          = 100,
        T      tau_0                = 14.0,
        T      curvature_threshold  = 1e-6,
        T      singularity_threshold = 0.95,
        T      breaking_ds2_thr     = 0.50,
        T      breaking_xi_thr      = 0.10,
        T      breaking_R_thr       = 1e6,
        T      smoothing_alpha      = 0.30,
        T      c_collapse_thr       = 0.08,
        size_t rolling_window_size  = 7,
        size_t sg_window_size       = 5)
        : max_history_(max_history),
          tau_0_(tau_0),
          curvature_thr_(curvature_threshold),
          singularity_thr_(singularity_threshold),
          breaking_ds2_thr_(breaking_ds2_thr),
          breaking_xi_thr_(breaking_xi_thr),
          breaking_R_thr_(breaking_R_thr),
          alpha_(smoothing_alpha),
          c_collapse_thr_(c_collapse_thr),
          rolling_window_size_(rolling_window_size),
          sg_window_size_(sg_window_size),
          ref_ds2_(0), ref_ds2_set_(false),
          smooth_R_(0), smooth_xi_(0),
          dc_dlnV_(0), dc_dlnP_(0),
          d2c_dlnV2_(0), d2c_dlnP2_(0),
          last_lnV_(0), last_lnP_(0), last_c_(1),
          entry_lnV_(0), entry_lnP_(0),
          entry_lnA_(0), entry_A_(0),
          entry_set_(false),
          total_bars_(0), entry_bars_(0),
          proper_tau_(0), min_xi_(0), max_xi_(0) {}

    // ---- Configuration ----
    void set_curvature_threshold   (T v) { curvature_thr_    = v; }
    void set_singularity_threshold (T v) { singularity_thr_  = v; }
    void set_breaking_ds2_threshold(T v) { breaking_ds2_thr_ = v; }
    void set_breaking_xi_threshold (T v) { breaking_xi_thr_  = v; }
    void set_breaking_R_threshold  (T v) { breaking_R_thr_   = v; }
    void set_smoothing_alpha       (T v) { alpha_ = std::max(0.0, std::min(1.0, v)); }
    void set_c_collapse_threshold  (T v) { c_collapse_thr_   = v; }
    void set_rolling_window_size(size_t w) { rolling_window_size_ = std::max(size_t(3), w); }
    void set_sg_window_size     (size_t w) { sg_window_size_      = std::max(size_t(3), w); }

    // Call once at impulse entry.
    // lnA = ln(VWAP), A = VWAP (raw price), c_first = first bar's c.
    void set_entry(T lnV, T lnP, T A, T lnA, T c_first = -1.0) {
        entry_lnV_ = lnV;  entry_lnP_ = lnP;
        entry_A_   = A;    entry_lnA_ = lnA;
        entry_set_  = true;
        entry_bars_ = 0;
        proper_tau_ = 0.0;
        last_lnV_  = lnV;  last_lnP_ = lnP;
        if (c_first > 0.0) last_c_ = c_first;
        ref_ds2_ = 0;  ref_ds2_set_ = false;
        smooth_R_ = smooth_xi_ = 0;
        min_xi_   = max_xi_    = 0;
        c_history_.clear();
        lnV_history_.clear();
        lnP_history_.clear();
        dc_dlnV_ = dc_dlnP_ = d2c_dlnV2_ = d2c_dlnP2_ = 0;
    }

    // ---- Core update ----
    void update(T lnV, T lnP, T c, T timestamp = 0.0) {
        ++total_bars_;

        const T dlnV = lnV - last_lnV_;
        const T dlnP = lnP - last_lnP_;
        const T v    = (std::abs(dlnV) > 1e-12) ? dlnP/dlnV : 0.0;
        const T ds2  = LorentzMetric::interval(dlnV, dlnP, c);

        if (!ref_ds2_set_ && std::abs(ds2) > 1e-12) {
            ref_ds2_ = ds2;  ref_ds2_set_ = true;
        }

        // ---- Accumulate proper time since entry ----
        if (entry_set_) {
            ++entry_bars_;
            // proper time increment: dtau = dlnV / gamma (from normalisation)
            T gam_tmp = LorentzMetric::gamma(v, c);
            if (gam_tmp > 1e-12 && std::abs(dlnV) > 1e-12)
                proper_tau_ += std::abs(dlnV) / gam_tmp;
        }

        // ---- Curved space-time: rolling c(x) and Savitzky-Golay ----
        c_history_.push_back(last_c_);
        lnV_history_.push_back(lnV);
        lnP_history_.push_back(lnP);
        if (c_history_.size() > rolling_window_size_) {
            c_history_.pop_front();
            lnV_history_.pop_front();
            lnP_history_.pop_front();
        }

        T c_effective = c;
        if (c_history_.size() >= sg_window_size_) {
            std::vector<T> c_win, lnV_win, lnP_win;
            const size_t start = c_history_.size() - sg_window_size_;
            for (size_t i = start; i < c_history_.size(); ++i) {
                c_win.push_back(c_history_[i]);
                lnV_win.push_back(lnV_history_[i]);
                lnP_win.push_back(lnP_history_[i]);
            }
            const size_t mid = sg_window_size_ / 2;
            dc_dlnV_   = SavitzkyGolay::derivative_1(c_win, lnV_win, mid);
            dc_dlnP_   = SavitzkyGolay::derivative_1(c_win, lnP_win, mid);
            d2c_dlnV2_ = SavitzkyGolay::derivative_2(c_win, lnV_win, mid);
            d2c_dlnP2_ = SavitzkyGolay::derivative_2(c_win, lnP_win, mid);
            c_effective = c_win[mid];
        }

        // ---- Christoffel symbols ----
        const ChristoffelSymbols cs =
            ChristoffelSymbols::compute(c_effective, dc_dlnV_, dc_dlnP_);

        // ---- Curved Ricci scalar ----
        const T R_raw = compute_R_curved();
        smooth_R_ = (total_bars_ == 1) ? R_raw
                  : (1.0-alpha_)*smooth_R_ + alpha_*R_raw;
        const T R = smooth_R_;

        // ---- Curved geodesic deviation ----
        const T xi_curved = compute_xi_curved(lnP, v, c_effective, cs);

        // ---- Flat-space geodesic deviation (anchor: entry lnA) ----
        const T xi_raw = entry_set_
            ? lnP - LorentzMetric::ln_price_from_geodesic(entry_lnA_, v, c_effective)
            : 0.0;
        smooth_xi_ = (total_bars_ == 1) ? xi_raw
                   : (1.0-alpha_)*smooth_xi_ + alpha_*xi_raw;
        const T xi = smooth_xi_;

        if (xi < min_xi_) min_xi_ = xi;
        if (xi > max_xi_) max_xi_ = xi;

        // ---- Lorentz factor, tau_L ----
        const T gam     = LorentzMetric::gamma(v, c_effective);
        const T v_ratio = (std::abs(c_effective) > 1e-12)
                        ? std::abs(v/c_effective) : 0.0;
        const T tau_L   = (v_ratio < 1.0)
            ? tau_0_ * std::sqrt(std::max(0.0, 1.0 - v_ratio*v_ratio))
            : 0.0;

        // ---- Rest-frame invariant A ----
        const T A_new = std::exp(lnP) / std::max(gam, 1e-12);
        const T A_est = (current_.A_estimate < 1e-12) ? A_new
                      : 0.9*current_.A_estimate + 0.1*A_new;
        const T lnA_est = std::log(std::max(A_est, 1e-12));

        // ---- State classification ----
        const IsomorphicState st = classify_state(R, v_ratio, xi);

        current_ = ManifoldPoint(lnV, lnP, v, c_effective, ds2,
                                 R, R_raw, gam, tau_L, st,
                                 xi, xi_curved,
                                 A_est, lnA_est,
                                 timestamp, total_bars_);

        history_.push_back({c_effective, lnV, lnP, R, ds2, v, gam, tau_L, xi, R_raw});
        if (history_.size() > max_history_) history_.pop_front();

        const T prev_c = last_c_;
        last_lnV_ = lnV;  last_lnP_ = lnP;  last_c_ = c_effective;

        if (detect_breaking(prev_c)) current_.state = IsomorphicState::BREAKING;

        if (ref_ds2_set_)
            ref_ds2_ = 0.90*ref_ds2_ + 0.10*ds2;
    }

    // ---- Accessors ----
    const ManifoldPoint&            current()  const { return current_; }
    const std::deque<HistoryEntry>& history()  const { return history_; }
    T      reference_ds2() const { return ref_ds2_;    }
    T      smoothed_R()    const { return smooth_R_;   }
    T      smoothed_xi()   const { return smooth_xi_;  }
    size_t total_bars()    const { return total_bars_; }
    bool   has_entry()     const { return entry_set_;  }
    T      entry_lnA()     const { return entry_lnA_;  }
    T      entry_A()       const { return entry_A_;    }
    T      min_xi()        const { return min_xi_;     }
    T      max_xi()        const { return max_xi_;     }
    T      dc_dlnV()       const { return dc_dlnV_;    }
    T      dc_dlnP()       const { return dc_dlnP_;    }
    T      d2c_dlnV2()     const { return d2c_dlnV2_;  }
    T      d2c_dlnP2()     const { return d2c_dlnP2_;  }

    bool is_isomorphic() const { return current_.is_isomorphic(); }
    bool is_flat()        const { return current_.state == IsomorphicState::FLAT; }
    bool is_singularity() const { return current_.state == IsomorphicState::SINGULARITY; }
    bool is_breaking()    const { return current_.state == IsomorphicState::BREAKING; }
    IsomorphicState state() const { return current_.state; }

    // ---- Breaking detection ----
    bool detect_breaking(T prev_c = -1.0) const {
        const T use_prev_c = (prev_c > 0.0) ? prev_c : last_c_;
        const T v_ratio    = (std::abs(current_.c) > 1e-12)
                           ? std::abs(current_.v / current_.c) : 0.0;

        // 1. Velocity singularity
        if (v_ratio > singularity_thr_) return true;

        // 2. c-collapse: sudden liquidity drain
        if (c_collapse_thr_ > 0.0 && use_prev_c > 1e-12 && !history_.empty()) {
            const T dc_rel = (current_.c - use_prev_c) / use_prev_c;
            if (dc_rel < -c_collapse_thr_) return true;
        }

        // 3. Geodesic deviation
        const T xi_thr = adaptive_xi_threshold();
        if (std::abs(current_.xi) > xi_thr) return true;

        // 4. Curvature spike (requires ≥5 bars for stable mean)
        if (history_.size() >= 5) {
            const T R_adaptive  = adaptive_R_threshold();
            const T R_spike_thr = (R_adaptive > curvature_thr_)
                ? breaking_R_thr_ * R_adaptive
                : breaking_R_thr_ * curvature_thr_;
            if (std::abs(current_.R) > R_spike_thr) return true;
        }

        // 5. ds2 sign flip (timelike→spacelike)
        if (ref_ds2_set_ &&
            std::abs(ref_ds2_) > 1e-6 && std::abs(current_.ds2) > 1e-6) {
            if ((current_.ds2 * ref_ds2_) < 0.0) return true;
        }
        return false;
    }

    // ---- Adaptive thresholds ----
    T adaptive_xi_threshold() const {
        if (history_.size() < 3) return breaking_xi_thr_;
        const size_t n = std::min(history_.size(), size_t(10));
        T sum = 0.0;
        for (size_t i = history_.size()-n; i < history_.size(); ++i)
            sum += std::abs(std::get<8>(history_[i]));
        return std::max(breaking_xi_thr_, 3.0 * sum / static_cast<T>(n));
    }

    T adaptive_R_threshold() const {
        if (history_.size() < 3) return curvature_thr_;
        const size_t n = std::min(history_.size(), size_t(10));
        T sum = 0.0;
        for (size_t i = history_.size()-n; i < history_.size(); ++i)
            sum += std::abs(std::get<3>(history_[i]));
        return std::max(curvature_thr_, 0.5 * sum / static_cast<T>(n));
    }

    // ---- Statistics ----
    T average_curvature(size_t w = 0) const {
        if (history_.empty()) return 0.0;
        const size_t n = w ? std::min(w, history_.size()) : history_.size();
        T s = 0.0;
        for (size_t i = history_.size()-n; i < history_.size(); ++i)
            s += std::get<3>(history_[i]);
        return s / static_cast<T>(n);
    }
    T max_curvature() const {
        T mx = 0.0;
        for (const auto& e : history_) {
            T r = std::get<3>(e);
            if (std::abs(r) > std::abs(mx)) mx = r;
        }
        return mx;
    }
    T average_v_ratio(size_t w = 0) const {
        if (history_.empty()) return 0.0;
        const size_t n = w ? std::min(w, history_.size()) : history_.size();
        T s = 0.0;
        for (size_t i = history_.size()-n; i < history_.size(); ++i) {
            const T c = std::get<0>(history_[i]);
            const T v = std::get<5>(history_[i]);
            s += (std::abs(c) > 1e-12) ? std::abs(v/c) : 0.0;
        }
        return s / static_cast<T>(n);
    }
    T average_xi(size_t w = 0) const {
        if (history_.empty()) return 0.0;
        const size_t n = w ? std::min(w, history_.size()) : history_.size();
        T s = 0.0;
        for (size_t i = history_.size()-n; i < history_.size(); ++i)
            s += std::abs(std::get<8>(history_[i]));
        return s / static_cast<T>(n);
    }
    T remaining_tau_L() const { return current_.tau_L; }

    // ---- State strings / colours ----
    static const char* state_string(IsomorphicState s) {
        switch (s) {
            case IsomorphicState::FLAT:           return "FLAT";
            case IsomorphicState::CURVED_ATTRACT: return "ATTRACT";
            case IsomorphicState::CURVED_REPULS:  return "REPULS";
            case IsomorphicState::SINGULARITY:    return "SINGULARITY";
            case IsomorphicState::BREAKING:       return "BREAKING";
            default:                               return "UNKNOWN";
        }
    }
    static const char* state_color(IsomorphicState s) {
        switch (s) {
            case IsomorphicState::FLAT:           return "\033[32m";
            case IsomorphicState::CURVED_ATTRACT: return "\033[36m";
            case IsomorphicState::CURVED_REPULS:  return "\033[33m";
            case IsomorphicState::SINGULARITY:    return "\033[31m";
            case IsomorphicState::BREAKING:       return "\033[35m";
            default:                               return "\033[0m";
        }
    }

    std::string to_string() const {
        auto f = [](T v){ return std::to_string(v); };
        return "{"
            "\"lnV\":"     + f(current_.lnV)           +
            ",\"lnP\":"    + f(current_.lnP)            +
            ",\"v\":"      + f(current_.v)              +
            ",\"c\":"      + f(current_.c)              +
            ",\"ds2\":"    + f(current_.ds2)            +
            ",\"R\":"      + f(current_.R)              +
            ",\"R_raw\":"  + f(current_.R_raw)          +
            ",\"gamma\":"  + f(current_.gamma)          +
            ",\"tau_L\":"  + f(current_.tau_L)          +
            ",\"state\":\"" + state_string(current_.state) + "\"" +
            ",\"xi\":"     + f(current_.xi)             +
            ",\"xi_curved\":" + f(current_.xi_curved)  +
            ",\"A\":"      + f(current_.A_estimate)     +
            ",\"lnA\":"    + f(current_.lnA_estimate)   +
            ",\"bar\":"    + std::to_string(current_.bar_index) +
            "}";
    }

    void reset() {
        history_.clear();
        current_       = ManifoldPoint();
        ref_ds2_       = 0;  ref_ds2_set_ = false;
        smooth_R_      = 0;  smooth_xi_   = 0;
        last_lnV_      = 0;  last_lnP_    = 0;  last_c_ = 1;
        total_bars_    = 0;  entry_set_   = false;
        entry_bars_    = 0;  proper_tau_  = 0.0;
        min_xi_        = 0;  max_xi_      = 0;
        c_history_.clear();
        lnV_history_.clear();
        lnP_history_.clear();
        dc_dlnV_ = dc_dlnP_ = d2c_dlnV2_ = d2c_dlnP2_ = 0;
    }

private:
    // ---- Curved space-time internals ----
    T compute_R_curved() const {
        if (std::abs(last_c_) < 1e-12) return 0.0;
        const T c  = last_c_;
        const T c2 = c*c, c3 = c2*c;
        const T t1 = (2.0/c2) * (d2c_dlnV2_ - d2c_dlnP2_);
        const T t2 = (2.0/c3) * (dc_dlnP_*dc_dlnP_ - dc_dlnV_*dc_dlnV_);
        return t1 + t2;
    }

    // Curved geodesic deviation: lnP_obs - lnP_geo(curved)
    // lnP_geo(curved) is the second-order Taylor expansion of the curved
    // geodesic from the entry point, corrected by the a^1 Christoffel term.
    T compute_xi_curved(T lnP_obs, T v, T /*c*/,
                        const ChristoffelSymbols& cs) const
    {
        if (!entry_set_) return 0.0;
        // Proper time elapsed since entry (approximated as 1 unit per bar)
        // proper_tau_ is accumulated dlnV/gamma since entry — the
        // invariant proper-time elapsed in log-volume coordinates.
        const T tau = proper_tau_;
        const T u0  = 1.0;   // d(lnV)/dtau = 1 at rest
        const T u1  = v;     // d(lnP)/dtau = v
        // Christoffel correction to the lnP geodesic
        const T a1  = -(cs.G1_00*u0*u0 + 2.0*cs.G1_01*u0*u1);
        const T lnP_geo = entry_lnP_ + u1*tau + 0.5*a1*tau*tau;
        return lnP_obs - lnP_geo;
    }

    IsomorphicState classify_state(T R, T v_ratio, T xi) const {
        if (v_ratio > singularity_thr_) return IsomorphicState::SINGULARITY;
        const T R_thr  = adaptive_R_threshold();
        const T xi_thr = adaptive_xi_threshold();
        if (std::abs(R) >= R_thr) {
            if (std::abs(xi) > xi_thr) return IsomorphicState::BREAKING;
            return (R < 0.0) ? IsomorphicState::CURVED_ATTRACT
                             : IsomorphicState::CURVED_REPULS;
        }
        if (std::abs(xi) > xi_thr) return IsomorphicState::BREAKING;
        return IsomorphicState::FLAT;
    }

    // ---- Member variables ----
    size_t max_history_;
    T tau_0_, curvature_thr_, singularity_thr_;
    T breaking_ds2_thr_, breaking_xi_thr_, breaking_R_thr_, alpha_;
    T c_collapse_thr_;
    size_t rolling_window_size_, sg_window_size_;

    // Rolling histories for curved-space estimation
    std::deque<T> c_history_, lnV_history_, lnP_history_;

    T    ref_ds2_;       bool ref_ds2_set_;
    T    smooth_R_,      smooth_xi_;
    T    dc_dlnV_,       dc_dlnP_;
    T    d2c_dlnV2_,     d2c_dlnP2_;
    T    last_lnV_,      last_lnP_,  last_c_;
    T    entry_lnV_,     entry_lnP_, entry_lnA_, entry_A_;
    bool entry_set_;

    ManifoldPoint            current_;
    std::deque<HistoryEntry> history_;
    size_t total_bars_;
    size_t entry_bars_;  // bars elapsed since set_entry
    T      proper_tau_;  // accumulated dlnV/gamma since entry
    T min_xi_, max_xi_;
};

// ---- Free helpers ----
inline T geodesic_distance(const ManifoldPoint& a, const ManifoldPoint& b) {
    const T dlnV = b.lnV-a.lnV, dlnP = b.lnP-a.lnP, c = (a.c+b.c)*0.5;
    const T ds2  = c*c*dlnV*dlnV - dlnP*dlnP;
    return (ds2 >= 0.0) ? std::sqrt(ds2) : -std::sqrt(-ds2);
}
inline T  curvature_difference(const ManifoldPoint& a, const ManifoldPoint& b) { return b.R - a.R; }
inline bool same_isomorphism  (const ManifoldPoint& a, const ManifoldPoint& b) { return a.state == b.state; }

inline std::ostream& operator<<(std::ostream& os, const ManifoldPoint& p) {
    os << "Point(lnV=" << p.lnV << " lnP=" << p.lnP << " v=" << p.v
       << " c=" << p.c << " R=" << p.R << " gamma=" << p.gamma
       << " tau_L=" << p.tau_L << " lnA=" << p.lnA_estimate
       << " xi=" << p.xi << " xi_curved=" << p.xi_curved
       << " state=" << ManifoldTracker::state_string(p.state) << ")";
    return os;
}
inline std::ostream& operator<<(std::ostream& os, IsomorphicState s) {
    return os << ManifoldTracker::state_string(s);
}

inline ManifoldTracker create_default_tracker(
    size_t max_history       = 100,
    T      tau_0             = 14.0,
    T      curvature_thr     = 1e-6,
    T      singularity_thr   = 0.95,
    T      ds2_thr           = 0.50,
    T      xi_thr            = 0.10,
    T      R_thr             = 1e6,
    T      alpha             = 0.30,
    T      c_collapse_thr    = 0.08,
    size_t rolling_window    = 7,
    size_t sg_window         = 5)
{
    return ManifoldTracker(max_history, tau_0, curvature_thr, singularity_thr,
                           ds2_thr, xi_thr, R_thr, alpha, c_collapse_thr,
                           rolling_window, sg_window);
}

} // namespace gisa
