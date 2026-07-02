// main.cpp — GISA Engine v2 integration test
// Three scenarios: flat geodesic, curved space-time, phase explosion.

#include "gisa.hpp"
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <string>

using namespace gisa;

static const char* RESET = "\033[0m";
static const char* GREEN = "\033[32m";
static const char* YELLOW = "\033[33m";
static const char* RED = "\033[31m";

static const char* sig_color(ExitSignal s) {
	return s == ExitSignal::SELL ? RED
		: s == ExitSignal::PREPARE_EXIT ? YELLOW
		: GREEN;
}
static const char* sig_str(ExitSignal s) {
	return s == ExitSignal::SELL ? ">>> SELL"
		: s == ExitSignal::PREPARE_EXIT ? ">>> PREP"
		: "HOLD    ";
}

struct Bar { T lnV, lnP, c; };

static void run(const std::string& title, T vwap,
	const std::vector<Bar>& bars,
	T c_collapse_thr = 0.05)
{
	const T   lnVWAP = std::log(vwap);
	const int W = 165;

	ManifoldTracker tracker = create_default_tracker(
		200,            // max_history
		14.0,           // tau_0
		1e-6,           // curvature_threshold
		0.95,           // singularity_threshold
		0.50,           // breaking_ds2_threshold
		0.25,           // breaking_xi_threshold
		1e6,            // breaking_R_threshold
		0.25,           // smoothing_alpha
		c_collapse_thr, // c_collapse_threshold
		7,              // rolling_window_size  (curved c(x) estimation)
		5               // sg_window_size       (Savitzky-Golay)
	);
	GeodesicTracker geo;

	const T seed_lnV = bars.empty() ? 0.0 : bars[0].lnV;
	const T seed_c = bars.empty() ? 1.0 : bars[0].c;
	tracker.set_entry(seed_lnV, lnVWAP, vwap, lnVWAP, seed_c);
	geo.set_entry(seed_lnV, lnVWAP, vwap);

	std::cout << "\n" << std::string(W, '=') << "\n"
		<< "  " << title << "\n"
		<< "  VWAP=" << vwap << "  lnVWAP=" << lnVWAP << "\n"
		<< std::string(W, '=') << "\n"
		<< std::setw(4) << "bar"
		<< std::setw(10) << "lnV"
		<< std::setw(9) << "dlnP"
		<< std::setw(10) << "price"
		<< std::setw(9) << "c"
		<< std::setw(11) << "R"
		<< std::setw(8) << "v/c"
		<< std::setw(9) << "xi"
		<< std::setw(10) << "xi_curved"
		<< std::setw(9) << "lnA_off"
		<< std::setw(9) << "tau_L"
		<< std::setw(12) << "state"
		<< std::setw(10) << "signal"
		<< "\n" << std::string(W, '-') << "\n";

	for (size_t i = 0; i < bars.size(); ++i) {
		tracker.update(bars[i].lnV, bars[i].lnP, bars[i].c);
		const auto& pt = tracker.current();
		const auto  sig = geo.evaluate(pt, tracker);
		const T     vc = (std::abs(pt.c) > 1e-12)
			? std::abs(pt.v / pt.c) : 0.0;

		std::cout << std::fixed << std::setprecision(5)
			<< std::setw(4) << i
			<< std::setw(10) << bars[i].lnV
			<< std::setw(9) << (bars[i].lnP - lnVWAP)
			<< std::setw(10) << std::exp(bars[i].lnP)
			<< std::setw(9) << pt.c
			<< std::setw(11) << pt.R
			<< std::setw(8) << vc
			<< std::setw(9) << pt.xi
			<< std::setw(10) << pt.xi_curved
			<< std::setw(9) << (pt.lnA_estimate - lnVWAP)
			<< std::setw(9) << pt.tau_L
			<< ManifoldTracker::state_color(tracker.state())
			<< std::setw(12) << ManifoldTracker::state_string(tracker.state())
			<< RESET
			<< sig_color(sig)
			<< std::setw(10) << sig_str(sig)
			<< RESET << "\n";
	}

	std::cout << std::string(W, '-') << "\n"
		<< "  state=" << ManifoldTracker::state_string(tracker.state())
		<< "  R=" << tracker.current().R
		<< "  xi=" << tracker.current().xi
		<< "  xi_curved=" << tracker.current().xi_curved
		<< "  tau_L=" << tracker.current().tau_L
		<< "  max_xi=" << geo.max_xi()
		<< "  avg_v/c=" << tracker.average_v_ratio()
		<< "  avg_R=" << tracker.average_curvature()
		<< "\n" << std::string(W, '=') << "\n";
}

// Build bars on the Lorentz geodesic so that observed v_obs = v_target[i].
static std::vector<Bar> lorentz_bars(T lnA, T c_default,
	const std::vector<T>& v_seq,
	const std::vector<T>& c_seq = {})
{
	std::vector<Bar> out;
	T lnV = 0.0, prev_lnP = lnA;
	for (size_t i = 0; i < v_seq.size(); ++i) {
		const T vt = v_seq[i];
		const T ci = c_seq.empty() ? c_default : c_seq[i];
		const T r2 = std::min((vt / ci) * (vt / ci), 0.9999 * 0.9999);
		const T gam = 1.0 / std::sqrt(1.0 - r2);
		const T lnP = lnA + std::log(gam);
		const T dlnP = lnP - prev_lnP;
		T dlnV = (std::abs(vt) > 1e-9) ? dlnP / vt : 0.08;
		if (dlnV < 0.005) dlnV = 0.005;
		out.push_back({ lnV, lnP, ci });
		prev_lnP = lnP;
		lnV += dlnV;
	}
	return out;
}

int main() {
	// ── Scenario A: stable c, flat geodesic ─────────────────────────────────
	{
		const T vwap = 0.21638;
		const T lnA = std::log(vwap);
		const T c = 0.27797;
		constexpr int N = 20;
		std::vector<T> vt(N), cs(N, c);
		for (int i = 0; i < N; ++i) vt[i] = 0.220 * i / (N - 1);
		run("Scenario A — Stable c=0.27797, 20-bar Lorentz geodesic",
			vwap, lorentz_bars(lnA, c, vt, cs), 0.05);
	}

	// ── Scenario B: gentle c decay, curved space-time ───────────────────────
	{
		const T vwap = 0.21638;
		const T lnA = std::log(vwap);
		const std::vector<T> vt = {
			0.000, 0.020, 0.040, 0.060, 0.080,
			0.100, 0.120, 0.140, 0.160, 0.180, 0.195, 0.210
		};
		const std::vector<T> cv = {
			0.278, 0.277, 0.276, 0.274, 0.272,
			0.270, 0.268, 0.265, 0.261, 0.257, 0.252, 0.240
		};
		run("Scenario B — Gentle c decay (−14%), curved space-time",
			vwap, lorentz_bars(lnA, 0.278, vt, cv), 0.02);
	}

	// ── Scenario C: phase explosion, c −61% ─────────────────────────────────
	{
		const T vwap = 0.21638;
		const T lnVWAP = std::log(vwap);
		const std::vector<T> cs = {
			0.27797, 0.27000, 0.26000, 0.24800, 0.23400,
			0.21800, 0.20000, 0.18000, 0.15800, 0.13400, 0.10800
		};
		const std::vector<T> dlnP = {
			0.0000, 0.0020, 0.0050, 0.0090, 0.0150,
			0.0230, 0.0330, 0.0450, 0.0590, 0.0750, 0.0930
		};
		std::vector<Bar> bars;
		bars.reserve(cs.size());
		for (size_t i = 0; i < cs.size(); ++i)
			bars.push_back({ static_cast<T>(i) * 0.1, lnVWAP + dlnP[i], cs[i] });
		run("Scenario C — Phase explosion: c −61%, v/c → 1.67",
			vwap, bars, 0.03);
	}

	return 0;
}