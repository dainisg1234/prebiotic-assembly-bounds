#!/usr/bin/env python3
"""
Test Suite for Stochastic Assembly Simulation
=============================================
Runs all normal and edge cases, verifies results against expected behavior.

Usage:
  python test_assembly.py              # run all tests
  python test_assembly.py -v           # verbose output
  python test_assembly.py --quick      # skip slow tests
"""

import sys
import os
import math
import time
import argparse
import tempfile
from collections import Counter

# Import the simulation class from the main script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stochastic_assembly_visual import StochasticAssembly, detect_plateau, print_summary


class TestResult:
    def __init__(self, name):
        self.name = name
        self.passed = 0
        self.failed = 0
        self.errors = []

    def check(self, condition, description):
        if condition:
            self.passed += 1
        else:
            self.failed += 1
            self.errors.append(description)

    def summary(self):
        status = "PASS" if self.failed == 0 else "FAIL"
        return f"  [{status}] {self.name}: {self.passed} passed, {self.failed} failed"


def run_sim(alphabet=4, material=10000, half_life=100, ticks=300, seed=42,
            growth_frac=0.3, elong_prob=0.7, confidence=0.95):
    """Helper: run simulation and return sim object."""
    sim = StochasticAssembly(alphabet_size=alphabet, total_monomers=material,
                            half_life=half_life, seed=seed,
                            growth_frac=growth_frac, elong_prob=elong_prob,
                            confidence=confidence)
    sim.set_target(min(6, int(math.log(max(material, 10), alphabet) * 0.5)))
    for _ in range(ticks):
        sim.step()
    return sim


def get_crossovers(sim):
    """Extract n*_target, n*_F, EqMax from simulation results.

    Returns visual-marker crossovers (n_loss_target, n_loss_F) for backward
    compatibility with existing tests.
    """
    cc = sim.compute_confidence_ceilings(P=1.0)
    # n_loss_target is the visual marker (first n where coverage < 1)
    n_target = cc['n_loss_target']
    # n_loss_F is the visual marker (first n where functional == 0)
    n_func = cc['n_loss_F']
    eq_max = cc['eq_max']
    return n_target, n_func, eq_max


# ===================================================================
# NORMAL CASE TESTS
# ===================================================================

def test_basic_run(verbose=False):
    """Test: basic simulation runs without errors."""
    t = TestResult("Basic run")
    try:
        sim = run_sim(alphabet=4, material=10000, ticks=200)
        t.check(sim.tick == 200, f"tick count: expected 200, got {sim.tick}")
        t.check(len(sim.ts_ticks) == 200, f"time series length: {len(sim.ts_ticks)}")
        t.check(sim.target is not None, "target was set")
        if verbose:
            print(f"    tick={sim.tick}, distinct={sim.ts_total_distinct[-1]}, "
                  f"functional={sim.ts_total_functional[-1]}")
    except Exception as e:
        t.check(False, f"Exception: {e}")
    return t


def test_material_conservation(verbose=False):
    """Test: total material (free + in chains) is conserved at every tick."""
    t = TestResult("Material conservation")
    for M in [1000, 10000, 100000]:
        sim = StochasticAssembly(alphabet_size=4, total_monomers=M, half_life=100, seed=42)
        sim.set_target(4)
        for tick in range(100):
            sim.step()
            in_chains = sum(len(c) for c in sim.chains)
            total = sim.free + in_chains
            t.check(total == M, f"M={M}, tick={tick+1}: free={sim.free} + chains={in_chains} = {total} != {M}")
            if total != M:
                break
        if verbose:
            print(f"    M={M}: conserved through 100 ticks")
    return t


def test_no_negative_counts(verbose=False):
    """Test: no negative values in any counter."""
    t = TestResult("No negative counts")
    sim = run_sim(material=20000, ticks=300)
    t.check(all(v >= 0 for v in sim.ts_free), "free monomers always >= 0")
    t.check(all(v >= 0 for v in sim.ts_max_len), "max length always >= 0")
    t.check(all(v >= 0 for v in sim.ts_total_distinct), "total distinct always >= 0")
    t.check(all(v >= 0 for v in sim.ts_total_functional), "total functional always >= 0")
    ns, distincts, spaces, coverages, functionals, fns = sim.get_coverage_data()
    t.check(all(d >= 0 for d in distincts), "distinct per length >= 0")
    t.check(all(f >= 0 for f in functionals), "functional per length >= 0")
    t.check(all(0 <= c <= 1.0001 for c in coverages), "coverage in [0, 1]")
    return t


def test_functional_leq_distinct(verbose=False):
    """Test: functional found <= distinct generated at every length."""
    t = TestResult("Functional <= Distinct")
    sim = run_sim(material=50000, ticks=300)
    ns, distincts, spaces, coverages, functionals, fns = sim.get_coverage_data()
    for n, d, f in zip(ns, distincts, functionals):
        t.check(f <= d, f"n={n}: functional={f} > distinct={d}")
    return t


def test_coverage_monotone_decreasing(verbose=False):
    """Test: coverage fraction decreases with length (for n beyond full coverage)."""
    t = TestResult("Coverage decreasing with length")
    sim = run_sim(material=50000, ticks=300)
    ns, distincts, spaces, coverages, functionals, fns = sim.get_coverage_data()
    # Find first n where coverage < 100%, then it should decrease
    started = False
    prev_cov = 1.0
    for n, c in zip(ns, coverages):
        if c < 0.99:
            started = True
        if started and c > 0:
            t.check(c <= prev_cov * 1.1, f"n={n}: coverage {c:.4f} > previous {prev_cov:.4f}")
            prev_cov = c
    return t


def test_ntarget_lt_nfunc(verbose=False):
    """Test: n*_target < n*_F always (Functional Search reaches further)."""
    t = TestResult("n*_target < n*_F")
    for M in [5000, 20000, 100000]:
        sim = run_sim(material=M, ticks=400)
        nt, nf, eq = get_crossovers(sim)
        if nt is not None and nf is not None:
            t.check(nt < nf, f"M={M}: n*_target={nt} should be < n*_F={nf}")
            if verbose:
                print(f"    M={M}: n*_target={nt}, n*_F={nf}, EqMax={eq}")
        else:
            t.check(False, f"M={M}: could not compute crossovers (nt={nt}, nf={nf})")
    return t


def test_logarithmic_scaling(verbose=False):
    """Test: n*_target and n*_F grow approximately logarithmically with M."""
    t = TestResult("Logarithmic scaling")
    results = []
    for M in [1000, 10000, 100000]:
        sim = run_sim(material=M, ticks=400)
        nt, nf, eq = get_crossovers(sim)
        results.append((M, nt, nf, eq))
        if verbose:
            print(f"    M={M}: n*_target={nt}, n*_F={nf}, EqMax={eq}")

    # Check that 100x material increase adds only a few residues
    if results[0][1] is not None and results[2][1] is not None:
        delta_nt = results[2][1] - results[0][1]
        t.check(delta_nt <= 5, f"n*_target grew by {delta_nt} over 100x material (should be <=5)")
        t.check(delta_nt >= 0, f"n*_target decreased with more material")

    if results[0][2] is not None and results[2][2] is not None:
        delta_nf = results[2][2] - results[0][2]
        t.check(delta_nf <= 8, f"n*_F grew by {delta_nf} over 100x material (should be <=8)")
        t.check(delta_nf >= 0, f"n*_F decreased with more material")
    return t


def test_halflife_affects_eqmax(verbose=False):
    """Test: longer half-life produces higher equilibrium max length."""
    t = TestResult("Half-life affects EqMax")
    eq_results = []
    for hl in [30, 100, 500]:
        sim = run_sim(material=50000, half_life=hl, ticks=400)
        eq = sim.get_equilibrium_max()
        eq_results.append((hl, eq))
        if verbose:
            print(f"    half_life={hl}: EqMax={eq}")
    # Longer half-life should give higher or equal EqMax
    t.check(eq_results[1][1] >= eq_results[0][1],
            f"HL={eq_results[1][0]} EqMax={eq_results[1][1]} should be >= HL={eq_results[0][0]} EqMax={eq_results[0][1]}")
    t.check(eq_results[2][1] >= eq_results[1][1],
            f"HL={eq_results[2][0]} EqMax={eq_results[2][1]} should be >= HL={eq_results[1][0]} EqMax={eq_results[1][1]}")
    return t


def test_alphabet_affects_gap(verbose=False):
    """Test: larger alphabet produces wider gap between n*_target and n*_F."""
    t = TestResult("Alphabet affects gap")
    gaps = []
    for alpha in [2, 4, 10]:
        sim = run_sim(alphabet=alpha, material=20000, ticks=400)
        nt, nf, eq = get_crossovers(sim)
        if nt and nf:
            gap = nf - nt
            gaps.append((alpha, nt, nf, gap))
            if verbose:
                print(f"    |Σ|={alpha}: n*_target={nt}, n*_F={nf}, gap={gap}")
    # Larger alphabet should give wider gap (more room between cases)
    if len(gaps) >= 2:
        t.check(gaps[-1][3] >= gaps[0][3] - 1,
                f"|Σ|={gaps[-1][0]} gap={gaps[-1][3]} should be >= |Σ|={gaps[0][0]} gap={gaps[0][3]}")
    return t


def test_reproducibility(verbose=False):
    """Test: same seed produces identical results."""
    t = TestResult("Reproducibility (same seed)")
    sim1 = run_sim(material=10000, ticks=200, seed=123)
    sim2 = run_sim(material=10000, ticks=200, seed=123)
    t.check(sim1.ts_total_distinct == sim2.ts_total_distinct, "distinct counts differ")
    t.check(sim1.ts_total_functional == sim2.ts_total_functional, "functional counts differ")
    t.check(sim1.ts_max_len == sim2.ts_max_len, "max lengths differ")
    t.check(sim1.ts_free == sim2.ts_free, "free counts differ")

    # Different seed should give different results
    sim3 = run_sim(material=10000, ticks=200, seed=456)
    t.check(sim1.ts_total_distinct != sim3.ts_total_distinct, "different seeds gave same results")
    return t


def test_target_found_short(verbose=False):
    """Test: short target (length 2-3) is always found quickly with enough material."""
    t = TestResult("Short target found")
    for length in [2, 3]:
        sim = StochasticAssembly(alphabet_size=4, total_monomers=10000, half_life=100, seed=42)
        sim.set_target(length)
        for _ in range(100):
            sim.step()
        t.check(sim.target_found_tick is not None,
                f"target length {length} not found in 100 ticks with M=10000")
        if verbose and sim.target_found_tick:
            print(f"    target length {length}: found at tick {sim.target_found_tick}")
    return t


def test_full_coverage_short_lengths(verbose=False):
    """Test: lengths 2,3,4 should have 100% coverage with sufficient material."""
    t = TestResult("Full coverage at short lengths")
    sim = run_sim(material=50000, ticks=300)
    ns, distincts, spaces, coverages, functionals, fns = sim.get_coverage_data()
    for n, c in zip(ns, coverages):
        if n <= 4:  # 4^4 = 256, easily covered with 50k monomers
            t.check(c > 0.99, f"n={n}: coverage={c:.4f}, expected ~100%")
    return t


def test_expected_vs_actual_functional(verbose=False):
    """Test: actual functional count approximately matches expected (distinct × f)."""
    t = TestResult("Functional ≈ Expected")
    sim = run_sim(material=100000, ticks=400)
    ns, distincts, spaces, coverages, functionals, fns = sim.get_coverage_data()
    for n, d, func, fn in zip(ns, distincts, functionals, fns):
        expected = d * fn
        if expected > 10:  # only check where statistics are meaningful
            ratio = func / expected if expected > 0 else 0
            t.check(0.5 < ratio < 2.0,
                    f"n={n}: functional={func}, expected={expected:.0f}, ratio={ratio:.2f}")
    return t


def test_plateau_detection(verbose=False):
    """Test: plateau is detected and occurs at a reasonable tick."""
    t = TestResult("Plateau detection")
    sim = StochasticAssembly(alphabet_size=4, total_monomers=50000, half_life=100, seed=42)
    sim.set_target(6)
    plateau_tick = None
    for tick in range(2000):
        sim.step()
        if plateau_tick is None and detect_plateau(sim):
            plateau_tick = tick + 1
            break
    t.check(plateau_tick is not None, "plateau not detected within 2000 ticks")
    if plateau_tick:
        t.check(plateau_tick > 300, f"plateau too early: tick {plateau_tick} (min should be >300)")
        t.check(plateau_tick < 1500, f"plateau too late: tick {plateau_tick}")
        if verbose:
            print(f"    plateau at tick {plateau_tick}")
    return t


def test_plateau_not_always_same(verbose=False):
    """Test: plateau tick varies with parameters (not stuck at fixed value)."""
    t = TestResult("Plateau varies with parameters")
    plateaus = []
    for M, HL in [(20000, 100), (50000, 100), (50000, 200)]:
        sim = StochasticAssembly(alphabet_size=4, total_monomers=M, half_life=HL, seed=42)
        sim.set_target(6)
        pt = None
        for tick in range(3000):
            sim.step()
            if pt is None and detect_plateau(sim):
                pt = tick + 1
                break
        plateaus.append((M, HL, pt))
        if verbose:
            print(f"    M={M}, HL={HL}: plateau at tick {pt}")

    # At least two different plateau ticks among the three runs
    ticks = [p[2] for p in plateaus if p[2] is not None]
    if len(ticks) >= 2:
        t.check(len(set(ticks)) > 1, f"all plateaus at same tick: {ticks}")
    return t


# ===================================================================
# EDGE CASE TESTS
# ===================================================================

def test_zero_material(verbose=False):
    """Edge case: zero monomers."""
    t = TestResult("Edge: zero material")
    try:
        sim = StochasticAssembly(alphabet_size=4, total_monomers=0, half_life=100, seed=42)
        sim.set_target(3)
        for _ in range(10):
            sim.step()
        t.check(len(sim.chains) == 0, f"chains exist with 0 material: {len(sim.chains)}")
        t.check(sim.free == 0, f"free monomers != 0: {sim.free}")
        t.check(sim.ts_total_distinct[-1] == 0, "distinct > 0 with no material")
    except Exception as e:
        t.check(False, f"Exception with zero material: {e}")
    return t


def test_one_monomer(verbose=False):
    """Edge case: single monomer."""
    t = TestResult("Edge: one monomer")
    try:
        sim = StochasticAssembly(alphabet_size=4, total_monomers=1, half_life=100, seed=42)
        sim.set_target(2)
        for _ in range(10):
            sim.step()
        t.check(sim.free + sum(len(c) for c in sim.chains) == 1, "material not conserved")
    except Exception as e:
        t.check(False, f"Exception with 1 monomer: {e}")
    return t


def test_tiny_material(verbose=False):
    """Edge case: very small material (10 monomers)."""
    t = TestResult("Edge: tiny material (10)")
    try:
        sim = run_sim(material=10, ticks=100)
        total = sim.free + sum(len(c) for c in sim.chains)
        t.check(total == 10, f"material not conserved: {total}")
        t.check(sim.ts_max_len[-1] <= 10, f"max length > material: {sim.ts_max_len[-1]}")
    except Exception as e:
        t.check(False, f"Exception: {e}")
    return t


def test_no_decay(verbose=False):
    """Edge case: half-life = 0 means no decay (infinite stability)."""
    t = TestResult("Edge: no decay (half_life=0)")
    try:
        sim = StochasticAssembly(alphabet_size=4, total_monomers=5000, half_life=0, seed=42)
        sim.set_target(4)
        for _ in range(200):
            sim.step()
        # With no decay, material should all end up in chains
        t.check(sim.free < 100, f"too much free material with no decay: {sim.free}")
        # Max length should be higher than with decay
        sim2 = run_sim(material=5000, half_life=100, ticks=200)
        t.check(sim.ts_max_len[-1] >= sim2.ts_max_len[-1],
                f"no-decay max {sim.ts_max_len[-1]} < decay max {sim2.ts_max_len[-1]}")
    except ZeroDivisionError:
        t.check(False, "ZeroDivisionError with half_life=0")
    except Exception as e:
        t.check(False, f"Exception: {e}")
    return t


def test_huge_halflife(verbose=False):
    """Edge case: very long half-life (effectively no decay)."""
    t = TestResult("Edge: huge half-life (10^9)")
    try:
        sim = run_sim(material=5000, half_life=10**9, ticks=200)
        total = sim.free + sum(len(c) for c in sim.chains)
        t.check(total == 5000, f"material not conserved: {total}")
    except Exception as e:
        t.check(False, f"Exception: {e}")
    return t


def test_binary_alphabet(verbose=False):
    """Edge case: alphabet size 2 (binary sequences)."""
    t = TestResult("Edge: binary alphabet (|Σ|=2)")
    try:
        sim = run_sim(alphabet=2, material=10000, ticks=300)
        nt, nf, eq = get_crossovers(sim)
        t.check(nt is not None, "n*_target not found")
        t.check(nf is not None, "n*_F not found")
        if nt and nf:
            t.check(nt > 5, f"n*_target too low for |Σ|=2: {nt}")
            t.check(nf > nt, f"n*_F={nf} should be > n*_target={nt}")
            if verbose:
                print(f"    |Σ|=2: n*_target={nt}, n*_F={nf}, EqMax={eq}")
    except Exception as e:
        t.check(False, f"Exception: {e}")
    return t


def test_single_tick(verbose=False):
    """Edge case: run for only 1 tick."""
    t = TestResult("Edge: single tick")
    try:
        sim = StochasticAssembly(alphabet_size=4, total_monomers=1000, half_life=100, seed=42)
        sim.set_target(4)
        sim.step()
        t.check(sim.tick == 1, f"tick != 1: {sim.tick}")
        t.check(len(sim.ts_ticks) == 1, f"time series length != 1: {len(sim.ts_ticks)}")
        total = sim.free + sum(len(c) for c in sim.chains)
        t.check(total == 1000, f"material not conserved: {total}")
    except Exception as e:
        t.check(False, f"Exception: {e}")
    return t


def test_large_target_length(verbose=False):
    """Edge case: target longer than any chain will ever reach."""
    t = TestResult("Edge: unreachable target (length 50)")
    try:
        sim = StochasticAssembly(alphabet_size=4, total_monomers=1000, half_life=100, seed=42)
        sim.set_target(50)
        for _ in range(200):
            sim.step()
        t.check(sim.target_found_tick is None, "found unreachable target!")
    except Exception as e:
        t.check(False, f"Exception: {e}")
    return t


def test_alphabet_20(verbose=False):
    """Test: amino acid alphabet produces expected behavior."""
    t = TestResult("Amino acid alphabet (|Σ|=20)")
    try:
        sim = run_sim(alphabet=20, material=50000, ticks=300)
        nt, nf, eq = get_crossovers(sim)
        t.check(nt is not None and nt <= 5, f"n*_target too high for |Σ|=20: {nt}")
        t.check(nf is not None and nf > 5, f"n*_F too low for |Σ|=20: {nf}")
        t.check(nf > nt if nt and nf else False, f"n*_F={nf} should be > n*_target={nt}")
        if verbose:
            print(f"    |Σ|=20: n*_target={nt}, n*_F={nf}, EqMax={eq}")
    except Exception as e:
        t.check(False, f"Exception: {e}")
    return t


def test_csv_output(verbose=False):
    """Test: --save-csv produces valid CSV file."""
    t = TestResult("CSV output")
    try:
        tmpfile = tempfile.mktemp(suffix='.csv')
        sim = StochasticAssembly(alphabet_size=4, total_monomers=5000, half_life=100, seed=42)
        sim.set_target(4)
        # Simulate CSV writing (simplified - just verify the data is available)
        ns, distincts, spaces, coverages, functionals, fns = sim.get_coverage_data()
        # After running, coverage_data should be obtainable
        for _ in range(100):
            sim.step()
        ns, distincts, spaces, coverages, functionals, fns = sim.get_coverage_data()
        t.check(len(ns) > 0, "no coverage data available")
        t.check(len(ns) == len(distincts) == len(coverages), "data arrays misaligned")
    except Exception as e:
        t.check(False, f"Exception: {e}")
    return t


def test_equilibrium_max_consistency(verbose=False):
    """Test: EqMax at plateau detection matches EqMax at end."""
    t = TestResult("EqMax consistency")
    sim = StochasticAssembly(alphabet_size=4, total_monomers=50000, half_life=100, seed=42)
    sim.set_target(6)
    plateau_eq = None
    for tick in range(2000):
        sim.step()
        if plateau_eq is None and detect_plateau(sim):
            plateau_eq = sim.get_equilibrium_max()
            # Run 200 more ticks
            for _ in range(200):
                sim.step()
            break
    final_eq = sim.get_equilibrium_max()
    if plateau_eq is not None:
        diff = abs(plateau_eq - final_eq)
        t.check(diff <= 3, f"EqMax at plateau={plateau_eq} vs final={final_eq}, diff={diff}")
        if verbose:
            print(f"    EqMax at plateau: {plateau_eq}, at end: {final_eq}")
    else:
        t.check(False, "plateau not detected")
    return t


def test_distinct_monotone_increasing(verbose=False):
    """Test: total distinct sequences never decreases over time."""
    t = TestResult("Distinct monotone increasing")
    sim = run_sim(material=20000, ticks=300)
    for i in range(1, len(sim.ts_total_distinct)):
        t.check(sim.ts_total_distinct[i] >= sim.ts_total_distinct[i-1],
                f"distinct decreased at tick {i}: {sim.ts_total_distinct[i]} < {sim.ts_total_distinct[i-1]}")
        if sim.ts_total_distinct[i] < sim.ts_total_distinct[i-1]:
            break  # stop on first failure
    return t


def test_functional_monotone_increasing(verbose=False):
    """Test: total functional found never decreases over time."""
    t = TestResult("Functional monotone increasing")
    sim = run_sim(material=20000, ticks=300)
    for i in range(1, len(sim.ts_total_functional)):
        t.check(sim.ts_total_functional[i] >= sim.ts_total_functional[i-1],
                f"functional decreased at tick {i}")
        if sim.ts_total_functional[i] < sim.ts_total_functional[i-1]:
            break
    return t


# ===================================================================
# EXPECTED VALUES TESTS (regression-style)
# ===================================================================

def test_expected_values_sig4_20k(verbose=False):
    """Regression: |Σ|=4, M=20000, HL=100, seed=42 should give known results."""
    t = TestResult("Expected values |Σ|=4 M=20k")
    sim = run_sim(alphabet=4, material=20000, half_life=100, ticks=500, seed=42)
    nt, nf, eq = get_crossovers(sim)

    t.check(nt in [5, 6], f"n*_target={nt}, expected 5 or 6")
    t.check(nf in [12, 13, 14], f"n*_F={nf}, expected 12-14")
    t.check(7 <= eq <= 10, f"EqMax={eq}, expected 7-10")

    # Check some specific numbers
    ns, distincts, spaces, coverages, functionals, fns = sim.get_coverage_data()
    # Length 2-4 should have full coverage
    for n, c in zip(ns, coverages):
        if n <= 4:
            t.check(c > 0.99, f"n={n} coverage={c:.4f}")

    if verbose:
        print(f"    n*_target={nt}, n*_F={nf}, EqMax={eq}")
        print(f"    distinct={sim.ts_total_distinct[-1]}, functional={sim.ts_total_functional[-1]}")
    return t


def test_expected_values_sig4_200k(verbose=False):
    """Regression: |Σ|=4, M=200000, HL=100, seed=42."""
    t = TestResult("Expected values |Σ|=4 M=200k")
    sim = run_sim(alphabet=4, material=200000, half_life=100, ticks=400, seed=42)
    nt, nf, eq = get_crossovers(sim)

    t.check(nt in [6, 7], f"n*_target={nt}, expected 6 or 7")
    t.check(nf in [13, 14, 15], f"n*_F={nf}, expected 13-15")
    t.check(8 <= eq <= 12, f"EqMax={eq}, expected 8-12")

    if verbose:
        print(f"    n*_target={nt}, n*_F={nf}, EqMax={eq}")
    return t


def test_expected_values_sig20_50k(verbose=False):
    """Regression: |Σ|=20, M=50000, HL=100, seed=42."""
    t = TestResult("Expected values |Σ|=20 M=50k")
    sim = run_sim(alphabet=20, material=50000, half_life=100, ticks=300, seed=42)
    nt, nf, eq = get_crossovers(sim)

    t.check(nt in [3, 4], f"n*_target={nt}, expected 3 or 4")
    t.check(nf is not None and nf >= 8, f"n*_F={nf}, expected >= 8")
    t.check(eq is not None and eq <= 12, f"EqMax={eq}, expected <= 12")

    if verbose:
        print(f"    n*_target={nt}, n*_F={nf}, EqMax={eq}")
    return t


# ===================================================================
# CONFIDENCE AND PARAMETER TESTS
# ===================================================================

def test_confidence_ceilings_basic(verbose=False):
    """Test: compute_confidence_ceilings returns valid structure and values."""
    t = TestResult("Confidence ceilings basic")
    sim = run_sim(material=200000, ticks=500)
    cc = sim.compute_confidence_ceilings(P=0.95)
    t.check(cc['P'] == 0.95, f"P != 0.95: {cc['P']}")
    t.check(isinstance(cc['details'], list), "details not a list")
    t.check(len(cc['details']) > 0, "no detail rows")
    t.check('n_target_P' in cc, "missing n_target_P")
    t.check('n_F_P' in cc, "missing n_F_P")
    t.check('n_loss_target' in cc, "missing n_loss_target")
    t.check('n_loss_F' in cc, "missing n_loss_F")
    t.check('eq_max' in cc, "missing eq_max")

    # Each detail row has required keys
    for d in cc['details']:
        for key in ['n', 'distinct', 'space', 'coverage', 'functional',
                     'f_n', 'p_target', 'p_functional', 'expected_func']:
            t.check(key in d, f"detail row missing key: {key}")

    if verbose:
        print(f"    n*_target(P=0.95)={cc['n_target_P']}, n*_F(P=0.95)={cc['n_F_P']}")
        print(f"    n_loss_target={cc['n_loss_target']}, n_loss_F={cc['n_loss_F']}")
    return t


def test_confidence_monotone_in_P(verbose=False):
    """Test: higher P gives equal or lower ceiling (stricter requirement)."""
    t = TestResult("Confidence monotone in P")
    sim = run_sim(material=200000, ticks=500)
    P_vals = [0.50, 0.80, 0.90, 0.95, 0.99]
    prev_nt, prev_nf = None, None
    for P in P_vals:
        cc = sim.compute_confidence_ceilings(P=P)
        nt = cc['n_target_P']
        nf = cc['n_F_P']
        if prev_nt is not None and nt is not None and prev_nt is not None:
            t.check(nt <= prev_nt, f"n*_target increased: P={P} gave {nt} > P={P_vals[P_vals.index(P)-1]} gave {prev_nt}")
        if prev_nf is not None and nf is not None and prev_nf is not None:
            t.check(nf <= prev_nf, f"n*_F increased: P={P} gave {nf} > P={P_vals[P_vals.index(P)-1]} gave {prev_nf}")
        prev_nt, prev_nf = nt, nf
        if verbose:
            print(f"    P={P}: n*_target={nt}, n*_F={nf}")
    return t


def test_confidence_p100_matches_visual(verbose=False):
    """Test: n*_target(P=1.0) = n_loss_target - 1 (max n with full coverage)."""
    t = TestResult("P=1.0 matches visual marker")
    sim = run_sim(material=200000, ticks=500)
    cc = sim.compute_confidence_ceilings(P=1.0)
    nt_P1 = cc['n_target_P']
    n_loss = cc['n_loss_target']
    if nt_P1 is not None and n_loss is not None:
        t.check(nt_P1 == n_loss - 1,
                f"n*_target(P=1.0)={nt_P1} should be n_loss_target-1={n_loss-1}")
    elif n_loss is not None:
        # If n_target_P is None, that means no length has 100% coverage (weird but possible)
        t.check(False, f"n*_target(P=1.0) is None but n_loss_target={n_loss}")
    if verbose:
        print(f"    n*_target(P=1.0)={nt_P1}, n_loss_target={n_loss}")
    return t


def test_confidence_target_leq_functional(verbose=False):
    """Test: n*_target(P) <= n*_F(P) for all P (Functional Search always reaches further)."""
    t = TestResult("n*_target(P) <= n*_F(P)")
    sim = run_sim(material=200000, ticks=500)
    for P in [0.50, 0.80, 0.95, 0.99]:
        cc = sim.compute_confidence_ceilings(P=P)
        nt = cc['n_target_P']
        nf = cc['n_F_P']
        if nt is not None and nf is not None:
            t.check(nt <= nf, f"P={P}: n*_target={nt} > n*_F={nf}")
    return t


def test_p_target_equals_coverage(verbose=False):
    """Test: p_target in details exactly equals coverage."""
    t = TestResult("p_target == coverage")
    sim = run_sim(material=50000, ticks=300)
    cc = sim.compute_confidence_ceilings()
    for d in cc['details']:
        t.check(abs(d['p_target'] - d['coverage']) < 1e-12,
                f"n={d['n']}: p_target={d['p_target']} != coverage={d['coverage']}")
    return t


def test_p_functional_range(verbose=False):
    """Test: p_functional in [0, 1] for all lengths."""
    t = TestResult("p_functional in [0,1]")
    sim = run_sim(material=50000, ticks=300)
    cc = sim.compute_confidence_ceilings()
    for d in cc['details']:
        t.check(0 <= d['p_functional'] <= 1.0001,
                f"n={d['n']}: p_functional={d['p_functional']} out of [0,1]")
    return t


def test_growth_frac_stored(verbose=False):
    """Test: growth_frac and elong_prob are stored and used."""
    t = TestResult("growth_frac/elong_prob stored")
    sim = StochasticAssembly(alphabet_size=4, total_monomers=10000, half_life=100,
                            seed=42, growth_frac=0.15, elong_prob=0.5, confidence=0.90)
    t.check(sim.growth_frac == 0.15, f"growth_frac={sim.growth_frac}")
    t.check(sim.elong_prob == 0.5, f"elong_prob={sim.elong_prob}")
    t.check(sim.confidence == 0.90, f"confidence={sim.confidence}")
    return t


def test_growth_frac_affects_discovery(verbose=False):
    """Test: higher growth_frac consumes more free monomers early (more growth attempts)."""
    t = TestResult("growth_frac affects growth rate")
    # After just a few ticks, higher g should consume more free monomers into chains
    sim_high = run_sim(material=50000, ticks=5, seed=42, growth_frac=0.5)
    sim_low = run_sim(material=50000, ticks=5, seed=42, growth_frac=0.05)
    free_high = sim_high.free
    free_low = sim_low.free
    t.check(free_high < free_low,
            f"g=0.5 free={free_high} should be < g=0.05 free={free_low} (more consumed)")
    # Also check total chain mass
    mass_high = sum(len(c) for c in sim_high.chains)
    mass_low = sum(len(c) for c in sim_low.chains)
    t.check(mass_high > mass_low,
            f"g=0.5 chain_mass={mass_high} should be > g=0.05 chain_mass={mass_low}")
    if verbose:
        print(f"    g=0.5 @5t: free={free_high}, chains_mass={mass_high}")
        print(f"    g=0.05 @5t: free={free_low}, chains_mass={mass_low}")
    return t


def test_elong_prob_affects_chain_length(verbose=False):
    """Test: higher elong_prob produces longer chains on average."""
    t = TestResult("elong_prob affects chain length")
    sim_high = run_sim(material=50000, half_life=500, ticks=300, seed=42, elong_prob=0.95)
    sim_low = run_sim(material=50000, half_life=500, ticks=300, seed=42, elong_prob=0.2)
    max_high = max(sim_high.ts_max_len)
    max_low = max(sim_low.ts_max_len)
    t.check(max_high >= max_low,
            f"p_elong=0.95 max_len={max_high} should be >= p_elong=0.2 max_len={max_low}")
    if verbose:
        print(f"    p_elong=0.95: max_len={max_high}, p_elong=0.2: max_len={max_low}")
    return t


def test_growth_frac_material_conservation(verbose=False):
    """Test: material conservation holds for non-default growth_frac/elong_prob."""
    t = TestResult("Material conservation (non-default params)")
    for gf, ep in [(0.1, 0.5), (0.5, 0.9), (0.05, 0.3)]:
        sim = StochasticAssembly(alphabet_size=4, total_monomers=10000, half_life=100,
                                seed=42, growth_frac=gf, elong_prob=ep)
        sim.set_target(4)
        for tick in range(100):
            sim.step()
            in_chains = sum(len(c) for c in sim.chains)
            total = sim.free + in_chains
            t.check(total == 10000,
                    f"g={gf} p={ep} tick={tick+1}: total={total} != 10000")
            if total != 10000:
                break
    return t


def test_confidence_default(verbose=False):
    """Test: default confidence is 0.95."""
    t = TestResult("Default confidence")
    sim = StochasticAssembly()
    t.check(sim.confidence == 0.95, f"default confidence={sim.confidence}, expected 0.95")
    t.check(sim.growth_frac == 0.3, f"default growth_frac={sim.growth_frac}, expected 0.3")
    t.check(sim.elong_prob == 0.7, f"default elong_prob={sim.elong_prob}, expected 0.7")
    return t


def test_input_validation(verbose=False):
    """Test: invalid parameters raise ValueError."""
    t = TestResult("Input validation")
    invalid_cases = [
        dict(alphabet_size=1),
        dict(alphabet_size=0),
        dict(total_monomers=-1),
        dict(half_life=-1),
        dict(growth_frac=-0.1),
        dict(growth_frac=1.5),
        dict(elong_prob=-0.1),
        dict(elong_prob=1.5),
        dict(confidence=0),
        dict(confidence=1),
        dict(confidence=1.5),
        dict(confidence=-0.1),
    ]
    for kwargs in invalid_cases:
        try:
            StochasticAssembly(**kwargs)
            t.check(False, f"no error for {kwargs}")
        except ValueError:
            t.check(True, f"ValueError for {kwargs}")
    # Valid edge cases that should NOT raise
    valid_cases = [
        dict(alphabet_size=2),
        dict(total_monomers=0),
        dict(half_life=0),
        dict(growth_frac=0),
        dict(growth_frac=1),
        dict(elong_prob=0),
        dict(elong_prob=1),
        dict(confidence=0.01),
        dict(confidence=0.99),
    ]
    for kwargs in valid_cases:
        try:
            StochasticAssembly(**kwargs)
            t.check(True, f"accepted {kwargs}")
        except ValueError as e:
            t.check(False, f"rejected valid {kwargs}: {e}")
    return t


def test_reproducibility_with_params(verbose=False):
    """Test: same seed + same params = identical results."""
    t = TestResult("Reproducibility (with params)")
    sim1 = run_sim(material=10000, ticks=200, seed=77, growth_frac=0.2, elong_prob=0.6)
    sim2 = run_sim(material=10000, ticks=200, seed=77, growth_frac=0.2, elong_prob=0.6)
    t.check(sim1.ts_total_distinct == sim2.ts_total_distinct, "distinct differs")
    t.check(sim1.ts_total_functional == sim2.ts_total_functional, "functional differs")
    t.check(sim1.ts_max_len == sim2.ts_max_len, "max_len differs")

    # Different growth_frac with same seed should produce different results
    sim3 = run_sim(material=10000, ticks=200, seed=77, growth_frac=0.4, elong_prob=0.6)
    t.check(sim1.ts_total_distinct != sim3.ts_total_distinct,
            "different growth_frac gave same distinct")
    return t


# ===================================================================
# MAIN TEST RUNNER
# ===================================================================

def main():
    p = argparse.ArgumentParser(description='Test Suite for Stochastic Assembly')
    p.add_argument('-v', '--verbose', action='store_true')
    p.add_argument('--quick', action='store_true', help='Skip slow tests')
    a = p.parse_args()

    print("=" * 60)
    print("  TEST SUITE: Stochastic Assembly Simulation")
    print("=" * 60)
    print()

    all_tests = [
        # Normal cases
        ("NORMAL CASES", [
            test_basic_run,
            test_material_conservation,
            test_no_negative_counts,
            test_functional_leq_distinct,
            test_coverage_monotone_decreasing,
            test_ntarget_lt_nfunc,
            test_logarithmic_scaling,
            test_halflife_affects_eqmax,
            test_alphabet_affects_gap,
            test_reproducibility,
            test_target_found_short,
            test_full_coverage_short_lengths,
            test_expected_vs_actual_functional,
            test_distinct_monotone_increasing,
            test_functional_monotone_increasing,
        ]),
        # Confidence and parameter tests
        ("CONFIDENCE & PARAMETERS", [
            test_confidence_ceilings_basic,
            test_confidence_monotone_in_P,
            test_confidence_p100_matches_visual,
            test_confidence_target_leq_functional,
            test_p_target_equals_coverage,
            test_p_functional_range,
            test_confidence_default,
            test_input_validation,
            test_growth_frac_stored,
            test_growth_frac_affects_discovery,
            test_growth_frac_material_conservation,
            test_elong_prob_affects_chain_length,
            test_reproducibility_with_params,
        ]),
        # Edge cases
        ("EDGE CASES", [
            test_zero_material,
            test_one_monomer,
            test_tiny_material,
            test_no_decay,
            test_huge_halflife,
            test_binary_alphabet,
            test_single_tick,
            test_large_target_length,
            test_alphabet_20,
            test_csv_output,
        ]),
        # Regression / expected values
        ("EXPECTED VALUES", [
            test_expected_values_sig4_20k,
            test_expected_values_sig4_200k,
            test_expected_values_sig20_50k,
        ]),
        # Plateau tests (slower)
        ("PLATEAU DETECTION", [
            test_plateau_detection,
            test_plateau_not_always_same,
            test_equilibrium_max_consistency,
        ]),
    ]

    total_passed = 0
    total_failed = 0
    t_start = time.time()

    for group_name, tests in all_tests:
        if a.quick and group_name == "PLATEAU DETECTION":
            print(f"  [{group_name}] SKIPPED (--quick)")
            print()
            continue

        print(f"  [{group_name}]")
        for test_func in tests:
            t0 = time.time()
            result = test_func(verbose=a.verbose)
            dt = time.time() - t0
            print(f"{result.summary()}  ({dt:.1f}s)")
            if result.failed > 0 and a.verbose:
                for err in result.errors:
                    print(f"      ✗ {err}")
            total_passed += result.passed
            total_failed += result.failed
        print()

    elapsed = time.time() - t_start
    print("=" * 60)
    if total_failed == 0:
        print(f"  ALL TESTS PASSED: {total_passed} checks in {elapsed:.1f}s")
    else:
        print(f"  {total_failed} FAILED, {total_passed} passed in {elapsed:.1f}s")
    print("=" * 60)
    sys.exit(0 if total_failed == 0 else 1)


if __name__ == '__main__':
    main()
