# Stochastic Assembly Simulation

**Visual Monte Carlo demonstration of Target Search vs Functional Search in stochastic chain assembly.**

Companion code for: D. Geidmanis, *Elemental and Kinetic Constraints on the Stochastic Production of Functional Biopolymers: Target Search vs Functional Search*.

The program illustrates, at small scale, the difference between two detection problems:

1. **Target Search**: can blind stochastic assembly produce one pre-specified sequence?
2. **Functional Search**: can blind stochastic assembly produce at least one member of a functional sequence class?

The simulation is intended as an illustration and consistency check. It is not an independent validation of Earth-scale chemical estimates.

---

## Installation

```bash
pip install numpy matplotlib
```

---

## Quick Start

```bash
# Default demo (|Σ|=4, M=20000, live visualization)
python stochastic_assembly_visual.py

# Nucleotide-like, large budget
python stochastic_assembly_visual.py --alphabet 4 -m 200000 -t 500

# Amino-acid-like
python stochastic_assembly_visual.py --alphabet 20 -m 50000 -t 500

# Fast headless run with figure at end
python stochastic_assembly_visual.py --max-speed -m 100000 -t 10000

# Run until plateau is detected
python stochastic_assembly_visual.py --until-plateau -m 200000

# Budget sweep (logarithmic scaling table)
python stochastic_assembly_visual.py --sweep

# Full parameterization
python stochastic_assembly_visual.py --until-plateau -m 2000000 --half-life 2150 \
    --confidence 0.95 --growth-frac 0.3 --elong-prob 0.7

# Save results to CSV
python stochastic_assembly_visual.py --save-csv results.csv -t 1000
```

---

## Parameters

| Parameter | Default | Meaning |
|---|---:|---|
| `--alphabet`, `-a` | `4` | Alphabet size `S = |Σ|`. Use `4` for nucleotide-like, `20` for amino-acid-like. |
| `--material`, `-m` | `20000` | Total monomer budget `M`. |
| `--half-life` | `100` | Bond half-life in ticks. |
| `--ticks`, `-t` | `500` | Number of simulation ticks (ignored in `--until-plateau` mode). |
| `--target-length` | `6` | Length of the pre-specified target sequence. |
| `--confidence` | `0.95` | Detection confidence `P` for computing `n*_target(P)` and `n*_F(P)`. |
| `--growth-frac` | `0.3` | Per-tick growth utilization: `A_t = floor(g × F_t)` growth attempts. |
| `--elong-prob` | `0.7` | Probability that a growth attempt elongates an existing chain (vs nucleating a new dimer). |
| `--update-every` | `10` | Visualization / progress refresh interval. |
| `--seed` | `42` | Random seed for reproducibility. |
| `--no-plot` | off | Text-only mode. |
| `--max-speed` | off | Run without live visualization; show final figure. |
| `--until-plateau` | off | Run until plateau detected, then confirm. |
| `--save-csv` | none | Save per-tick coverage table to CSV. |
| `--sweep` | off | Run budget sweep over material sizes. |

---

## Interactive Controls

| Key | Action |
|---|---|
| `SPACE` | Pause/resume. In max-speed mode, pause and show current figure. |
| `+` or `=` | Speed up live visualization. |
| `-` | Slow down live visualization. |
| `S` | Save figure as PNG. |
| `Ctrl+C` | Cancel program. |

Close the plot window to return to the prompt.

---

## Core Notation

Let `S = |Σ|` be the alphabet size. A chain of length `n` is a sequence `w = (w_1, ..., w_n) ∈ Σⁿ`. The full sequence space at length `n` has size `|Σⁿ| = Sⁿ`.

Let `M` be the total number of monomers. At tick `t`, let `C_t` be the multiset of chains present at the beginning of the growth phase. Then the number of free monomers is:

```
F_t = M − Σ_{w ∈ C_t} |w|
```

Mass is conserved: `M = F_t + Σ_{w ∈ C_t} |w|`.

---

## Growth Rule

At tick `t`, the number of growth attempts is:

```
A_t = floor(g × F_t)
```

where `g = growth_frac` (default 0.3).

For each growth attempt:

- with probability `p_elong` (default 0.7), one free monomer is appended to a randomly chosen existing chain;
- with probability `1 − p_elong`, a new dimer is nucleated (consuming two free monomers).

These are illustrative simulation parameters, not established chemical constants. Their qualitative effects:

| Parameter change | Expected effect |
|---|---|
| Increase `g` | More growth attempts per tick, faster approach to plateau. |
| Decrease `g` | Fewer growth attempts per tick, slower discovery. |
| Increase `p_elong` | Favors longer chains over many short ones. |
| Decrease `p_elong` | Favors many short chains, suppresses long-chain production. |

The expected free-monomer consumption per growth attempt is `E[ΔF] = p_elong × 1 + (1 − p_elong) × 2 = 2 − p_elong`. For `p_elong = 0.7`, this gives `E[ΔF] = 1.3`.

### Relation to the paper-level utilization factor

The paper defines a coarse-grained assembly-utilization factor `g ∈ (0, 1]` multiplying the ideal trial budget: `k_max(n; g) = g × k_max(n)`. The simulation-level `growth_frac` is a per-tick kinetic analog — not identical, but representing the same concept (kinetic utilization of available material).

---

## Decay Rule

The input half-life is a bond half-life `τ₁/₂`. The per-bond cleavage probability per tick is:

```
p_cleave = 1 − 0.5^(1/τ₁/₂)
```

A chain of length `n` has `n − 1` independently cleavable bonds. The probability that it survives one tick is:

```
P_survive(n) = (1 − p_cleave)^(n−1) = 2^(−(n−1)/τ₁/₂)
```

Longer chains are less stable. When cleaved, fragments of length ≥ 2 remain as chains; singletons return to the free-monomer pool.

---

## Current Chains vs Cumulative Discoveries

**Snapshot semantics.** The simulation records chain states at the end of each tick's growth phase (after growth, before decay for recording purposes). Intermediate chain states created during multi-step elongation within a single tick, and fragments created by decay, are not recorded as discoveries until they appear in a subsequent end-of-tick snapshot. This is a conservative modeling choice: it undercounts distinct sequences relative to a continuous-recording model, which is safe for an upper-bound argument. The README notation `D_n(T)` refers specifically to these end-of-tick observations.

### Relation to the paper's trial-budget ceiling

The paper defines an idealized trial-budget ceiling `k_cap(n;g)` — a material-recycling capacity ceiling that assumes all material is perfectly allocated into full-length chains of exactly length n. The simulation does not compute k_cap; it computes actual stochastic assembly outcomes `D_n(T)` under a specific growth-decay model. The simulation's distinct counts are always far below k_cap, illustrating how generous the ceiling is. Both the paper's Poisson threshold (`k_cap/|Σ|ⁿ ≥ ln(1/(1−P))`) and the simulation's coverage criterion (`C_n(T) ≥ P`) answer the same detection question at different levels of abstraction.

### Current living chains: `H_n(t)`

```
H_n(t) = #{w ∈ C_t : |w| = n}
```

The top-left panel plots `H_n(T)` at the final tick.

### Cumulative distinct sequences: `D_n(T)`

The set of distinct length-`n` sequences that appeared at least once during ticks `1, ..., T`. The code's blue time-evolution curve is `Σ_n |D_n(t)|`.

### Cumulative functional discoveries: `F_n(T)`

Distinct length-`n` sequences classified as functional. The bottom-left panel plots `F_n(T)` by length. The green time-evolution curve is `Σ_n F_n(t)`.

---

## Sequence-Space Coverage

```
C_n(T) = |D_n(T)| / S^n
```

This is the probability that a uniformly chosen target from `Σⁿ` lies in the discovered set.

---

## Confidence-Based Target Search

For required confidence `P`, the simulation-based target-search ceiling is:

```
n*_target(P) = max{n : C_n(T) ≥ P}
```

The visual marker is the first length where coverage drops below 100%:

```
n_loss,target = min{n : C_n(T) < 1}
```

These are related by: `n*_target(P=1.0) = n_loss,target − 1`.

---

## Confidence-Based Functional Search

Given `|D_n(T)|` distinct sampled sequences and functional fraction `f(n)`, the probability of at least one functional discovery is:

```
P_F(n,T) = 1 − (1 − f(n))^|D_n(T)|
```

For required confidence `P`:

```
n*_F(P) = max{n : P_F(n,T) ≥ P}
```

The visual marker is:

```
n_loss,F = min{n : F_n(T) = 0 and |D_n(T)| > 0}
```

The Poisson approximation (valid for small `f(n)`) gives `P_F ≈ 1 − exp(−|D_n(T)| × f(n))`.

---

## Functional Fraction

Each newly discovered sequence of length `n` is independently classified as functional with probability:

```
f(n) = min(10^(a + b×n + c×n²), 1)
```

Defaults:

| Alphabet | `a` | `b` | `c` |
|---|---|---|---|
| `S ≤ 4` (nucleotide) | 2.0 | −0.35 | 0.0 |
| `S > 4` (amino acid) | 3.0 | −0.55 | 0.0 |

These are heuristic visibility settings for the simulation, not universal biological constants.

---

## Equilibrium Maximum Chain Length

```
L_eq(T) = max{L_max(t) : T−W+1 ≤ t ≤ T}
```

where `W = 50` (window). The effective functional ceiling is:

```
n*_F,effective(P) = min(n*_F(P), L_eq(T))
```

---

## The Four Panels

| Panel | Quantity | Cumulative? |
|---|---|---|
| Top-left | `H_n(T)` — current living chain length distribution | No |
| Top-right | `C_n(T)` — target search coverage (log scale) | Yes (over time) |
| Bottom-left | `F_n(T)` — functional discoveries by length | Yes (over time) |
| Bottom-right | Time evolution: distinct (blue), functional (green), max length (red) | Yes |

Panel annotations include both visual markers and confidence-based ceilings when available.

---

## Terminal Output Columns

| Column | Formula | Meaning |
|---|---|---|
| `n` | — | Chain length |
| `Space` | `S^n` | Sequence space size |
| `Distinct` | `|D_n(T)|` | Distinct sequences ever observed |
| `Coverage` | `|D_n(T)|/S^n` | Fraction of space sampled |
| `P_target` | `C_n(T)` | Target-search probability (= coverage) |
| `Functional` | `F_n(T)` | Functional sequences found |
| `f(n)` | `min(10^(a+bn+cn²), 1)` | Functional fraction |
| `E[func]` | `|D_n(T)| × f(n)` | Expected functional count |
| `P_func` | `1−(1−f(n))^|D_n(T)|` | Functional-search probability |

---

## Reporting Convention

The code distinguishes visual markers (diagnostic) from confidence ceilings (analytical):

| Quantity | Notation | Meaning |
|---|---|---|
| First incomplete coverage | `n_loss,target` | First `n` where `C_n(T) < 1` |
| 100% target ceiling | `n*_target(P=1.0)` | `n_loss,target − 1` |
| Confidence target ceiling | `n*_target(P)` | Max `n` where coverage ≥ `P` |
| First zero functional | `n_loss,F` | First sampled `n` with no functional discovery |
| Confidence functional ceiling | `n*_F(P)` | Max `n` where `P_F ≥ P` |
| Equilibrium max length | `L_eq` | Recent-window max of current chain length |
| Effective functional ceiling | `min(n*_F(P), L_eq)` | Functional ceiling capped by survival |

---

## Plateau Detection

In `--until-plateau` mode, the simulation stops when both discovery and length dynamics have stabilized:

1. Recent discovery rate below 1% of the early peak for three consecutive windows (150 ticks).
2. Recent maximum chain length sufficiently stable.

After detection, additional confirmation ticks are run.

---

## Test Suite

```bash
python test_assembly.py           # all tests
python test_assembly.py -v        # verbose
python test_assembly.py --quick   # skip slow plateau tests
```

40 tests, ~1,660 checks covering: material conservation, monotonicity invariants, confidence ceiling correctness (monotone in P, P=1.0 matches visual marker, target ≤ functional), growth_frac/elong_prob parameter effects, edge cases (zero material, no decay, binary alphabet), expected-value regression, and plateau detection.

---

## License

MIT License.
#   p r e b i o t i c - a s s e m b l y - b o u n d s  
 