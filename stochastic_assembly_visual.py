#!/usr/bin/env python3
"""
Visual Monte Carlo: Stochastic Chain Assembly
==============================================
Demonstrates Target Search (specific target) vs Functional Search (any functional).

Interactive controls (visual mode):
  SPACE  - pause/resume + show current figure
  +/=    - speed up
  -      - slow down  
  S      - save figure as PNG

Controls (max-speed / until-plateau modes):
  SPACE  - pause and show current state figure; close figure to resume
  Ctrl+C - cancel program

Usage examples:
  python stochastic_assembly_visual.py                             # default demo
  python stochastic_assembly_visual.py --alphabet 4 -m 200000     # nucleotide
  python stochastic_assembly_visual.py --max-speed -m 100000 -t 5000
  python stochastic_assembly_visual.py --until-plateau -m 200000
  python stochastic_assembly_visual.py --sweep
"""

import math
import random
import argparse
import sys
import time as time_mod
from collections import defaultdict, Counter

try:
    import matplotlib
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib not found; install with: pip install matplotlib")

import numpy as np

# Non-blocking keyboard check (Windows / Unix)
try:
    import msvcrt
    def key_pressed():
        if msvcrt.kbhit():
            ch = msvcrt.getch()
            if ch == b' ': return 'space'
            if ch == b'\x03': raise KeyboardInterrupt
        return None
except ImportError:
    import select
    def key_pressed():
        if select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if ch == ' ': return 'space'
        return None


class StochasticAssembly:
    """Simulation engine for stochastic chain assembly."""

    def __init__(self, alphabet_size=4, total_monomers=10000,
                 half_life=100, f_params=None, seed=42,
                 growth_frac=0.3, elong_prob=0.7, confidence=0.95):
        # Input validation
        if alphabet_size < 2:
            raise ValueError("alphabet_size must be >= 2")
        if total_monomers < 0:
            raise ValueError("total_monomers must be non-negative")
        if half_life < 0:
            raise ValueError("half_life must be non-negative")
        if not (0 <= growth_frac <= 1):
            raise ValueError("growth_frac must be in [0, 1]")
        if not (0 <= elong_prob <= 1):
            raise ValueError("elong_prob must be in [0, 1]")
        if not (0 < confidence < 1):
            raise ValueError("confidence must be in (0, 1)")

        self.S = alphabet_size
        self.M = total_monomers
        self.half_life = half_life
        self.growth_frac = growth_frac
        self.elong_prob = elong_prob
        self.confidence = confidence
        self.rng = random.Random(seed)
        self.p_cleave = 1 - 0.5 ** (1.0 / half_life) if half_life > 0 else 0

        if f_params is None:
            if alphabet_size <= 4:
                self.f_a, self.f_b, self.f_c = 2.0, -0.35, 0.0
            else:
                self.f_a, self.f_b, self.f_c = 3.0, -0.55, 0.0
        else:
            self.f_a, self.f_b, self.f_c = f_params

        # State
        self.free = total_monomers
        self.chains = []
        self.tick = 0

        # Tracking
        self.distinct_ever = defaultdict(set)
        self.functional_found = defaultdict(set)
        self.target = None
        self.target_found_tick = None

        # Time series
        self.ts_ticks = []
        self.ts_free = []
        self.ts_n_chains = []
        self.ts_max_len = []           # current max (fluctuates)
        self.ts_total_distinct = []
        self.ts_total_functional = []

    def f(self, n):
        log_f = self.f_a + self.f_b * n + self.f_c * n**2
        return min(10 ** log_f, 1.0) if log_f < 0 else 1.0

    def set_target(self, length):
        self.target = tuple(self.rng.randint(0, self.S - 1) for _ in range(length))
        return self.target

    def step(self):
        self.tick += 1

        # GROWTH
        n_add = int(self.free * self.growth_frac)
        for _ in range(n_add):
            if self.free <= 0: break
            m = self.rng.randint(0, self.S - 1)
            if self.chains and self.rng.random() < self.elong_prob:
                self.rng.choice(self.chains).append(m)
                self.free -= 1
            else:
                if self.free < 2: continue
                m2 = self.rng.randint(0, self.S - 1)
                self.chains.append([m, m2])
                self.free -= 2

        # RECORD
        for ch in self.chains:
            n = len(ch)
            seq = tuple(ch)
            if seq not in self.distinct_ever[n]:
                self.distinct_ever[n].add(seq)
                if self.rng.random() < self.f(n):
                    self.functional_found[n].add(seq)
            if (self.target_found_tick is None and self.target is not None
                    and n == len(self.target) and seq == self.target):
                self.target_found_tick = self.tick

        # DECAY
        if self.p_cleave > 0:
            new_chains = []
            for ch in self.chains:
                L = len(ch)
                if L < 2:
                    self.free += 1; continue
                sites = [i for i in range(1, L) if self.rng.random() < self.p_cleave]
                if not sites:
                    new_chains.append(ch); continue
                prev = 0
                for site in sites:
                    frag = ch[prev:site]
                    if len(frag) == 1: self.free += 1
                    else: new_chains.append(frag)
                    prev = site
                frag = ch[prev:]
                if len(frag) == 1: self.free += 1
                else: new_chains.append(frag)
            self.chains = new_chains

        # TIME SERIES
        lens = [len(c) for c in self.chains]
        current_max = max(lens) if lens else 0
        self.ts_ticks.append(self.tick)
        self.ts_free.append(self.free)
        self.ts_n_chains.append(len(self.chains))
        self.ts_max_len.append(current_max)
        self.ts_total_distinct.append(sum(len(v) for v in self.distinct_ever.values()))
        self.ts_total_functional.append(sum(len(v) for v in self.functional_found.values()))

    def get_coverage_data(self, max_n=None):
        if max_n is None:
            max_n = max(self.distinct_ever.keys()) if self.distinct_ever else 2
        ns, distincts, spaces, coverages, functionals, fns = [], [], [], [], [], []
        for n in range(2, max_n + 1):
            space = self.S ** n
            distinct = len(self.distinct_ever.get(n, set()))
            func = len(self.functional_found.get(n, set()))
            ns.append(n); distincts.append(distinct); spaces.append(space)
            coverages.append(distinct / space if space > 0 else 0)
            functionals.append(func); fns.append(self.f(n))
        return ns, distincts, spaces, coverages, functionals, fns

    def get_length_distribution(self):
        if not self.chains: return [], []
        cnt = Counter(len(c) for c in self.chains)
        lengths = sorted(cnt.keys())
        return lengths, [cnt[l] for l in lengths]

    def get_equilibrium_max(self, window=50):
        """Get the equilibrium (plateau) max chain length."""
        if len(self.ts_max_len) < window:
            return max(self.ts_max_len) if self.ts_max_len else 0
        recent = self.ts_max_len[-window:]
        return max(recent)

    def compute_confidence_ceilings(self, P=None):
        """Compute confidence-based ceilings n*_target(P) and n*_F(P).

        Returns dict with keys:
          n_target_P       — max n where coverage >= P
          n_F_P            — max n where P_functional >= P
          n_loss_target    — first n where coverage < 1.0 (visual marker)
          n_loss_F         — first n where functional == 0 and distinct > 0 (visual marker)
          details          — list of per-length dicts with p_target, p_functional
        """
        if P is None:
            P = self.confidence
        ns, distincts, spaces, coverages, functionals, fns = self.get_coverage_data()

        n_target_P = None
        n_F_P = None
        n_loss_target = None
        n_loss_F = None
        details = []

        for i, n in enumerate(ns):
            coverage = coverages[i]
            distinct = distincts[i]
            f_n = fns[i]

            # Target: P_target = coverage
            p_target = coverage

            # Functional: P_F = 1 - (1 - f_n)^distinct
            if distinct > 0 and f_n > 0:
                if f_n >= 1.0:
                    p_functional = 1.0
                else:
                    log_survive = distinct * math.log1p(-f_n)
                    p_functional = 1.0 - math.exp(log_survive)
            else:
                p_functional = 0.0

            details.append({
                'n': n, 'distinct': distinct, 'space': spaces[i],
                'coverage': coverage, 'functional': functionals[i],
                'f_n': f_n, 'p_target': p_target, 'p_functional': p_functional,
                'expected_func': distinct * f_n,
            })

            # Confidence ceilings: track the maximum n meeting the threshold
            if p_target >= P:
                n_target_P = n
            if p_functional >= P:
                n_F_P = n

            # Visual markers (first failure)
            if n_loss_target is None and coverage < 1.0:
                n_loss_target = n
            if (n_loss_F is None and functionals[i] == 0
                    and distinct > 0):
                n_loss_F = n

        return {
            'P': P,
            'n_target_P': n_target_P,
            'n_F_P': n_F_P,
            'n_loss_target': n_loss_target,
            'n_loss_F': n_loss_F,
            'eq_max': self.get_equilibrium_max(),
            'details': details,
        }


def make_figure(sim, plateau_tick=None, title_suffix=''):
    """Create the 4-panel figure from current simulation state."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f'Stochastic Assembly: |Σ|={sim.S}, M={sim.M:,}, '
                 f'τ½={sim.half_life} ticks  {title_suffix}', fontsize=13)
    ax_hist, ax_cov, ax_func, ax_ts = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    ns, distincts, spaces, coverages, functionals, fns = sim.get_coverage_data()

    # Panel 1: Length distribution
    lens, counts = sim.get_length_distribution()
    if lens: ax_hist.bar(lens, counts, color='steelblue', alpha=0.7)
    ax_hist.set_xlabel('Chain length n')
    ax_hist.set_ylabel('Number of chains')
    ax_hist.set_title(f'Chain Length Distribution (tick {sim.tick:,})')
    ax_hist.set_xlim(1, max(30, max(lens) + 2) if lens else 30)

    # Panel 2: Coverage (Target Search)
    cov_nz = [(n, c) for n, c in zip(ns, coverages) if c > 0]
    if cov_nz:
        cn, cc = zip(*cov_nz)
        ax_cov.semilogy(cn, cc, 'o-', color='crimson', markersize=4)
    ax_cov.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='100% coverage')
    ax_cov.axhline(y=0.5, color='orange', linestyle=':', alpha=0.5, label='50% coverage')
    ax_cov.set_xlabel('Chain length n')
    ax_cov.set_ylabel('Coverage (fraction of space sampled)')
    ax_cov.set_title('Target Search: Sequence-Space Coverage')
    ax_cov.set_ylim(1e-10, 2)
    ax_cov.legend(fontsize=8)

    # Confidence-based ceiling and visual marker
    cc_data = sim.compute_confidence_ceilings()
    P_val = cc_data['P']
    if cc_data['n_loss_target'] is not None:
        n_vis = cc_data['n_loss_target']
        ax_cov.axvline(x=n_vis, color='crimson', linestyle='--', alpha=0.3)
        label_parts = [f'coverage lost at n={n_vis}']
        if cc_data['n_target_P'] is not None:
            label_parts.append(f'n*_target(P={P_val})={cc_data["n_target_P"]}')
        ax_cov.text(n_vis, 1.5, '\n'.join(label_parts),
                   fontsize=8, color='crimson', ha='center')

    # Panel 3: Functional (Functional Search)
    ax_func.bar(ns, functionals, color='forestgreen', alpha=0.7, label='Found functional')
    expected = [d * f for d, f in zip(distincts, fns)]
    ax_func.plot(ns, expected, 'r--', linewidth=1.5, alpha=0.7, label='Expected (distinct × f)')
    ax_func.set_xlabel('Chain length n')
    ax_func.set_ylabel('Functional sequences found')
    ax_func.set_title('Functional Search: Discoveries by Length')
    ax_func.legend(fontsize=8)

    # Confidence-based ceiling and visual marker
    if cc_data['n_loss_F'] is not None:
        n_vis_f = cc_data['n_loss_F']
        ax_func.axvline(x=n_vis_f, color='red', linestyle='--', alpha=0.3)
        flabel_parts = [f'zero functional at n={n_vis_f}']
        if cc_data['n_F_P'] is not None:
            flabel_parts.append(f'n*_F(P={P_val})={cc_data["n_F_P"]}')
        ax_func.text(n_vis_f, max(functionals) * 0.9 if max(functionals) > 0 else 1,
                    '\n'.join(flabel_parts), fontsize=8, color='red', ha='center')

    # Mark equilibrium max on functional panel
    eq_max = sim.get_equilibrium_max()
    if eq_max > 0:
        ax_func.axvline(x=eq_max, color='blue', linestyle=':', alpha=0.5)
        ax_func.text(eq_max + 0.3, max(functionals) * 0.7 if max(functionals) > 0 else 1,
                    f'equil. max≈{eq_max}', fontsize=8, color='blue', rotation=90, va='top')

    # Panel 4: Time series (dual y-axis)
    ax_ts.plot(sim.ts_ticks, sim.ts_total_distinct, 'b-', label='Distinct seqs')
    ax_ts.plot(sim.ts_ticks, sim.ts_total_functional, 'g-', label='Functional found')
    ax_ts.set_xlabel('Tick')
    ax_ts.set_ylabel('Count (sequences)', color='blue')
    ax_ts.set_title('Time Evolution')
    ax_ts.legend(fontsize=8, loc='center left')
    if sim.target_found_tick:
        ax_ts.axvline(x=sim.target_found_tick, color='gold', linewidth=2)
        ax_ts.text(sim.target_found_tick, max(sim.ts_total_distinct) * 0.8,
                  f'Target found!\n(tick {sim.target_found_tick})',
                  fontsize=9, color='darkgoldenrod', ha='right')
    if plateau_tick:
        ax_ts.axvline(x=plateau_tick, color='purple', linewidth=2, linestyle=':')
        ax_ts.text(plateau_tick, max(sim.ts_total_distinct) * 0.6,
                  f'Plateau\n(tick {plateau_tick})',
                  fontsize=9, color='purple', ha='right')
    # Secondary axis for current max chain length
    ax_ts2 = ax_ts.twinx()
    ax_ts2.plot(sim.ts_ticks, sim.ts_max_len, 'r-', alpha=0.5, linewidth=1,
               label='Current max length')
    ax_ts2.set_ylabel('Current max chain length', color='red')
    ax_ts2.tick_params(axis='y', labelcolor='red')
    ax_ts2.legend(fontsize=8, loc='center right')
    # Annotate final value
    if sim.ts_max_len:
        final_max = sim.ts_max_len[-1]
        ax_ts2.annotate(f'{final_max}', xy=(sim.ts_ticks[-1], final_max),
                       fontsize=10, fontweight='bold', color='red',
                       ha='left', va='center')

    fig.tight_layout()
    return fig


def detect_plateau(sim, window=50):
    """Detect plateau: discovery rate AND max-length have stabilized.
    
    Requires rate < 1% of peak for 3 CONSECUTIVE windows (150 ticks),
    preventing false triggers from the initial burst dying down.
    """
    if len(sim.ts_total_distinct) < 300:
        return False
    rates = []
    for i in range(window, len(sim.ts_total_distinct), window):
        r = sim.ts_total_distinct[i] - sim.ts_total_distinct[i - window]
        rates.append(r)
    if len(rates) < 6:
        return False
    peak_rate = max(rates[:max(1, len(rates)//3)])
    if peak_rate <= 0:
        return True
    # Require LAST 3 windows ALL below 1% of peak
    rate_stable = all(r / peak_rate < 0.01 for r in rates[-3:])

    # Max-length stability (last 100 ticks)
    if len(sim.ts_max_len) >= 100:
        recent_max = sim.ts_max_len[-100:]
        max_range = max(recent_max) - min(recent_max)
        mean_max = np.mean(recent_max)
        len_stable = max_range < max(3, mean_max * 0.5)
    else:
        len_stable = False

    return rate_stable and len_stable


def run_max_speed(sim, max_ticks, until_plateau, report_every):
    """Run at maximum speed with SPACE=pause/show and Ctrl+C=cancel."""
    from datetime import datetime
    mode = "until-plateau" if until_plateau else "max-speed"
    tick_limit = max_ticks if not until_plateau else 1_000_000
    print(f'  Mode: {mode}')
    print(f'  Max ticks: {max_ticks if not until_plateau else "unlimited (until plateau)"}')
    print(f'  Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'  Press SPACE to pause and view figure, Ctrl+C to cancel')
    print()

    t_start = time_mod.time()
    plateau_tick = None

    try:
        for tick in range(tick_limit):
            # Check for SPACE key (non-blocking)
            k = key_pressed()
            if k == 'space' and HAS_MATPLOTLIB:
                print(f'\n  PAUSED at tick {sim.tick}. Showing figure...')
                fig = make_figure(sim, plateau_tick, f'[PAUSED @ tick {sim.tick:,}]')
                print(f'  Close the figure window to resume.')
                plt.show()
                print(f'  Resuming...')

            sim.step()

            # Plateau detection
            if plateau_tick is None and detect_plateau(sim):
                plateau_tick = sim.tick
                eq_max = sim.get_equilibrium_max()
                if until_plateau:
                    print(f'\r  PLATEAU at tick {plateau_tick} '
                          f'(equil. max chain length ≈ {eq_max})')
                    confirm = max(100, tick // 10)
                    print(f'  Confirming with {confirm} more ticks...')
                    for _ in range(confirm):
                        sim.step()
                    break

            # Progress
            if (tick + 1) % report_every == 0:
                elapsed = time_mod.time() - t_start
                tps = (tick + 1) / elapsed
                eq_max = sim.get_equilibrium_max()
                if until_plateau:
                    print(f'\r  Tick {sim.tick:>8}  '
                          f'Distinct:{sim.ts_total_distinct[-1]:>8,}  '
                          f'Functional:{sim.ts_total_functional[-1]:>6,}  '
                          f'MaxLen:{sim.ts_max_len[-1]:>4}  '
                          f'EqMax:{eq_max:>4}  '
                          f'[{elapsed:.0f}s, {tps:.0f} t/s]     ',
                          end='', flush=True)
                else:
                    eta = (max_ticks - tick - 1) / tps
                    print(f'\r  Tick {sim.tick:>6}/{max_ticks}  '
                          f'Distinct:{sim.ts_total_distinct[-1]:>8,}  '
                          f'Functional:{sim.ts_total_functional[-1]:>6,}  '
                          f'MaxLen:{sim.ts_max_len[-1]:>4}  '
                          f'[{elapsed:.0f}s, ETA {eta:.0f}s, {tps:.0f} t/s]     ',
                          end='', flush=True)

    except KeyboardInterrupt:
        print(f'\n\n  CANCELLED by user at tick {sim.tick}.')

    elapsed = time_mod.time() - t_start
    print(f'\n\n  Finished: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'  Total: {sim.tick:,} ticks in {elapsed:.1f}s ({sim.tick/max(elapsed,0.1):.0f} ticks/s)')
    eq_max = sim.get_equilibrium_max()
    print(f'  Equilibrium max chain length: {eq_max}')
    if plateau_tick:
        print(f'  Plateau detected at tick {plateau_tick}')
    print()
    print_summary(sim)

    if HAS_MATPLOTLIB:
        print('\n  Generating final figure...')
        fig = make_figure(sim, plateau_tick, f'[FINAL @ tick {sim.tick:,}]')
        print('  Close the plot window to return to the command prompt.')
        plt.show()


def run_visual(sim, n_ticks, update_every, target_length, save_csv=None, delay_ms=50):
    """Run with live matplotlib visualization."""
    sim.set_target(target_length)
    if not HAS_MATPLOTLIB:
        run_text_only(sim, n_ticks, update_every)
        return

    state = {'paused': False, 'delay': delay_ms, 'plateau_tick': None}

    def on_key(event):
        if event.key == ' ':
            state['paused'] = not state['paused']
            if state['paused']:
                fig.suptitle(f'⏸ PAUSED  |  Press SPACE to resume, S to save', fontsize=12)
            else:
                fig.suptitle(f'Stochastic Assembly: |Σ|={sim.S}, M={sim.M:,}, '
                           f'τ½={sim.half_life} ticks', fontsize=13)
            fig.canvas.draw_idle()
        elif event.key in ('+', '='):
            state['delay'] = max(1, state['delay'] // 2)
        elif event.key == '-':
            state['delay'] = min(5000, state['delay'] * 2)
        elif event.key == 's':
            from datetime import datetime
            fname = f'assembly_{datetime.now().strftime("%H%M%S")}.png'
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            print(f'\r  Saved: {fname}', end='', flush=True)

    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f'Stochastic Assembly: |Σ|={sim.S}, M={sim.M:,}, '
                 f'τ½={sim.half_life} ticks', fontsize=13)
    ax_hist, ax_cov, ax_func, ax_ts = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    fig.canvas.mpl_connect('key_press_event', on_key)

    print(f'  Controls: SPACE=pause  +/-=speed  S=save')
    print()

    csv_file = open(save_csv, 'w') if save_csv else None
    if csv_file:
        csv_file.write('tick,n,distinct,space,coverage,functional,f_n\n')

    t_start = time_mod.time()

    for tick in range(n_ticks):
        while state['paused']:
            plt.pause(0.1)
        sim.step()

        if (tick + 1) % update_every == 0 or tick == n_ticks - 1:
            ns, distincts, spaces, coverages, functionals, fns = sim.get_coverage_data()

            if csv_file:
                for i, n in enumerate(ns):
                    csv_file.write(f'{sim.tick},{n},{distincts[i]},{spaces[i]},'
                                   f'{coverages[i]:.6e},{functionals[i]},{fns[i]:.6e}\n')

            ax_hist.clear()
            lens, counts = sim.get_length_distribution()
            if lens: ax_hist.bar(lens, counts, color='steelblue', alpha=0.7)
            ax_hist.set_xlabel('Chain length n')
            ax_hist.set_ylabel('Number of chains')
            ax_hist.set_title(f'Chain Length Distribution (tick {sim.tick})')
            ax_hist.set_xlim(1, max(30, max(lens) + 2) if lens else 30)

            ax_cov.clear()
            cov_nz = [(n, c) for n, c in zip(ns, coverages) if c > 0]
            if cov_nz:
                cn, cc = zip(*cov_nz)
                ax_cov.semilogy(cn, cc, 'o-', color='crimson', markersize=4)
            ax_cov.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='100%')
            ax_cov.axhline(y=0.5, color='orange', linestyle=':', alpha=0.5, label='50%')
            ax_cov.set_xlabel('Chain length n')
            ax_cov.set_ylabel('Coverage')
            ax_cov.set_title('Target Search: Sequence-Space Coverage')
            ax_cov.set_ylim(1e-10, 2)
            ax_cov.legend(fontsize=8)
            live_cc = sim.compute_confidence_ceilings()
            if live_cc['n_loss_target'] is not None:
                n_vis = live_cc['n_loss_target']
                ax_cov.axvline(x=n_vis, color='crimson', linestyle='--', alpha=0.3)
                lbl = f'coverage lost n={n_vis}'
                if live_cc['n_target_P'] is not None:
                    lbl += f'\nn*_target(P={sim.confidence})={live_cc["n_target_P"]}'
                ax_cov.text(n_vis, 1.5, lbl, fontsize=8, color='crimson', ha='center')

            ax_func.clear()
            ax_func.bar(ns, functionals, color='forestgreen', alpha=0.7, label='Found functional')
            expected = [d * f for d, f in zip(distincts, fns)]
            ax_func.plot(ns, expected, 'r--', linewidth=1.5, alpha=0.7, label='Expected')
            ax_func.set_xlabel('Chain length n')
            ax_func.set_ylabel('Functional found')
            ax_func.set_title('Functional Search: Discoveries by Length')
            ax_func.legend(fontsize=8)
            if live_cc['n_loss_F'] is not None:
                n_vis_f = live_cc['n_loss_F']
                ax_func.axvline(x=n_vis_f, color='red', linestyle='--', alpha=0.3)
                flbl = f'zero func n={n_vis_f}'
                if live_cc['n_F_P'] is not None:
                    flbl += f'\nn*_F(P={sim.confidence})={live_cc["n_F_P"]}'
                ax_func.text(n_vis_f, max(functionals)*0.9 if max(functionals)>0 else 1,
                            flbl, fontsize=8, color='red', ha='center')
            eq_max = sim.get_equilibrium_max()
            if eq_max > 0:
                ax_func.axvline(x=eq_max, color='blue', linestyle=':', alpha=0.5)
                ax_func.text(eq_max+0.3, max(functionals)*0.7 if max(functionals)>0 else 1,
                            f'equil.max≈{eq_max}', fontsize=8, color='blue', rotation=90, va='top')

            ax_ts.clear()
            ax_ts.plot(sim.ts_ticks, sim.ts_total_distinct, 'b-', label='Distinct seqs')
            ax_ts.plot(sim.ts_ticks, sim.ts_total_functional, 'g-', label='Functional found')
            ax_ts.set_xlabel('Tick')
            ax_ts.set_ylabel('Count', color='blue')
            ax_ts.set_title('Time Evolution')
            ax_ts.legend(fontsize=8, loc='center left')
            if sim.target_found_tick:
                ax_ts.axvline(x=sim.target_found_tick, color='gold', linewidth=2)
            # Twin axis for current max
            for child_ax in [a for a in fig.get_axes() if a is not ax_hist and a is not ax_cov
                             and a is not ax_func and a is not ax_ts]:
                child_ax.remove()
            ax_ts2 = ax_ts.twinx()
            ax_ts2.plot(sim.ts_ticks, sim.ts_max_len, 'r-', alpha=0.5, label='Current max len')
            ax_ts2.set_ylabel('Current max length', color='red')
            ax_ts2.tick_params(axis='y', labelcolor='red')
            ax_ts2.legend(fontsize=8, loc='center right')

            # Plateau detection
            if state['plateau_tick'] is None and detect_plateau(sim):
                state['plateau_tick'] = sim.tick
                ax_ts.axvline(x=sim.tick, color='purple', linewidth=2, linestyle=':')
                ax_ts.text(sim.tick, max(sim.ts_total_distinct)*0.6,
                          f'Plateau\n(tick {sim.tick})', fontsize=9, color='purple', ha='right')

            fig.tight_layout()
            plt.pause(max(0.001, state['delay'] / 1000.0))

            elapsed = time_mod.time() - t_start
            eta = elapsed / (tick + 1) * (n_ticks - tick - 1)
            plateau_str = f'  PLATEAU@{state["plateau_tick"]}' if state['plateau_tick'] else ''
            print(f'\r  Tick {sim.tick:>6}/{n_ticks}  '
                  f'Free:{sim.free:>7,}  Chains:{len(sim.chains):>7,}  '
                  f'MaxLen:{sim.ts_max_len[-1]:>4}  EqMax:{eq_max:>4}  '
                  f'Distinct:{sim.ts_total_distinct[-1]:>8,}  '
                  f'Functional:{sim.ts_total_functional[-1]:>6,}  '
                  f'[{elapsed:.0f}s, ETA {eta:.0f}s]{plateau_str}     ',
                  end='', flush=True)

    if csv_file: csv_file.close()

    from datetime import datetime
    print(f'\n\n  Finished: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'  Total time: {time_mod.time() - t_start:.1f}s')
    eq_max = sim.get_equilibrium_max()
    print(f'  Equilibrium max chain length: {eq_max}')
    print()
    print_summary(sim)

    plt.ioff()
    print('\n  Close the plot window to return to the command prompt.')
    plt.show()


def run_text_only(sim, n_ticks, update_every):
    for tick in range(n_ticks):
        sim.step()
        if (tick + 1) % update_every == 0 or tick == n_ticks - 1:
            eq_max = sim.get_equilibrium_max()
            print(f'  Tick {sim.tick:>6}: free={sim.free:>6} chains={len(sim.chains):>6} '
                  f'max={sim.ts_max_len[-1]:>4} eq_max={eq_max:>4} '
                  f'distinct={sim.ts_total_distinct[-1]:>8} '
                  f'functional={sim.ts_total_functional[-1]:>6}')
    print()
    print_summary(sim)


def print_summary(sim):
    cc = sim.compute_confidence_ceilings()
    details = cc['details']
    eq_max = cc['eq_max']
    P = cc['P']

    print('=' * 85)
    print('  FINAL RESULTS')
    print('=' * 85)
    print(f'  Alphabet: |Σ| = {sim.S}')
    print(f'  Material: M = {sim.M:,} monomers')
    print(f'  Half-life: τ½ = {sim.half_life} ticks')
    print(f'  Growth fraction: g = {sim.growth_frac}')
    print(f'  Elongation probability: p_elong = {sim.elong_prob}')
    print(f'  Confidence: P = {sim.confidence}')
    print(f'  Equilibrium max chain length: {eq_max}')
    print(f'  Target (Target Search): {sim.target}')
    if sim.target_found_tick:
        print(f'  Target FOUND at tick {sim.target_found_tick}')
    else:
        print(f'  Target NOT FOUND')
    print()

    print(f'  {"n":>4} {"Space":>12} {"Distinct":>10} {"Coverage":>10} '
          f'{"P_target":>10} {"Functional":>11} {"f(n)":>10} {"E[func]":>10} {"P_func":>10}')
    print('  ' + '-' * 90)

    for d in details:
        n = d['n']
        space_s = f'{d["space"]}' if d['space'] < 1e8 else f'10^{math.log10(d["space"]):.0f}'
        print(f'  {n:>4} {space_s:>12} {d["distinct"]:>10} {d["coverage"]:>10.4%} '
              f'{d["p_target"]:>10.4%} {d["functional"]:>11} {d["f_n"]:>10.2e} '
              f'{d["expected_func"]:>10.1f} {d["p_functional"]:>10.4%}')

    print()
    print(f'  --- Visual Markers (diagnostic) ---')
    n_loss_t = cc['n_loss_target']
    n_loss_f = cc['n_loss_F']
    print(f'  Full coverage lost at n = {n_loss_t or "?"}')
    print(f'  First zero functional at n = {n_loss_f or "?"}')
    print()

    print(f'  --- Confidence Ceilings (P = {P}) ---')
    n_t_P = cc['n_target_P']
    n_f_P = cc['n_F_P']
    print(f'  n*_target(P={P}) = {n_t_P or "?"}')
    print(f'  n*_F(P={P}) = {n_f_P or "?"}')
    print()

    print(f'  EQUILIBRIUM MAX LENGTH: {eq_max}')
    print(f'    Chains longer than this do not persist under decay (τ½={sim.half_life} ticks).')
    if n_f_P is not None:
        effective = min(n_f_P, eq_max) if eq_max > 0 else n_f_P
        print(f'    Effective n*_F(P={P}): min({n_f_P}, {eq_max}) = {effective}')
    print()

    if n_t_P is not None and n_f_P is not None:
        gap = n_f_P - n_t_P
        print(f'  THE GAP: n*_target(P={P}) = {n_t_P}  →  n*_F(P={P}) = {n_f_P}  '
              f'(Functional Search reaches {gap} further)')
    elif n_loss_t is not None and n_loss_f is not None:
        gap = n_loss_f - n_loss_t
        print(f'  THE GAP (visual markers): {n_loss_t} → {n_loss_f}  '
              f'(Functional Search reaches {gap} further)')
    print('=' * 85)


def run_budget_sweep(budgets, n_ticks, alphabet_size, half_life, seed=42,
                     growth_frac=0.3, elong_prob=0.7, confidence=0.95):
    print()
    print('=' * 80)
    print('  BUDGET SWEEP: n*_target and n*_F vs Material')
    print('=' * 80)
    print(f'  g={growth_frac}, p_elong={elong_prob}, P={confidence}')
    print(f'  {"Material":>12} {"log10(M)":>10} {"n*_target(P)":>12} {"n*_F(P)":>10} '
          f'{"n_loss_tgt":>10} {"n_loss_F":>10} {"EqMax":>6}')
    print('  ' + '-' * 65)
    for M in budgets:
        sim = StochasticAssembly(alphabet_size=alphabet_size, total_monomers=M,
                                half_life=half_life, seed=seed,
                                growth_frac=growth_frac, elong_prob=elong_prob,
                                confidence=confidence)
        sim.set_target(max(3, int(math.log(M, alphabet_size) * 0.8)))
        for _ in range(n_ticks): sim.step()
        cc = sim.compute_confidence_ceilings()
        print(f'  {M:>12,} {math.log10(M):>10.1f} '
              f'{str(cc["n_target_P"]):>12} {str(cc["n_F_P"]):>10} '
              f'{str(cc["n_loss_target"]):>10} {str(cc["n_loss_F"]):>10} '
              f'{cc["eq_max"]:>6}')
    print()
    print('  n*_target(P), n*_F(P), and EqMax all grow LOGARITHMICALLY with material.')
    print('=' * 80)


def main():
    p = argparse.ArgumentParser(description='Stochastic Assembly Simulation')
    p.add_argument('--alphabet', '-a', type=int, default=4)
    p.add_argument('--material', '-m', type=int, default=20000)
    p.add_argument('--half-life', type=int, default=100)
    p.add_argument('--ticks', '-t', type=int, default=500)
    p.add_argument('--target-length', type=int, default=6)
    p.add_argument('--update-every', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-plot', action='store_true')
    p.add_argument('--max-speed', action='store_true')
    p.add_argument('--until-plateau', action='store_true')
    p.add_argument('--delay', type=int, default=50)
    p.add_argument('--save-csv', type=str, default=None)
    p.add_argument('--sweep', action='store_true')
    p.add_argument('--confidence', type=float, default=0.95,
                   help='Detection confidence P for n*_target(P) and n*_F(P)')
    p.add_argument('--growth-frac', type=float, default=0.3,
                   help='Fraction of free monomers used for growth attempts per tick')
    p.add_argument('--elong-prob', type=float, default=0.7,
                   help='Probability that a growth attempt elongates vs nucleates')
    a = p.parse_args()

    if a.sweep:
        budgets = [1000, 5000, 20000, 100000, 500000]
        run_budget_sweep(budgets, min(a.ticks, 300), a.alphabet, a.half_life, a.seed,
                         a.growth_frac, a.elong_prob, a.confidence)
        return

    sim = StochasticAssembly(alphabet_size=a.alphabet, total_monomers=a.material,
                            half_life=a.half_life, seed=a.seed,
                            growth_frac=a.growth_frac, elong_prob=a.elong_prob,
                            confidence=a.confidence)

    from datetime import datetime
    print('=' * 70)
    print('  STOCHASTIC ASSEMBLY SIMULATION')
    print('=' * 70)
    print(f'  Alphabet: |Σ| = {a.alphabet}')
    print(f'  Material: M = {a.material:,} monomers')
    print(f'  Half-life: τ½ = {a.half_life} ticks')
    print(f'  Growth fraction: g = {a.growth_frac}')
    print(f'  Elongation probability: p_elong = {a.elong_prob}')
    print(f'  Confidence: P = {a.confidence}')
    print(f'  Target length: {a.target_length}')
    if a.until_plateau:
        print(f'  Mode: until-plateau (unlimited ticks)')
    else:
        print(f'  Ticks: {a.ticks}')
    print(f'  Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print()

    if a.max_speed or a.until_plateau:
        sim.set_target(a.target_length)
        run_max_speed(sim, a.ticks, a.until_plateau, a.update_every)
    elif a.no_plot or not HAS_MATPLOTLIB:
        sim.set_target(a.target_length)
        run_text_only(sim, a.ticks, a.update_every)
    else:
        run_visual(sim, a.ticks, a.update_every, a.target_length, a.save_csv, a.delay)


if __name__ == '__main__':
    main()
