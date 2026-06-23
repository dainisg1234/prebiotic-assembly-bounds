#!/usr/bin/env python3
"""
Generate figures for the elemental constraints paper.

Usage:
  python generate_paper_figures.py              # all figures
  python generate_paper_figures.py --fig1       # nucleotide demo only
  python generate_paper_figures.py --fig2       # amino acid demo only
  python generate_paper_figures.py --sweep      # budget sweep table only

Output files (current directory):
  figure1_demo.png    — |Σ|=4, M=200,000
  figure3_aa.png      — |Σ|=20, M=50,000
  sweep_table.txt     — budget sweep text table
"""

import argparse
import sys
import os

# Ensure matplotlib uses non-interactive backend for batch generation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stochastic_assembly_visual import (
    StochasticAssembly, make_figure, detect_plateau,
    print_summary, run_budget_sweep
)


def generate_figure(alphabet, material, half_life, ticks, seed, confidence,
                    growth_frac, elong_prob, target_length, outfile, until_plateau=False):
    """Run simulation and save figure."""
    print(f'  Generating {outfile}:')
    print(f'    |Σ|={alphabet}, M={material:,}, τ½={half_life}, '
          f'g={growth_frac}, p_elong={elong_prob}, P={confidence}')

    sim = StochasticAssembly(
        alphabet_size=alphabet, total_monomers=material,
        half_life=half_life, seed=seed,
        growth_frac=growth_frac, elong_prob=elong_prob,
        confidence=confidence
    )
    sim.set_target(target_length)

    plateau_tick = None
    tick_limit = 1_000_000 if until_plateau else ticks

    for tick in range(tick_limit):
        sim.step()
        if until_plateau and plateau_tick is None and detect_plateau(sim):
            plateau_tick = sim.tick
            confirm = max(100, tick // 10)
            for _ in range(confirm):
                sim.step()
            break
        if (tick + 1) % 100 == 0:
            print(f'\r    tick {sim.tick:>8} ...', end='', flush=True)

    print(f'\r    Finished: {sim.tick:,} ticks')
    print_summary(sim)

    fig = make_figure(sim, plateau_tick, f'[tick {sim.tick:,}]')
    fig.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'    Saved: {outfile}')
    print()


def generate_sweep(outfile, confidence=0.95, growth_frac=0.3, elong_prob=0.7):
    """Run budget sweep and save output."""
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        run_budget_sweep(
            budgets=[1000, 5000, 20000, 100000, 500000, 2000000, 10000000],
            n_ticks=300, alphabet_size=4, half_life=100, seed=42,
            growth_frac=growth_frac, elong_prob=elong_prob,
            confidence=confidence
        )
    text = buf.getvalue()
    print(text)
    with open(outfile, 'w') as f:
        f.write(text)
    print(f'    Saved: {outfile}')
    print()


def main():
    p = argparse.ArgumentParser(description='Generate paper figures')
    p.add_argument('--fig1', action='store_true', help='Figure 1: nucleotide demo')
    p.add_argument('--fig2', action='store_true', help='Figure 2: amino acid demo')
    p.add_argument('--sweep', action='store_true', help='Budget sweep table')
    p.add_argument('--confidence', type=float, default=0.95)
    p.add_argument('--growth-frac', type=float, default=0.3)
    p.add_argument('--elong-prob', type=float, default=0.7)
    p.add_argument('--outdir', type=str, default='.')
    a = p.parse_args()

    do_all = not (a.fig1 or a.fig2 or a.sweep)

    print('=' * 60)
    print('  PAPER FIGURE GENERATION')
    print('=' * 60)
    print()

    if do_all or a.fig1:
        generate_figure(
            alphabet=4, material=200_000, half_life=100,
            ticks=500, seed=42, confidence=a.confidence,
            growth_frac=a.growth_frac, elong_prob=a.elong_prob,
            target_length=6,
            outfile=os.path.join(a.outdir, 'figure1_demo.png'),
            until_plateau=True
        )

    if do_all or a.fig2:
        generate_figure(
            alphabet=20, material=50_000, half_life=100,
            ticks=500, seed=42, confidence=a.confidence,
            growth_frac=a.growth_frac, elong_prob=a.elong_prob,
            target_length=3,
            outfile=os.path.join(a.outdir, 'figure3_aa.png'),
            until_plateau=True
        )

    if do_all or a.sweep:
        generate_sweep(
            outfile=os.path.join(a.outdir, 'sweep_table.txt'),
            confidence=a.confidence,
            growth_frac=a.growth_frac,
            elong_prob=a.elong_prob
        )

    print('=' * 60)
    print('  DONE')
    print('=' * 60)


if __name__ == '__main__':
    main()
