#!/usr/bin/env python3
"""
Next-generation battery chemistry model  -  v2 (sourced data + India cost layer)
===============================================================================
Compares NMC / LFP baselines against LMFP, Si-anode/NMC, Na-ion, solid-state
(Li-metal) and Li-S on cost, energy density, life, safety, supply risk and
maturity, with a Monte Carlo over every uncertain input, for five use cases:
  global:  Long-range EV | Grid storage | Low-cost mobility
  India:   Grid storage (India) | Low-cost mobility (India)

PROVENANCE TAGS (every parameter carries one; see data_sources.csv)
  [BatPaC]   Argonne BatPaC report ANL-24/01 (publications.anl.gov/anlpubs/2024/01/187177.pdf),
             Table 18, pack cost in 2023 $/kWh-rated.
  [DOE]      DOE VTO roadmap / merit-review slides (energy.gov).
  [IEA]      IEA Global EV Outlook 2026 and battery-markets commentary.
  [MFR]      Manufacturer or press claim (CATL, BYD, QuantumScape) - a claim, not an
             independent test; treat as an upper bound for performance.
  [GOI]      Government of India (PIB, DHI) or analysis of it (IEEFA).
  [DERIVED]  computed from a sourced ratio.
  [ASSUMED]  analyst placeholder - NOT sourced. Replace when you have data.
  [UNVERIFIED] a policy number that I could not confirm from a primary page.
              Check it against CBIC / GST Council notifications before relying on it.

Usage: python3 battery_chemistry_model.py [--out DIR] [--draws N] [--seed S]
                                             [--list-assumptions]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

YEARS = np.arange(2026, 2036)
DOD, RTE = 0.90, 0.92            # depth of discharge, round-trip efficiency [ASSUMED]

# --------------------------------------------------------------------------
# 1. CHEMISTRY PARAMETERS   (lo, mode, hi) triples feed the Monte Carlo
#    cost in $/kWh pack (BatPaC values are 2023 dollars; not inflated here)
# --------------------------------------------------------------------------
CHEMS = {
    "NMC (graphite)": dict(
        wh_kg=(220, 250, 280),            # [ASSUMED]
        cost26=(115, 125, 140),           # [DERIVED] between BatPaC Ni/Mn MY2023 $133 (0% Si) and MY2026 $116 (5% Si)
        cost35=(80, 95, 120),             # [DOE] graphite/HV-NMC potential $100-160 (old slide); BatPaC floor $78 needs 35% Si
        cycles=(1000, 1500, 2000),        # [DOE] pack target 1000 cycles
        trl=9, safety=0.60, supply_risk=0.65,   # [ASSUMED] Ni/Co/Li exposure
        india_domestic_from=2026,
        src="[DERIVED] cost26 from BatPaC; [DOE] cost35, cycles; [ASSUMED] wh_kg, safety, supply"),
    "LFP": dict(
        wh_kg=(160, 175, 190),            # [ASSUMED]; CATL storage cell ~160 [MFR] consistent
        cost26=(98, 102, 110),            # [BatPaC] BEV pack MY2026 $98.9-102.2
        cost35=(72, 78, 90),              # [BatPaC] MY2035 $77.2-78.3, widened upward
        cycles=(3000, 4500, 6000),        # [ASSUMED]
        trl=9, safety=0.85, supply_risk=0.45,   # [IEA] no Co/Ni but China-concentrated
        anchors={2026: 102.18, 2030: 86.64, 2035: 78.27},   # [BatPaC] BEV250 midsize LFP
        india_domestic_from=2026,
        src="[BatPaC] cost; [IEA] supply; [ASSUMED] wh_kg, cycles, safety"),
    "LMFP": dict(
        wh_kg=(160, 190, 210),            # [MFR] BYD Blade 2.0: 160 (short) / 210 (long) Wh/kg
        cost26=(95, 108, 125),            # [ASSUMED] ~5-10% premium over LFP
        cost35=(65, 78, 95),              # [ASSUMED]
        cycles=(2000, 3000, 4500),        # [ASSUMED] (BYD gave no cycle life)
        trl=7, safety=0.82, supply_risk=0.45,
        india_domestic_from=2029,         # [ASSUMED]
        src="[MFR] wh_kg (BYD Blade 2.0, press); [ASSUMED] rest"),
    "Si-anode / HV-NMC": dict(
        wh_kg=(280, 320, 350),            # [DOE] 350+ Wh/kg target; 300 Wh/kg cell @ $100/kWh
        cost26=(110, 116, 130),           # [BatPaC] Ni/Mn with 5% Si, MY2026 $116.28
        cost35=(72, 78, 95),              # [BatPaC] Ni/Mn with 35% Si, MY2035 $78.36
        cycles=(500, 900, 1300),          # [DOE] current >300, target 1000
        trl=7, safety=0.58, supply_risk=0.65,
        anchors={2026: 116.28, 2030: 100.04, 2035: 78.36},  # [BatPaC] BEV250 midsize SUV Ni/Mn
        india_domestic_from=2031,         # [ASSUMED]
        src="[BatPaC] cost; [DOE] wh_kg, cycles; [ASSUMED] safety, supply"),
    "Na-ion": dict(
        wh_kg=(130, 160, 175),            # [MFR] CATL Naxtra up to 175; storage cell ~160
        cost26=(75, 95, 120),             # [DERIVED] Argonne: ~1/3 below Li-ion; PNNL materials-only $100/kWh
        cost35=(40, 60, 90),              # [DOE-adjacent] study: parity w/ low-cost Li-ion early 2030s
        cycles=(2000, 4000, 10000),       # [MFR] CATL storage cell claims >15,000 cycles @80%; PNNL lab: 94% @250
        trl=7, safety=0.85, supply_risk=0.20,   # [IEA] no Li/Co/Ni; but nearly all capacity in China
        india_domestic_from=2029,         # [GOI/IEEFA] no commissioned Na-ion production; [ASSUMED] date
        src="[MFR] wh_kg, cycles; [DERIVED] cost; [IEA] supply"),
    "Solid-state (Li-metal/NMC)": dict(
        wh_kg=(300, 380, 450),            # [MFR] QuantumScape QSE-5 samples 301 Wh/kg; [DOE] Li-metal 400-500 projected
        cost26=(300, 400, 600),           # [ASSUMED] pre-commercial; [DOE] ~$320 w/ 5x excess Li
        cost35=(70, 118, 180),            # [DOE] potential $70-120 if cycle-life and safety solved
        cycles=(300, 700, 1100),          # [DOE] >300; [MFR] QuantumScape Osprey lab: 1000 cycles to ~80% (2-layer cell)
        trl=5, safety=0.80, supply_risk=0.70,   # [IEA] not demonstrated at scale; pilot-line starts ~2027-28 [MFR]
        india_domestic_from=2036,         # [ASSUMED] imported only through 2035
        src="[MFR]+[DOE] wh_kg, cycles; [IEA] maturity; [ASSUMED] cost26"),
    "Li-S": dict(
        wh_kg=(300, 400, 500),            # [DOE] 250-400 earlier; 500 projected
        cost26=(300, 450, 700),           # [ASSUMED]
        cost35=(50, 100, 200),            # [DOE] ~$75-80 if 1.5x excess Li and 75% S - optimistic
        cycles=(100, 300, 800),           # [DOE] ~100-300 (dated)
        trl=4, safety=0.55, supply_risk=0.35,
        india_domestic_from=2036,         # [ASSUMED]
        src="[DOE] wh_kg, cost35, cycles; [ASSUMED] cost26 - no usable current spec found (Lyten page gave none)"),
}

# --------------------------------------------------------------------------
# 2. INDIA COST LAYER
# --------------------------------------------------------------------------
INDIA = dict(
    fx_inr_per_usd=88.0,        # [ASSUMED] - set to the current rate
    freight_insurance=0.04,     # [ASSUMED] landed-cost uplift on imported packs
    bcd_on_imports=0.05,        # [UNVERIFIED] basic customs duty on cells/packs - check CBIC tariff
    gst_on_batteries=0.18,      # [UNVERIFIED] GST on Li-ion batteries - check GST Council; mostly input-credit for businesses
    domestic_cost_premium_2026=0.25,  # [GOI/IEEFA] India-made batteries "at least 20-30% costlier" than Chinese
    domestic_cost_premium_2035=0.08,  # [ASSUMED] premium narrows with scale and DVA
    dva_target_year=2031,       # [GOI/PIB] PLI-ACC: >=25% DVA in 2 yrs, 60% in 5 yrs
    domestic_share_max=0.60,    # [GOI/PIB] 60% domestic value addition target; used as domestic supply share
    pli_cap_pct_of_price=0.20,  # [GOI/IEEFA] subsidy capped at 20% of sale price...
    pli_cap_inr_per_kwh=2000.0, # [GOI/IEEFA] ...and INR 2,000/kWh
    pli_last_year=2032,         # [GOI/PIB] Round-2 runs 7 years from 1 Jul 2025
    pli_effective_fraction=0.5, # [ASSUMED] share of the cap actually bid/paid (bids are below the cap)
)

# --------------------------------------------------------------------------
# 3. USE CASES  (weights sum to 1)
# --------------------------------------------------------------------------
USE_CASES = {
    "Long-range EV": dict(india=False, w=dict(energy=.30, cost=.20, life=.10, safety=.15, supply=.10, maturity=.15)),
    "Grid storage": dict(india=False, w=dict(energy=.02, cost=.30, life=.28, safety=.20, supply=.10, maturity=.10)),
    "Low-cost mobility": dict(india=False, w=dict(energy=.10, cost=.35, life=.15, safety=.20, supply=.10, maturity=.10)),
    "Grid storage (India)": dict(india=True, w=dict(energy=.02, cost=.30, life=.28, safety=.20, supply=.10, maturity=.10)),
    "Low-cost mobility (India)": dict(india=True, w=dict(energy=.10, cost=.35, life=.15, safety=.20, supply=.10, maturity=.10)),
}


def tri(rng, t, n):
    return rng.triangular(t[0], t[1], t[2], n)


def global_cost_path(p):
    """Median global pack-cost path. Uses BatPaC anchors when present
    (log-linear interpolation), else a geometric glide from cost26 to cost35."""
    if "anchors" in p:
        ys = sorted(p["anchors"])
        return np.exp(np.interp(YEARS, ys, np.log([p["anchors"][y] for y in ys])))
    c26, c35 = p["cost26"][1], p["cost35"][1]
    return c26 * (c35 / c26) ** ((YEARS - YEARS[0]) / (YEARS[-1] - YEARS[0]))


def india_cost_path(p, g):
    """India delivered pack cost ($/kWh, pre-GST) given global path g."""
    I = INDIA
    t = (YEARS - YEARS[0]) / (I["dva_target_year"] - YEARS[0])
    ds = np.clip(t, 0, 1) * I["domestic_share_max"]
    ds = np.where(YEARS >= p["india_domestic_from"], ds, 0.0)         # domestic supply share
    prem = I["domestic_cost_premium_2026"] + (I["domestic_cost_premium_2035"] - I["domestic_cost_premium_2026"]) * \
        (YEARS - YEARS[0]) / (YEARS[-1] - YEARS[0])
    imported = g * (1 + I["freight_insurance"] + I["bcd_on_imports"])
    dom = g * (1 + prem)
    cap_usd = I["pli_cap_inr_per_kwh"] / I["fx_inr_per_usd"] * I["pli_effective_fraction"]
    subsidy = np.minimum(cap_usd, I["pli_cap_pct_of_price"] * dom)   # per-kWh subsidy, USD
    subsidy = np.where(YEARS <= I["pli_last_year"], subsidy, 0.0)
    dom_net = dom - subsidy
    return (1 - ds) * imported + ds * dom_net, ds


def simulate(draws, seed, india_ratio):
    rng = np.random.default_rng(seed)
    names = list(CHEMS)
    sims = {n: dict(wh=tri(rng, CHEMS[n]["wh_kg"], draws), c26=tri(rng, CHEMS[n]["cost26"], draws),
                    c35=tri(rng, CHEMS[n]["cost35"], draws), cyc=tri(rng, CHEMS[n]["cycles"], draws)) for n in names}
    wh = np.vstack([sims[n]["wh"] for n in names])
    c35 = np.vstack([sims[n]["c35"] for n in names])
    c35_in = c35 * np.array([india_ratio[n] for n in names])[:, None]
    cyc = np.vstack([sims[n]["cyc"] for n in names])
    ones = np.ones((1, draws))
    safety = np.array([CHEMS[n]["safety"] for n in names])[:, None] * ones
    supply = (1 - np.array([CHEMS[n]["supply_risk"] for n in names]))[:, None] * ones
    maturity = (np.array([CHEMS[n]["trl"] for n in names]) / 9.0)[:, None] * ones

    def norm(a, invert=False):
        lo, hi = a.min(axis=0), a.max(axis=0)
        s = (a - lo) / (hi - lo + 1e-12)
        return 1 - s if invert else s

    scores, ranks, lcos_by = {}, {}, {}
    for uc, cfg in USE_CASES.items():
        cost = c35_in if cfg["india"] else c35
        lcos = cost / (cyc * DOD * RTE)
        lcos_by[uc] = lcos
        crit = dict(energy=norm(wh), cost=norm(cost, True), life=norm(lcos, True),
                    safety=safety, supply=supply, maturity=maturity)
        sc = sum(cfg["w"][k] * crit[k] for k in cfg["w"])
        scores[uc] = sc
        ranks[uc] = (-sc).argsort(axis=0).argsort(axis=0) + 1
    return names, sims, scores, ranks, lcos_by


def provenance_rows():
    rows = []
    for n, p in CHEMS.items():
        rows.append(dict(item=f"{n}", detail=p["src"]))
    for k, v in INDIA.items():
        rows.append(dict(item=f"India.{k}", detail=str(v)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="battery_model_output_v2")
    ap.add_argument("--draws", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--list-assumptions", action="store_true")
    a = ap.parse_args()

    if a.list_assumptions:
        for n, p in CHEMS.items():
            print(f"{n}: {p['src']}")
        print("India: fx, freight, domestic premium 2035 = ASSUMED; BCD and GST = UNVERIFIED")
        return
    os.makedirs(a.out, exist_ok=True)

    # cost paths (global and India)
    rows, india_ratio = [], {}
    for n, p in CHEMS.items():
        g = global_cost_path(p)
        ind, ds = india_cost_path(p, g)
        late = YEARS >= 2030                      # mean over 2030-35 so chemistries that localise later pay for it
        india_ratio[n] = float(np.mean(ind[late] / g[late]))
        for y, gc, ic, d in zip(YEARS, g, ind, ds):
            rows.append(dict(chemistry=n, year=int(y), global_pack_cost=round(float(gc), 1),
                             india_delivered_pre_gst=round(float(ic), 1),
                             india_with_gst=round(float(ic) * (1 + INDIA["gst_on_batteries"]), 1),
                             india_domestic_share=round(float(d), 2)))
    cost_df = pd.DataFrame(rows)
    cost_df.to_csv(f"{a.out}/cost_projection_global_and_india.csv", index=False)

    names, sims, scores, ranks, lcos_by = simulate(a.draws, a.seed, india_ratio)

    summ = []
    for i, n in enumerate(names):
        s = sims[n]
        summ.append(dict(
            chemistry=n, cell_wh_per_kg_p50=round(float(np.median(s["wh"]))),
            cost_2026_p50=round(float(np.median(s["c26"]))),
            cost_2035_p10=round(float(np.percentile(s["c35"], 10))),
            cost_2035_p50=round(float(np.median(s["c35"]))),
            cost_2035_p90=round(float(np.percentile(s["c35"], 90))),
            india_2035_vs_global=round(india_ratio[n], 3),
            cycles_p50=round(float(np.median(s["cyc"]))),
            lcos_2035_global_p50=round(float(np.median(lcos_by["Grid storage"][i])), 3),
            lcos_2035_india_p50=round(float(np.median(lcos_by["Grid storage (India)"][i])), 3),
            trl=CHEMS[n]["trl"], data_basis=CHEMS[n]["src"]))
    pd.DataFrame(summ).to_csv(f"{a.out}/chemistry_summary.csv", index=False)

    rr = []
    for uc in USE_CASES:
        for i, n in enumerate(names):
            r = ranks[uc][i]
            rr.append(dict(use_case=uc, chemistry=n, mean_score=round(float(scores[uc][i].mean()), 3),
                           prob_rank1=round(float((r == 1).mean()), 3),
                           prob_top3=round(float((r <= 3).mean()), 3), median_rank=int(np.median(r))))
    rank_df = pd.DataFrame(rr).sort_values(["use_case", "mean_score"], ascending=[True, False])
    rank_df.to_csv(f"{a.out}/use_case_ranking.csv", index=False)
    pd.DataFrame(provenance_rows()).to_csv(f"{a.out}/data_sources.csv", index=False)

    # charts
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, col, ttl in zip(axes, ["global_pack_cost", "india_delivered_pre_gst"],
                            ["Global pack cost", "India delivered cost (pre-GST)"]):
        for c, n in zip(colors, names):
            d = cost_df[cost_df.chemistry == n]
            ax.plot(d.year, d[col], color=c, lw=2, label=n)
        ax.axhline(75, ls="--", c="grey")
        ax.set_yscale("log"); ax.set_title(ttl); ax.grid(alpha=.3); ax.set_ylabel("$/kWh (log)")
    axes[0].legend(fontsize=7); fig.tight_layout()
    fig.savefig(f"{a.out}/cost_projection_global_vs_india.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, len(USE_CASES), figsize=(20, 4.5), sharey=True)
    for ax, uc in zip(axes, USE_CASES):
        d = rank_df[rank_df.use_case == uc].sort_values("prob_top3")
        ax.barh(d.chemistry, d.prob_top3, color="#3b7dd8"); ax.set_xlim(0, 1)
        ax.set_title(uc, fontsize=9); ax.set_xlabel("P(top-3)")
    fig.tight_layout(); fig.savefig(f"{a.out}/use_case_ranking.png", dpi=150); plt.close(fig)

    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(pd.DataFrame(summ).drop(columns="data_basis").to_string(index=False)); print()
    for uc in USE_CASES:
        print("==", uc)
        print(rank_df[rank_df.use_case == uc].drop(columns="use_case").to_string(index=False)); print()
    print(cost_df[cost_df.year.isin([2026, 2031, 2035])].to_string(index=False))


if __name__ == "__main__":
    main()
