# Next-Gen Battery

Scenario model and interactive dashboard comparing next-generation battery chemistries (NMC, LFP, LMFP, silicon-anode NMC, sodium-ion, solid-state lithium-metal, lithium-sulfur) from 2026 to 2035, globally and for India.

## Abstract

The model scores each chemistry on energy density, 2035 pack cost, lifetime cost per kWh delivered, safety, supply security and technology maturity. A 3,000-draw Monte Carlo propagates uncertainty in every input, and rankings are weighted for long-range EVs, grid storage and low-cost mobility. An India layer adds import duty, freight, a domestic-cost premium, PLI subsidy and a phased rise in domestic supply.

In the baseline, sodium-ion leads grid storage and low-cost mobility in about 90% of simulations, ahead of LFP and LMFP. Long-range EV is less settled: silicon-anode NMC leads in about half of outcomes, but this depends on BatPaC's assumption of 35% silicon content by 2035. India's delivered costs run about 7 to 9% above global without changing the ranking. Solid-state and lithium-sulfur do not reach cost competitiveness by 2035 in the median case.

## Contents

| Path | What it is |
|---|---|
| `dashboard/index.html` | Self-contained interactive dashboard (Battery Scenario Lab). Open it in a browser. |
| `model/battery_chemistry_model.py` | Python model with sourced data and the India cost layer. |
| `model/battery_chemistry_model_v1.py` | First version, kept for comparison (more assumed inputs). |
| `outputs/` | Model outputs: summary, cost projections, rankings, data sources, charts. |
| `tests/test_dashboard_model.js` | Runs the dashboard's model code in Node for a baseline check against the Python results. |

## Run

```
python3 model/battery_chemistry_model.py --out outputs     # needs numpy, pandas, matplotlib
python3 model/battery_chemistry_model.py --list-assumptions
node tests/test_dashboard_model.js
```

## Data basis

Every parameter carries a provenance tag in the code and in `outputs/data_sources.csv`.

- **Sourced:** Argonne BatPaC (pack cost for LFP and silicon-anode NMC, 2023 dollars), DOE Vehicle Technologies Office roadmaps, PNNL and Argonne sodium-ion work, IEA Global EV Outlook 2026, PIB and IEEFA on India's PLI-ACC scheme.
- **Manufacturer claims:** CATL sodium-ion, BYD Blade 2.0 (LMFP), QuantumScape. Treat these as upper bounds.
- **Assumed:** safety scores, LFP, NMC and LMFP cycle life, solid-state and Li-S 2026 cost, lithium exposure of pack cost, freight, exchange rate, and when each chemistry localises in India.
- **Unverified:** India customs duty on cells and the GST rate. Check CBIC and GST Council notices before relying on them.

## Sources

- [Argonne BatPaC report ANL-24/01](https://publications.anl.gov/anlpubs/2024/01/187177.pdf)
- [DOE VTO battery roadmap](https://www.energy.gov/node/4523480)
- [IEA Global EV Outlook 2026](https://iea.org/reports/global-ev-outlook-2026/electric-vehicle-batteries)
- [PNNL sodium-ion report](https://www.pnnl.gov/main/publications/external/technical_reports/PNNL-33878-3.pdf)
- [PIB, PLI-ACC scheme](https://pib.gov.in/PressReleasePage.aspx?PRID=2224552)
- [IEEFA, Assessing India's ACC PLI scheme](https://ieefa.org/sites/default/files/2026-01/Assessing%20India%27s%20ACC%20PLI%20scheme.pdf)
