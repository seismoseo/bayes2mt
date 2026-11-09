# bayes2mt

Rupture length, width, directivity and stress drop of small earthquakes from **second-degree
moments**, using apparent source time functions (ASTFs) obtained with empirical Green's functions
(EGFs).

```
waveforms ──pld──▶ ASTFs ──▶ durations τc ──inversion──▶ Lc, Wc, v0, stress drop
```

- `pld`: ASTFs by projected Landweber deconvolution, in batch or with an interactive picker
- `bayesian`: hierarchical Bayesian inversion ([BayesBay](https://bayes-bay.readthedocs.io); the data noise is sampled too)
- `deterministic`: semidefinite-program inversion (McGuire, 2004; [CVXPY](https://www.cvxpy.org))
- `moments`, `stressdrop`, `relloc`, `plotting`: rupture parameters, uncertainties and figures

## Installation

```bash
git clone https://github.com/seismoseo/bayes2mt.git
cd bayes2mt
conda env create -f environment.yml          # environment "bayes2mt"
conda activate bayes2mt
python -m ipykernel install --user --name bayes2mt
```

You can also run `pip install -e ".[all]"` in an existing environment. BayesBay needs a C++ compiler.

The picker needs widgets. If you use the classic Jupyter Notebook 6, run
`pip install "ipywidgets>=7.8,<8"`; otherwise the picker shows only `VBox(...)`.

## Quick start

Run from the repository folder, on the bundled example event:

```python
import json
from bayes2mt import pld, geometry, bayesian, moments, stressdrop
from bayes2mt.config import moment_from_mw

DATA = "examples/data/event016"                 # Main/ and EGF/ SAC files, picks.csv, event.json
ev = json.load(open(f"{DATA}/event.json"))

# 1. ASTFs by deconvolution, then the duration table (tau_c per station and phase)
picks = pld.read_picks(f"{DATA}/picks.csv")
astfs = pld.retrieve_from_picks(picks, DATA)
table = pld.duration_table(astfs, picks, DATA, ev["latitude"], ev["longitude"], ev["depth_km"])

# 2. Hierarchical Bayesian inversion (20 chains x 10^6 iterations)
G = geometry.slowness_matrix(table, ev["strike"], ev["dip"], ev["vp_km_s"], ev["vs_km_s"])
ensemble, per_chain = bayesian.run(bayesian.build(G, table["tauc"]))

# 3. Rupture parameters: representative (medoid) model and 95 % HDI
m2 = moments.cholesky_to_m2(ensemble.x)
best = moments.physical(m2[moments.medoid_index(m2)])
low, high = moments.hdi(moments.physical(m2)["Lc"])
sd = stressdrop.stress_drop(moment_from_mw(ev["mw"]), best["Lc"], best["Wc"], ev["rake"], best["angle"])
print(f"Lc {best['Lc']:.0f} m (95 % HDI {low:.0f}-{high:.0f} m), stress drop {sd:.1f} MPa")
```

Step 2 takes about 2 min on 20 cores. The run is repeatable because the seed is fixed.

## Example notebooks (`examples/`)

| Notebook | What it does |
|---|---|
| `01_pld_picker` | Choose the deconvolution window station by station with sliders, then accept or reject each station. The settings are saved to `picks.csv`. |
| `02_astf_retrieval` | Re-create all ASTFs from `picks.csv`, build the duration table, locate the EGF relative to the target, and plot the ASTFs by azimuth. |
| `03_sdm_inversion` | Choose the fault plane, run the Bayesian inversion, check convergence, and draw the summary and corner plots. |

The example event is the 2020-05-03 Mw 3.4 Haenam earthquake (South Korea), with 101 station–phase pairs (P, S and SmS).

## Using your own data

- **Waveforms**: one folder per event with `Main/` (target) and `EGF/` subfolders of SAC files named
  `<origin>.<net>.<sta>.<channel><component>.sac`, with the SAC reference time at each event's origin.
  - Components: `Z` for P, `T` for S and SmS.
  - Arrival times are read from the target headers: `a` for P and SmS, `t0` for S.
- **Picks** (`picks.csv`), one row per station and phase:
  `phase, network, station, channel, start, duration, shift, freqmin, freqmax, iter_max, interpolate`.
  The picker writes this file.
- **Duration table** (`tauc_table.csv`), the input of the inversion:
  `code, latitude, longitude, azimuth, vr, tauc, phase, Used, t1, toa`.
  - Only ASTFs with VR ≥ 70 % are kept.
  - Take-off angles use a southern Korean Peninsula model by default. For another region, pass
    `model=geometry.taup_model("your_model.nd")`.
- **Priors**: the default prior bounds (`bayesian.Priors`) suit about Mw 2.5–4. Widen them for larger events.

## Units and conventions

| Quantity | Unit or convention |
|---|---|
| `Lc`, `Wc` | m |
| `tauc` | s |
| `v0` | m/s |
| Stress drop | MPa |
| Seismic moment | M0 = 10^(1.5 Mw + 9.05) N·m |
| `rake` (rupture direction), `angle` (major axis) | Degrees in the fault plane, measured from strike, with up-dip at 90° |

The stress drop is for an elliptical crack with semi-axes Lc and Wc. For an ideal crack this is 1.40 times the
crack's own value; `semi_axis_factor=np.sqrt(5)/2` returns that value instead.

Tests: `pytest -q`.

## Citation and license

MIT license. Please cite Seo et al. (2026), *Rupture properties, source scaling, and directivity of small
intraplate earthquakes: insights from a dense seismic network and Moho-reflected phases in the southern
Korean Peninsula*, J. Geophys. Res. Solid Earth. Please also cite BayesBay, CVXPY and ObsPy.
