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

## Citation

If you use bayes2mt, please cite

> Seo, M.-S., Han, S., Kim, W.-Y., & Kim, Y. (2026). Rupture properties, source scaling, and directivity of
> small intraplate earthquakes: insights from a dense seismic network and Moho-reflected phases in the southern
> Korean Peninsula. *Journal of Geophysical Research: Solid Earth* (in revision).

together with the methods and software below that your work relies on. The data and scripts of the paper are
archived at [doi:10.5281/zenodo.19887457](https://doi.org/10.5281/zenodo.19887457).

### Methods

| What bayes2mt does | Module | References |
|---|---|---|
| Second-degree moments from apparent durations | `geometry`, `moments` | [Backus (1977)](https://doi.org/10.1111/j.1365-246X.1977.tb04187.x); [Silver (1983)](https://doi.org/10.1785/BSSA07306A1499); [McGuire (2004)](https://doi.org/10.1785/0120030091) |
| ASTFs by EGF deconvolution (projected Landweber) | `pld` | [Mori & Frankel (1990)](https://pubs.usgs.gov/publication/70016082); [Bertero et al. (1997)](https://doi.org/10.1088/0266-5611/13/2/017); [Lanza et al. (1999)](https://doi.org/10.1046/j.1365-246x.1999.00809.x) |
| Deterministic inversion (semidefinite program) | `deterministic` | [McGuire (2004)](https://doi.org/10.1785/0120030091); [McGuire (2017)](https://doi.org/10.1785/0220160170) |
| Bayesian inversion: Cholesky parameterization | `bayesian` | [Atterholt & Ross (2022)](https://doi.org/10.1029/2021JB023780) |
| Bayesian inversion: sampled data noise (hierarchical) | `bayesian` | [Malinverno & Briggs (2004)](https://doi.org/10.1190/1.1778243) |
| Representative model (medoid) | `moments` | [Mills et al. (2025)](https://doi.org/10.1029/2024JH000472) |
| Stress drop of an elliptical crack | `stressdrop` | [Eshelby (1957)](https://doi.org/10.1098/rspa.1957.0133); [McGuire & Kaneko (2018)](https://doi.org/10.1093/gji/ggy201) |
| Default velocity model for take-off angles | `geometry` | Kim & Kim (1983), as a 1-D gradient model |

### Software

| Package | Used for | References |
|---|---|---|
| [BayesBay](https://bayes-bay.readthedocs.io) | Markov chain Monte Carlo sampling | [Magrini et al. (2025)](https://doi.org/10.1785/0220240275) |
| [CVXPY](https://www.cvxpy.org) with [SCS](https://www.cvxgrp.org/scs/) | Semidefinite program | [Diamond & Boyd (2016)](https://jmlr.org/papers/v17/15-408.html); [Agrawal et al. (2018)](https://doi.org/10.1080/23307706.2017.1397554); [O'Donoghue et al. (2016)](https://doi.org/10.1007/s10957-016-0892-3) |
| [ObsPy](https://docs.obspy.org) (with TauP) | Waveforms, take-off angles, beach balls | [Beyreuther et al. (2010)](https://doi.org/10.1785/gssrl.81.3.530); [Crotwell et al. (1999)](https://doi.org/10.1785/gssrl.70.2.154) |
| [ArviZ](https://python.arviz.org) | Convergence diagnostics, density estimates | [Kumar et al. (2019)](https://doi.org/10.21105/joss.01143); [Martin et al. (2026)](https://doi.org/10.21105/joss.09889) |
| [NumPy](https://numpy.org), [SciPy](https://scipy.org) | Numerical computing | [Harris et al. (2020)](https://doi.org/10.1038/s41586-020-2649-2); [Virtanen et al. (2020)](https://doi.org/10.1038/s41592-019-0686-2) |
| [Numba](https://numba.pydata.org) | Fast medoid search | [Lam et al. (2015)](https://doi.org/10.1145/2833157.2833162) |
| [pandas](https://pandas.pydata.org) | Tables | [McKinney (2010)](https://doi.org/10.25080/Majora-92bf1922-00a) |
| [Matplotlib](https://matplotlib.org) | Figures | [Hunter (2007)](https://doi.org/10.1109/MCSE.2007.55) |
| [ipywidgets](https://ipywidgets.readthedocs.io) | Interactive picker | — |

<details>
<summary>Full references</summary>

- Agrawal, A., Verschueren, R., Diamond, S., & Boyd, S. (2018). A rewriting system for convex optimization problems. *Journal of Control and Decision*, 5(1), 42–60. https://doi.org/10.1080/23307706.2017.1397554
- Atterholt, J., & Ross, Z. E. (2022). Bayesian framework for inversion of second-order stress glut moments: Application to the 2019 Ridgecrest sequence mainshock. *Journal of Geophysical Research: Solid Earth*, 127(4). https://doi.org/10.1029/2021JB023780
- Backus, G. E. (1977). Interpreting the seismic glut moments of total degree two or less. *Geophysical Journal International*, 51(1), 1–25. https://doi.org/10.1111/j.1365-246X.1977.tb04187.x
- Bertero, M., Bindi, D., Boccacci, P., Cattaneo, M., Eva, C., & Lanza, V. (1997). Application of the projected Landweber method to the estimation of the source time function in seismology. *Inverse Problems*, 13(2), 465–486. https://doi.org/10.1088/0266-5611/13/2/017
- Beyreuther, M., Barsch, R., Krischer, L., Megies, T., Behr, Y., & Wassermann, J. (2010). ObsPy: A Python toolbox for seismology. *Seismological Research Letters*, 81(3), 530–533. https://doi.org/10.1785/gssrl.81.3.530
- Crotwell, H. P., Owens, T. J., & Ritsema, J. (1999). The TauP Toolkit: Flexible seismic travel-time and ray-path utilities. *Seismological Research Letters*, 70(2), 154–160. https://doi.org/10.1785/gssrl.70.2.154
- Diamond, S., & Boyd, S. (2016). CVXPY: A Python-embedded modeling language for convex optimization. *Journal of Machine Learning Research*, 17(83), 1–5. https://jmlr.org/papers/v17/15-408.html
- Eshelby, J. D. (1957). The determination of the elastic field of an ellipsoidal inclusion, and related problems. *Proceedings of the Royal Society of London. Series A*, 241(1226), 376–396. https://doi.org/10.1098/rspa.1957.0133
- Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). Array programming with NumPy. *Nature*, 585(7825), 357–362. https://doi.org/10.1038/s41586-020-2649-2
- Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90–95. https://doi.org/10.1109/MCSE.2007.55
- Kim, S. J., & Kim, S. G. (1983). A study on the crustal structure of South Korea by using seismic waves. *Economic and Environmental Geology*, 16(1), 51–61.
- Kumar, R., Carroll, C., Hartikainen, A., & Martin, O. (2019). ArviZ a unified library for exploratory analysis of Bayesian models in Python. *Journal of Open Source Software*, 4(33), 1143. https://doi.org/10.21105/joss.01143
- Lam, S. K., Pitrou, A., & Seibert, S. (2015). Numba: A LLVM-based Python JIT compiler. *Proceedings of the Second Workshop on the LLVM Compiler Infrastructure in HPC*, 1–6. https://doi.org/10.1145/2833157.2833162
- Lanza, V., Spallarossa, D., Cattaneo, M., Bindi, D., & Augliera, P. (1999). Source parameters of small events using constrained deconvolution with empirical Green's functions. *Geophysical Journal International*, 137(3), 651–662. https://doi.org/10.1046/j.1365-246x.1999.00809.x
- Magrini, F., He, J., & Sambridge, M. (2025). BayesBay: a versatile Bayesian inversion framework written in Python. *Seismological Research Letters*, 96(3), 2052–2064. https://doi.org/10.1785/0220240275
- Malinverno, A., & Briggs, V. A. (2004). Expanded uncertainty quantification in inverse problems: Hierarchical Bayes and empirical Bayes. *Geophysics*, 69(4), 1005–1016. https://doi.org/10.1190/1.1778243
- Martin, O. A., Abril-Pla, O., Deklerk, J., Axen, S. D., Carroll, C., Hartikainen, A., & Vehtari, A. (2026). ArviZ: a modular and flexible library for exploratory analysis of Bayesian models. *Journal of Open Source Software*, 11(119), 9889. https://doi.org/10.21105/joss.09889
- McGuire, J. J. (2004). Estimating finite source properties of small earthquake ruptures. *Bulletin of the Seismological Society of America*, 94(2), 377–393. https://doi.org/10.1785/0120030091
- McGuire, J. J. (2017). A MATLAB toolbox for estimating the second moments of earthquake ruptures. *Seismological Research Letters*, 88(2A), 371–378. https://doi.org/10.1785/0220160170
- McGuire, J. J., & Kaneko, Y. (2018). Directly estimating earthquake rupture area using second moments to reduce the uncertainty in stress drop. *Geophysical Journal International*, 214(3), 2224–2235. https://doi.org/10.1093/gji/ggy201
- McKinney, W. (2010). Data structures for statistical computing in Python. *Proceedings of the 9th Python in Science Conference*, 56–61. https://doi.org/10.25080/Majora-92bf1922-00a
- Meng, H., McGuire, J. J., & Ben-Zion, Y. (2020). Semiautomated estimates of directivity and related source properties of small to moderate Southern California earthquakes using second seismic moments. *Journal of Geophysical Research: Solid Earth*, 125(4). https://doi.org/10.1029/2019JB018566
- Mills, C. L., Rudolph, M. L., & Lekić, V. (2025). Exploring and summarizing ensemble solutions to geophysical inverse problems. *Journal of Geophysical Research: Machine Learning and Computation*, 2(4). https://doi.org/10.1029/2024JH000472
- Mori, J., & Frankel, A. (1990). Source parameters for small events associated with the 1986 North Palm Springs, California, earthquake determined using empirical Green functions. *Bulletin of the Seismological Society of America*, 80(2), 278–295. https://pubs.usgs.gov/publication/70016082
- O'Donoghue, B., Chu, E., Parikh, N., & Boyd, S. (2016). Conic optimization via operator splitting and homogeneous self-dual embedding. *Journal of Optimization Theory and Applications*, 169(3), 1042–1068. https://doi.org/10.1007/s10957-016-0892-3
- Silver, P. (1983). Retrieval of source-extent parameters and the interpretation of corner frequency. *Bulletin of the Seismological Society of America*, 73(6A), 1499–1511. https://doi.org/10.1785/BSSA07306A1499
- Virtanen, P., Gommers, R., Oliphant, T. E., et al. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods*, 17(3), 261–272. https://doi.org/10.1038/s41592-019-0686-2

</details>

## Acknowledgments

The deterministic inversion follows the MATLAB toolbox of [McGuire (2017)](https://doi.org/10.1785/0220160170) and the scripts of
[Meng et al. (2020)](https://doi.org/10.1029/2019JB018566).
We thank Jeff McGuire and Haoran Meng for making them publicly available.

## License

MIT (see `LICENSE`).
