# gw_filtering
Filtering pipeline for optical counterparts of GW events (fast transient vs SN)
A pipeline for screening optical counterpart candidates of gravitational-wave (GW) events. It separates **fast transients** (kilonova / short-GRB-like) from **supernovae** using three independent criteria: host-galaxy offset, early decline rate, and consistency with a GRB afterglow corridor.

Each candidate gets a flag for each criterion and a final verdict (`GOLD` / `SILVER` / `BRONZE` / `REJECT` / `INSUFFICIENT`).

---

## Installation

```bash
git clone https://github.com/yourusername/gw_screener.git
cd gw_screener_project
pip install -e .
```

Requirements (installed automatically): `numpy`, `pandas`, `matplotlib`, `astropy`, `astroquery`, `scipy`, `openpyxl`. Python ≥ 3.8.

The pipeline queries **NED** (via `astroquery.ipac.ned`), so it needs internet access.

### Project layout

```
gw_screener_project/
├── pyproject.toml
├── readme.md
└── src/
    └── gw_screener/
        ├── __init__.py      # exports GWCandidateScreener
        └── screener.py      # pipeline
```

---

## Quick start

```python
from gw_screener import GWCandidateScreener
import pandas as pd

scr = GWCandidateScreener()                      # default thresholds

ft = scr.process_data("fast_transients.xlsx", label="Fast transient", query_ned=True)
sn = scr.process_data("supernovae.csv",       label="SN")

results = pd.concat([ft, sn], ignore_index=True)
results.to_csv("pipeline_results_all.csv", index=False)

scr.plot_diagnostics(results)
```

`process_data` accepts a file path (`.xlsx` or `.csv`) or a `pandas.DataFrame`.

---

## Input format

One row per photometric point. Column names are case-insensitive and stripped of spaces.

| Column | Required | Description |
|---|---|---|
| `name` (or `grb`, `grb name`) | yes | Object ID. Blank cells are forward-filled, so the name only needs to appear on the first row of each object. |
| time column | yes | Days since t0. The first match is used from: `t-t0`, `time`, `t0-t`, `t`, `time(days)`, `time_rest_days` (pre-filter also accepts `dt`, `phase`). |
| `mag` or `magnitude` | yes | Apparent magnitude. |
| `locat_trans` | yes* | Transient position as `"RA, Dec"` in degrees. *If missing, set `query_ned=True` to look up coordinates from NED by `name`. |
| `z` | recommended | Redshift. Objects with `z ≥ z_max` are dropped. If missing, `z = 0.01` is assumed. |
| `z_err` | optional | Redshift uncertainty (default `0.001`). |
| `filter` | optional | Band name. Used to pick a single band for the slope. |
| `offset` | optional | Literature offset in kpc, used only for comparison (`diff_kpc`, `pull`). |

**Pre-filtering:** an object is kept only if it has at least `min_points` valid points (t > 0) and `z < z_max`.

---

## Method

### 1. Host galaxy and offset

1. Query NED within **3′** of the transient and keep only galaxy types (`G`, `GPair`, `GTrpl`, `GGroup`, `GClstr`, `PofG`).
2. Compute the chance-coincidence probability P_cc for each galaxy (Bloom et al. 2002). If NED has no magnitude, `m = 20` is assumed.
3. The galaxy with the lowest P_cc is taken as the host.
4. The host is **unambiguous** if there is no second galaxy, or if P_cc(2nd) / P_cc(1st) ≥ `pcc_ratio` (5).
5. Projected offset (kpc) is computed from the angular separation and z, using a flat ΛCDM cosmology (H0 = 69.6, Ωm = 0.286).
6. Uncertainty is estimated with a Monte Carlo (2000 draws, fixed seed 42): angular error = `host_matching_err`, redshift error = z_err ⊕ `v_peculiar`/c. The reported error is half the 16–84 percentile range.

**Offset tier** (reliability of the measurement, based on σ/offset):

| Tier | Condition |
|---|---|
| A | σ/offset < 0.20 and unambiguous host |
| B | σ/offset < 0.35 and unambiguous host |
| C | σ/offset < 0.60 |
| D | otherwise, or offset not measurable |

### 2. Decline slope

Uses points with `0 < t ≤ t_window` (1 day). If a `filter` column exists, the band with the most points is used (needs ≥ 2 points). The slope is |dm / d log10 t|: a linear fit for ≥ 3 points, or the two-point slope for 2 points.

### 3. GRB afterglow corridor

The corridor is bounded by two smoothly broken power laws in F_ν (upper and lower limits), converted to AB mag, with a tolerance of `grb_tol` = 0.5 mag. The pipeline reports the fraction of points inside the corridor (`grb_frac_inside`). Not computed for z ≥ `z_max`.

### 4. Flags and verdict

| Criterion | PASS | MARGINAL | FAIL |
|---|---|---|---|
| Offset | offset < `offset_max` | — | otherwise |
| Slope | slope ≥ `slope_min` | `slope_marginal` ≤ slope < `slope_min` | otherwise |
| GRB corridor | fraction ≥ `grb_frac_min` | — | otherwise |

A flag is `UNKNOWN` when the quantity cannot be measured.

| Verdict | Condition |
|---|---|
| GOLD | 3 PASS and unambiguous host |
| SILVER | 3 PASS with ambiguous host, or 2 PASS + 1 MARGINAL with unambiguous host |
| BRONZE | 2 PASS (other cases), or 1 PASS + ≥ 1 MARGINAL |
| REJECT | otherwise |
| INSUFFICIENT | slope and GRB flags are both UNKNOWN |

---

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `z_max` | 1.0 | Maximum redshift (GRB model not used above this) |
| `min_points` | 2 | Minimum number of photometric points |
| `offset_max` | 30.0 | Offset threshold (kpc) |
| `slope_min` | 1.860 | Slope threshold for PASS (mag / dex) |
| `slope_marginal` | 0.075 | Lower bound for MARGINAL |
| `grb_frac_min` | 0.80 | Minimum fraction of points inside GRB corridor |
| `t_window` | 1.0 | Time window for the slope (days) |
| `host_matching_err` | 2.0 | Angular uncertainty for the offset MC (arcsec) |
| `v_peculiar` | 300.0 | Peculiar velocity added to the z error (km/s) |

Example:

```python
scr = GWCandidateScreener(offset_max=20.0, t_window=2.0)
```

---

## Output

`process_data` returns one row per object:

| Column | Description |
|---|---|
| `name`, `source`, `z` | Object ID, input label, redshift used |
| `host_name`, `host_mag`, `theta_arcsec`, `n_galaxies` | Selected host and number of galaxies within 3′ |
| `host_unambiguous`, `pcc`, `pcc_second` | Host decision and P_cc of the 1st and 2nd galaxies |
| `offset_kpc`, `offset_err`, `offset_p16`, `offset_p84`, `offset_tier` | Offset and its uncertainty |
| `ned_ref_offset`, `diff_kpc`, `pull` | Comparison with the input `offset` column |
| `slope`, `filter_used` | Decline slope and band used |
| `grb_frac_inside` | Fraction of points inside the GRB corridor |
| `flag_offset`, `flag_slope`, `flag_grb`, `n_pass`, `verdict` | Flags and final verdict |

`plot_diagnostics(results)` draws three panels per input group: C1 flag outcome per criterion, C2 verdict counts, C3 offset tier counts.

---

## Notes

- NED queries are slow. `query_ned=True` waits 1 s between objects.
- Objects without a NED galaxy within 3′ get `offset = NaN` and `flag_offset = UNKNOWN`.
- Results are reproducible because the random seed is fixed (42).

## References

- Bloom, Kulkarni & Djorgovski 2002, AJ, 123, 1111 — chance-coincidence probability
- Rastinejad et al. 2022 — cosmology parameters

## Author

Manasanun Tanasan — Cchann491@gmail.com
