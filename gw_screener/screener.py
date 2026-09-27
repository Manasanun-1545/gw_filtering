import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astroquery.ipac.ned import Ned
import math, re, time
from scipy.stats import linregress
import warnings

warnings.filterwarnings("ignore")

class GWCandidateScreener:
    def __init__(self, 
                 z_max=1.0, 
                 min_points=2, 
                 offset_max=30.0, 
                 slope_min=1.860, 
                 slope_marginal=0.075,
                 grb_frac_min=0.80,
                 t_window=1.0,
                 host_matching_err=2.0,
                 v_peculiar=300.0):
        
        # เซตค่า Default ตามที่คุณต้องการไว้ให้แล้วครับ
        self.z_max = z_max
        self.min_points = min_points
        self.offset_max = offset_max
        self.slope_min = slope_min
        self.slope_marginal = slope_marginal
        self.grb_frac_min = grb_frac_min
        self.t_window = t_window
        self.host_matching_err = host_matching_err
        self.v_peculiar = v_peculiar
        
        self.cosmo = FlatLambdaCDM(H0=69.6, Om0=0.286)
        self.rng = np.random.default_rng(42)  # ล็อกตัวเลขสุ่ม
        self.n_mc = 2000
        self.c_kms = 299792.458
        self.galaxy_types = ("G", "GPair", "GTrpl", "GGroup", "GClstr", "PofG")
        self.bloom_norm = 1.0 / (0.33 * math.log(10.0))
        self.pcc_ratio = 5.0
        self.grb_tol = 0.5
        
        self.grb_values = {
            "Lower limit": {"F0": 0.2e-26, "t0": 0.0002, "s": 1.0, "a": 2.0, "b": -0.8},
            "Upper limit": {"F0": 0.5e-23, "t0": 0.0002, "s": 1.0, "a": 2.0, "b": -0.8},
        }

    def _calculate_pcc(self, theta_arcsec, host_mag):
        try:
            m = float(host_mag) if not pd.isna(host_mag) else 20.0
        except (TypeError, ValueError):
            m = 20.0
        sigma_m = self.bloom_norm * (10 ** (0.33 * (m - 24) + 4.55)) / (3600.0**2)
        return -math.expm1(-np.pi * float(theta_arcsec) ** 2 * sigma_m)

    def _decode_type(self, value):
        return value.decode("utf-8", "ignore") if isinstance(value, (bytes, bytearray)) else str(value)

    def _parse_ned_mag(self, value):
        s = self._decode_type(value)
        found = re.search(r"[-+]?\d+\.?\d*", s.strip())
        return float(found.group()) if found else np.nan

    def _corrected_F(self, t, F0, t0, s, a, b):
        t = np.asarray(t, dtype=float)
        return F0 * ((t / t0) ** (-s * a) + (t / t0) ** (-s * b)) ** (-1 / s)

    def _fnu_to_ab(self, fnu):
        return -2.5 * np.log10(np.asarray(fnu, float)) - 48.60

    def _check_grb_boundary(self, t_days, mags, z):
        if z >= self.z_max: return np.nan
        t, m = np.asarray(t_days, float), np.asarray(mags, float)
        ok = np.isfinite(t) & np.isfinite(m) & (t > 0)
        if ok.sum() < 2: return np.nan
        hi = self._fnu_to_ab(self._corrected_F(t[ok], **self.grb_values["Upper limit"]))
        lo = self._fnu_to_ab(self._corrected_F(t[ok], **self.grb_values["Lower limit"]))
        return float(np.mean((m[ok] >= np.minimum(hi, lo) - self.grb_tol) &
                             (m[ok] <= np.maximum(hi, lo) + self.grb_tol)))

    def _mc_offset(self, theta_arcsec, z, z_err):
        true_offset = theta_arcsec * (self.cosmo.kpc_proper_per_arcmin(z).value / 60.0)
        sig_z = math.hypot(float(z_err), self.v_peculiar / self.c_kms)
        theta_s = np.abs(self.rng.normal(theta_arcsec, self.host_matching_err, self.n_mc))
        z_s = np.clip(self.rng.normal(z, sig_z, self.n_mc), 1e-5, None)
        off_distribution = theta_s * (self.cosmo.kpc_proper_per_arcmin(z_s).value / 60.0)
        p16, p84 = np.percentile(off_distribution, [16, 84])
        error_sig = (p84 - p16) / 2.0
        return float(true_offset), float(error_sig), float(p16), float(p84)

    def _offset_tier(self, offset_kpc, sigma_kpc, unambiguous):
        if not np.isfinite(offset_kpc) or not np.isfinite(sigma_kpc) or offset_kpc <= 0: return "D"
        rel = sigma_kpc / offset_kpc
        if rel < 0.20 and unambiguous: return "A"
        if rel < 0.35 and unambiguous: return "B"
        if rel < 0.60: return "C"
        return "D"

    def _find_host(self, sn_pos):
        out = dict(pos=None, name=None, mag=np.nan, theta=np.nan, pcc=np.nan, pcc_second=np.nan, n_gal=0, unambiguous=False)
        try:
            catalog = Ned.query_region(sn_pos, radius=3 * u.arcmin)
        except:
            return out
        cols = {c.lower(): c for c in catalog.colnames}
        type_col, name_col = cols.get("type"), cols.get("object name")
        mag_col = next((cols[c] for c in cols if "magnitude" in c), None)

        cand = []
        for gal in catalog:
            if type_col is not None and self._decode_type(gal[type_col]).strip() not in self.galaxy_types: continue
            try: gal_pos = SkyCoord(float(gal["RA"]) * u.deg, float(gal["DEC"]) * u.deg)
            except: continue
            theta = sn_pos.separation(gal_pos).arcsecond
            if theta <= 0: continue
            gal_mag = self._parse_ned_mag(gal[mag_col]) if mag_col else np.nan
            cand.append((self._calculate_pcc(theta, gal_mag), theta, gal_pos, gal_mag, str(gal[name_col]) if name_col else None))

        if not cand: return out
        cand.sort(key=lambda c: c[0])
        pcc, theta, pos, mag, nm = cand[0]
        second = cand[1][0] if len(cand) > 1 else np.nan
        unamb = (not np.isfinite(second)) or (pcc > 0 and second / pcc >= self.pcc_ratio)
        return dict(pos=pos, name=nm, mag=mag, theta=theta, pcc=pcc, pcc_second=second, n_gal=len(cand), unambiguous=bool(unamb))

    def _add_coords_from_ned(self, df, delay=1.0):
        d = df.copy()
        cache = {}
        for nm in d["name"].dropna().unique():
            try:
                r = Ned.query_object(str(nm))
                cache[nm] = f"{float(r['RA'][0])}, {float(r['DEC'][0])}"
            except:
                cache[nm] = None
            time.sleep(delay)
        d["locat_trans"] = d["name"].map(cache)
        
        ok = sum(v is not None for v in cache.values()) 
        fail = sum(v is None for v in cache.values()) 
        n_drop = d["locat_trans"].isna().sum() 
        print(f"  NED Query: {ok} found, {fail} failed, {n_drop} rows dropped")
        
        return d.dropna(subset=["locat_trans"])

    def _filter_events(self, df, label="", verbose=True):
        d = df.copy()

        t_candidates = ["t-t0", "time", "t", "t0-t", "time(days)", "time_rest_days", "dt", "phase"]
        t_col = next((c for c in t_candidates if c in d.columns), None)
        m_col = "mag" if "mag" in d.columns else ("ab magnitude" if "ab magnitude" in d.columns else "magnitude")

        if not t_col: 
            if verbose: print(f"  {label:<18} ⚠️ ไม่เจอคอลัมน์เวลา (มี: {list(d.columns)})  -> ข้าม")
            return d.iloc[:0]
            
        if "name" not in d.columns: 
            if verbose: print(f"  {label:<18} ⚠️ ไม่มีคอลัมน์ name  -> ข้าม")
            return d.iloc[:0]

        d["_t"] = pd.to_numeric(d[t_col], errors="coerce")
        d["_m"] = pd.to_numeric(d[m_col], errors="coerce")
        valid = np.isfinite(d["_t"]) & np.isfinite(d["_m"]) & (d["_t"] > 0)
        n_pts = d[valid].groupby("name").size()

        has_z = "z" in d.columns
        if has_z:
            z_ev = d.groupby("name")["z"].apply(
                lambda s: pd.to_numeric(s, errors="coerce").dropna().iloc[0] 
                if pd.to_numeric(s, errors="coerce").notna().any() else np.nan)
        else:
            z_ev = pd.Series(0.0, index=n_pts.index)

        keep, drop_pts, drop_z = [], [], []
        for ev in d["name"].dropna().unique():
            if n_pts.get(ev, 0) < self.min_points:
                drop_pts.append(ev)
            elif has_z and not (np.isfinite(z_ev.get(ev, np.nan)) and z_ev[ev] < self.z_max):
                drop_z.append(ev)
            else:
                keep.append(ev)

        if verbose:
            warn = "" if has_z else "   ⚠️ ไม่มีคอลัมน์ z"
            print(f"  {label:<18} total {len(d['name'].dropna().unique()):>3}  "
                  f"->  keep {len(keep):>3}   "
                  f"(drop: pts<{self.min_points}: {len(drop_pts)}, z>={self.z_max}: {len(drop_z)}){warn}")

        return d[d["name"].isin(keep)].drop(columns=["_t", "_m"])

    def process_data(self, data, label="transient", query_ned=False, verbose=True):
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            if str(data).endswith(".xlsx"):
                df = pd.read_excel(data, engine="openpyxl")
            else:
                df = pd.read_csv(data)
                
        df.columns = df.columns.str.lower().str.strip()
        
        # [แก้ไขแล้ว]: เติมชื่อลงช่องว่างให้ครบทุกบรรทัดตั้งแต่แรกสุด 
        # ไม่ว่าจะรันผ่าน NED หรือไม่ก็ตาม ข้อมูล photometry จะได้ไม่โดนเตะทิ้ง[cite: 1]
        ncol = next((c for c in ["name", "grb", "grb name"] if c in df.columns), None)
        if ncol:
            df[ncol] = df[ncol].ffill()
            if ncol != "name":
                df["name"] = df[ncol]
        
        if query_ned and "locat_trans" not in df.columns:
            if verbose: print(f"Querying NED for coordinates ({label})...")
            df = self._add_coords_from_ned(df)
            
        df_filtered = self._filter_events(df, label=label, verbose=verbose)
        if df_filtered.empty:
            if verbose: print(f"  {label}: No events passed pre-filtering.")
            return pd.DataFrame()

        results = []
        if verbose: print(f"--- Running Pipeline [{label}]: {df_filtered['name'].nunique()} objects ---")

        for name, g in df_filtered.groupby("name"):
            try:
                row = g.iloc[0]
                ra_s, dec_s = str(row["locat_trans"]).replace(" ", "").split(",")
                sn_pos = SkyCoord(float(ra_s) * u.deg, float(dec_s) * u.deg, frame="icrs")

                has_z = ("z" in row) and (not pd.isna(row["z"]))
                z = float(row["z"]) if has_z else 0.01
                z_err = float(row["z_err"]) if ("z_err" in row and not pd.isna(row["z_err"])) else 0.001

                host = self._find_host(sn_pos)
                if host["pos"] is not None:
                    off, off_err, p16, p84 = self._mc_offset(host["theta"], z, z_err)
                else: off = off_err = p16 = p84 = np.nan
                tier = self._offset_tier(off, off_err, host["unambiguous"])

                t_col = next((c for c in ["t-t0", "time", "t0-t", "t", "time(days)", "time_rest_days"] if c in g.columns), None)
                m_col = "mag" if "mag" in g.columns else "magnitude"

                sub = g[[c for c in (t_col, m_col, "filter") if c in g.columns]].copy()
                sub[t_col], sub[m_col] = pd.to_numeric(sub[t_col], errors="coerce"), pd.to_numeric(sub[m_col], errors="coerce")
                sub = sub.dropna(subset=[t_col, m_col])

                win = sub[(sub[t_col] > 0) & (sub[t_col] <= self.t_window)].copy()
                filt_used = "mixed"
                if "filter" in win.columns and win["filter"].notna().any():
                    counts = win["filter"].astype(str).value_counts()
                    if len(counts) and counts.iloc[0] >= 2:
                        filt_used = counts.index[0]
                        win = win[win["filter"].astype(str) == filt_used]

                slope = np.nan
                if len(win) >= 2:
                    win = win.sort_values(t_col)
                    t_v, m_v = win[t_col].values, win[m_col].values
                    logt = np.log10(t_v)
                    if len(t_v) >= 3 and np.ptp(logt) > 0: slope = abs(float(linregress(logt, m_v).slope))
                    elif len(t_v) == 2 and np.ptp(logt) > 0: slope = abs(float((m_v[1] - m_v[0]) / (logt[1] - logt[0])))

                grb_frac = self._check_grb_boundary(sub[t_col].values, sub[m_col].values, z) if len(sub) >= 2 else np.nan

                ref = float(row["offset"]) if ("offset" in row and not pd.isna(row["offset"])) else np.nan
                diff = abs(off - ref) if np.isfinite(off) and np.isfinite(ref) else np.nan
                pull = ((off - ref) / off_err) if np.isfinite(diff) and np.isfinite(off_err) and off_err > 0 else np.nan

                f_off = "UNKNOWN" if not np.isfinite(off) else ("PASS" if off < self.offset_max else "FAIL")
                f_slope = "UNKNOWN"
                if np.isfinite(slope):
                    if slope >= self.slope_min: f_slope = "PASS"
                    elif slope >= self.slope_marginal: f_slope = "MARGINAL"
                    else: f_slope = "FAIL"
                f_grb = "UNKNOWN" if not np.isfinite(grb_frac) else ("PASS" if grb_frac >= self.grb_frac_min else "FAIL")

                n_pass = sum(f == "PASS" for f in (f_off, f_slope, f_grb))
                n_marginal = sum(f == "MARGINAL" for f in (f_off, f_slope, f_grb))
                
                if f_slope == "UNKNOWN" and f_grb == "UNKNOWN": verdict = "INSUFFICIENT"
                elif n_pass == 3: verdict = "GOLD" if host["unambiguous"] else "SILVER"
                elif n_pass == 2 and n_marginal == 1: verdict = "SILVER" if host["unambiguous"] else "BRONZE"
                elif n_pass >= 2: verdict = "BRONZE"
                elif n_pass == 1 and n_marginal >= 1: verdict = "BRONZE"
                else: verdict = "REJECT"

                results.append({
                    "name": name, "z": z, "host_name": host["name"], "host_mag": host["mag"],
                    "theta_arcsec": host["theta"], "n_galaxies": host["n_gal"], "host_unambiguous": host["unambiguous"],
                    "pcc": host["pcc"], "pcc_second": host["pcc_second"], "offset_kpc": off, "offset_err": off_err, 
                    "offset_p16": p16, "offset_p84": p84, "offset_tier": tier, "ned_ref_offset": ref, "diff_kpc": diff, "pull": pull,
                    "slope": slope, "filter_used": filt_used, "grb_frac_inside": grb_frac,
                    "flag_offset": f_off, "flag_slope": f_slope, "flag_grb": f_grb,
                    "n_pass": n_pass, "verdict": verdict, "source": label,
                })
                if verbose:
                    o = f"{off:7.2f}" if np.isfinite(off) else "    n/a"
                    v = f"{slope:.3f}" if np.isfinite(slope) else "  n/a"
                    print(f"  {str(name):<22} {verdict:<12} offset={o} kpc  tier={tier}  slope={v}")
            except Exception as e:
                if verbose: print(f"  x {name}: Error - {e}")

        return pd.DataFrame(results)

    def plot_diagnostics(self, results_df):
        if results_df.empty:
            print("ไม่มีข้อมูลสำหรับสร้างกราฟครับ")
            return

        FLAG_COLOR = {"PASS": "#2a9d5c", "MARGINAL": "#f1c40f", "FAIL": "#c1443c", "UNKNOWN": "#9aa0a6"}
        VERDICT_ORDER = ["GOLD", "SILVER", "BRONZE", "REJECT", "INSUFFICIENT"]
        VERDICT_COLOR = {"GOLD": "#d4af37", "SILVER": "#9aa0a6", "BRONZE": "#b87333", "REJECT": "#c1443c", "INSUFFICIENT": "#555555"}
        FLAG_COLS = ["flag_offset", "flag_slope", "flag_grb"]

        groups = results_df["source"].unique()
        if len(groups) == 0: return

        fig, axes = plt.subplots(len(groups), 3, figsize=(15, 3.6 * len(groups)), squeeze=False)
        for r, grp in enumerate(groups):
            d = results_df[results_df["source"] == grp]
            ax1, ax2, ax3 = axes[r]

            tab = pd.DataFrame({c.replace("flag_", ""): d[c].value_counts() for c in FLAG_COLS})
            tab = tab.reindex(["PASS", "MARGINAL", "FAIL", "UNKNOWN"]).fillna(0).astype(int)
            labels = list(tab.columns)
            y_pos = np.arange(len(labels))
            left = np.zeros(len(labels))
            for status in ["PASS", "MARGINAL", "FAIL", "UNKNOWN"]:
                vals = tab.loc[status].values if status in tab.index else np.zeros(len(labels))
                ax1.barh(y_pos, vals, left=left, color=FLAG_COLOR[status], label=status if r == 0 else None)
                left += vals
            ax1.set_yticks(y_pos); ax1.set_yticklabels(labels); ax1.set_xlabel("N objects"); ax1.set_ylabel(f"{grp}\n(N={len(d)})", fontweight="bold")
            if r == 0: ax1.set_title("C1: outcome per axis"); ax1.legend(fontsize=8, loc="lower right")

            vc = d["verdict"].value_counts().reindex(VERDICT_ORDER).fillna(0)
            ax2.bar(range(len(vc)), vc.values, color=[VERDICT_COLOR[k] for k in vc.index])
            ax2.set_xticks(range(len(vc))); ax2.set_xticklabels(vc.index, rotation=25, ha="rig ht", fontsize=9); ax2.set_ylabel("N objects")
            if r == 0: ax2.set_title("C2: final verdict")

            tc = d["offset_tier"].value_counts().reindex(list("ABCD")).fillna(0)
            ax3.bar(range(4), tc.values, color="#4c72b0")
            ax3.set_xticks(range(4)); ax3.set_xticklabels([f"tier {t}" for t in "ABCD"]); ax3.set_ylabel("N objects")
            if r == 0: ax3.set_title("C3: offset reliability tier")

        fig.suptitle("Diagnostics Report", fontsize=14, y=1.005)
        plt.tight_layout()
        plt.show()