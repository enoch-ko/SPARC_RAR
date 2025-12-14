import pandas as pd
import numpy as _np
import sys
import os

import jax.numpy as jnp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from SPARC_RAR.rm_hooks import vel2acc
from SPARC_RAR.vcdisk_differentiable import vcdisk
from SPARC_RAR.linear_ML import compute_A_B
from utils_analysis.get_SPARC import get_SPARC_data

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec


use_dens = True
fit_const = False
pdisk = 0.5     # M/L ratio for disk


def get_table(galaxy:str):
    file = "/mnt/users/koe/SPARC_Lelli2016c.mrt.txt"
    SPARC_c = [ "Galaxy", "T", "D", "e_D", "f_D", "Inc",
            "e_Inc", "L", "e_L", "Reff", "SBeff", "Rdisk",
            "SBdisk", "MHI", "RHI", "Vflat", "e_Vflat", "Q", "Ref."]
    table = pd.read_fwf(file, skiprows=98, names=SPARC_c)
    i_table = _np.where(table["Galaxy"] == galaxy)[0][0]

    inc = table["Inc"][i_table]     # in degrees
    Rdisk = table["Rdisk"][i_table] # in kpc
    L = table["L"][i_table] * 1e9   # L[3.6] in Lsun

    return inc, Rdisk, L


def get_SBdisk(galaxy:str, inc:float):
    file = f"/mnt/users/koe/SPARC_RAR/BulgeDiskDec_LTG/{galaxy}.dens"
    columns = [ "Rad", "SBdisk", "SBbul" ]
    data = pd.read_csv(file, names=columns, skiprows=1, sep="\t")

    rad = jnp.array(data["Rad"])    # in kpc
    # SBdisk = jnp.array(data["SBdisk"]) * 1e6 * jnp.cos(inc*jnp.pi/180)  # in Lsun / kpc^2, corrected for inclination

    # Normalise SBdisk such that total disk luminosity matches L[3.6] in SPARC data (see email from Federico)
    SBdisk_raw = jnp.array(data["SBdisk"]) * 1e6
    Ltot, _ = compute_A_B(rad, SBdisk_raw)
    _, _, L = get_table(galaxy)
    SBdisk = SBdisk_raw * L / Ltot

    print(f"\nGalaxy {galaxy}: L_disk (from .dens) = {Ltot:.0f}, L_disk (SPARC) = {L:.0f},"
          f"\n cos(inc) = {jnp.cos(inc*jnp.pi/180):.3f}, scale factor = {L / Ltot:.3f}")

    return rad, SBdisk


def fit_multiplicative_constant(rad_model, SB_model, r_data, SB_data):
    """
    Fit a multiplicative constant to scale one galaxy's model surface brightness profile
    to best match its data surface brightness profile in a least-squares sense.

    Parameters
    ----------
    rad_model : 1D array
        Radii of the model surface brightness profile (kpc).
    SB_model : 1D array
        Model surface brightness profile (Lsun/kpc^2).
    r_data : 1D array
        Radii of the data surface brightness profile (kpc).
    SB_data : 1D array
        Data surface brightness profile (Lsun/kpc^2).
    
    Returns
    -------
    scale : float
        Best-fit multiplicative constant.
    """
    # Convert to numpy and ensure monotonic radius for interpolation
    rm = _np.asarray(rad_model)
    sm = _np.asarray(SB_model)
    order = _np.argsort(rm)
    rm = rm[order]; sm = sm[order]

    rd = _np.asarray(r_data)
    sd = _np.asarray(SB_data)

    # Use only data points inside the model radius range
    mask = (rd >= rm[0]) & (rd <= rm[-1])
    if mask.sum() == 0:
        raise ValueError("No overlap between model and data radii.")
    rd_mask = rd[mask]
    sd_mask = sd[mask]

    # Interpolate model onto data radii
    sm_interp = _np.interp(rd_mask, rm, sm)

    # Remove zero model points
    nonzero = sm_interp != 0
    if nonzero.sum() == 0:
        raise ValueError("Interpolated model is zero at overlapping radii.")
    sm_interp = sm_interp[nonzero]
    sd_mask = sd_mask[nonzero]

    # Least-squares scale for data ≈ scale * model (no offset)
    scale = float(_np.dot(sd_mask, sm_interp) / _np.dot(sm_interp, sm_interp))
    return scale


if __name__ == "__main__":
    SPARC_data, _, _ = get_SPARC_data()

    gals = [
        "DDO170", "NGC4100", "NGC0024", "NGC5585",
        "NGC0247", "UGC02259", "NGC3877", "UGC04325"
        ]   # 8 'upward hooked' galaxies from https://arxiv.org/pdf/2307.09507
    

    # Fit multiplicative constant for each galaxy and store results
    if fit_const:
        SB_scales = {}
        for gal in gals:
            try:
                inc_i, Rdisk_i, _ = get_table(gal)
                if use_dens:
                    rad_i, SBdisk_i = get_SBdisk(gal, inc_i)
                else:
                    rad_i = jnp.array(SPARC_data[gal]["r"])
                    SBdisk_i = jnp.array(SPARC_data[gal]["data"]["SBdisk"]) * 1e6

                r_i = jnp.array(SPARC_data[gal]["r"])
                data_SB = jnp.array(SPARC_data[gal]["data"]["SBdisk"]) * 1e6

                SB_scales[gal] = fit_multiplicative_constant(rad_i, SBdisk_i, r_i, data_SB)
            except Exception:
                SB_scales[gal] = None

        print("Best-fit SB multiplicative constants:", SB_scales)


    # --- Rotation curve grid (2x4) ---
    fig0, axes0 = plt.subplots(2, 4, figsize=(16, 8), sharex=False, sharey=False)
    axes0 = axes0.flatten()
    for i, gal_i in enumerate(gals):
        try:
            inc_i, Rdisk_i, _ = get_table(gal_i)
            if use_dens: rad_i, SBdisk_i = get_SBdisk(gal_i, inc_i)
            else: rad_i, SBdisk_i = jnp.array(SPARC_data[gal_i]["r"]), jnp.array(SPARC_data[gal_i]["data"]["SBdisk"]) * 1e6

            surface_density_i = pdisk * SBdisk_i
            v_disk_i = vcdisk(rad_i, surface_density_i, Rdisk_i, rhoz='exp')

            r_i = jnp.array(SPARC_data[gal_i]["r"])
            data_i = SPARC_data[gal_i]["data"]

            old_ax = axes0[i]
            old_spec = old_ax.get_subplotspec()
            old_ax.remove()

            gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=old_spec, height_ratios=[7, 3], hspace=0.05)
            ax_main = fig0.add_subplot(gs[0])
            ax_res = fig0.add_subplot(gs[1], sharex=ax_main)

            # --- Main panel: rotation velocity ---
            ax = ax_main
            ax.plot(r_i, data_i["Vdisk"].values * jnp.sqrt(pdisk), 'o', c='tab:blue', label="Data V_disk", alpha=0.5)
            ax.plot(rad_i, v_disk_i, '--', marker='o', c='tab:orange', label="Model V_disk", linewidth=1.5, alpha=0.5)

            if use_dens and fit_const:
                SBdisk_i_adjusted = SBdisk_i * SB_scales[gal_i]
                surface_density_i_adjusted = pdisk * SBdisk_i_adjusted
                v_disk_i_adjusted = vcdisk(rad_i, surface_density_i_adjusted, Rdisk_i, rhoz='exp')
                ax.plot(rad_i, v_disk_i_adjusted, '-', c='tab:green', label="Model V_disk (adjusted)", linewidth=1.5)

            ax.set_title(gal_i, fontsize=10)
            if i >= 4: ax.set_xlabel("Radius (kpc)", fontsize=9)
            ax.set_ylabel("V_disk (km/s)", fontsize=9)
            ax.grid(True)
            ax.legend(fontsize=8)

            # --- Residual panel: fractional residuals ---
            lines = {ln.get_label(): ln for ln in ax.get_lines()}
            data_line = lines.get("Data V_disk", None)
            if fit_const: adjusted_line = lines.get("Model V_disk (adjusted)", None)
            else: adjusted_line = lines.get("Model V_disk", None)

            if (data_line is not None):
                x_data = _np.asarray(data_line.get_xdata())
                y_data = _np.asarray(data_line.get_ydata())
                x_model = _np.asarray(adjusted_line.get_xdata())
                y_model = _np.asarray(adjusted_line.get_ydata())

                sm_interp = _np.interp(x_data, x_model, y_model)
                nonzero = y_data != 0
                max_allowed = min(_np.max(x_data), _np.max(x_model))
                in_range = (x_data >= _np.min(x_model)) & (x_data <= max_allowed)
                valid = in_range & nonzero

                frac = _np.full_like(x_data, _np.nan, dtype=float)
                if valid.any():
                    sm_interp_valid = _np.interp(x_data[valid], x_model, y_model)
                    frac[valid] = (sm_interp_valid - y_data[valid]) / y_data[valid]

                ax_res.plot(x_data, frac, c="tab:blue", marker="o", ms=3, label="(model - data) / data", alpha=0.7)
                ax_res.axhline(0.0, ls="--", color='k', alpha=0.5, lw=0.8)
                valid = _np.isfinite(frac)
                if valid.any():
                    vmax = _np.nanmax(_np.abs(frac[valid]))
                    ax_res.set_ylim(-1.1 * vmax, 1.1 * vmax if vmax != 0 else 1.0)
                else:
                    ax_res.set_ylim(-1, 1)
                ax_res.legend(fontsize=7, loc="upper right")
            else:
                ax_res.text(0.5, 0.5, "No adjusted model", ha='center', va='center', fontsize=8)
                ax_res.set_ylim(-1, 1)

            ax_res.grid(True, which="both", ls="--", lw=0.5)
            if i >= 4:
                ax_res.set_xlabel("Radius (kpc)", fontsize=9)
            if i == 0 or i == 4:
                ax_res.set_ylabel("Frac. resid.", fontsize=9)
            ax_main.xaxis.set_ticklabels([])

        except Exception as e:
            axes0[i].text(0.5, 0.5, f"Error: {gal_i}\n{e}", ha='center', va='center', wrap=True)
            axes0[i].set_title(gal_i, fontsize=10)
            axes0[i].grid(False)

    plt.tight_layout()
    if use_dens: fig0.savefig("/mnt/users/koe/SPARC_RAR/test_Vdisk.png", dpi=300)
    else: fig0.savefig("/mnt/users/koe/SPARC_RAR/test_Vdisk_rotmod.png", dpi=300)
    plt.close(fig0)


    # --- Acceleration curve grid (2x4) ---
    fig1, axes1 = plt.subplots(2, 4, figsize=(16, 8), sharex=False, sharey=False)
    axes1 = axes1.flatten()
    for i, gal_i in enumerate(gals):
        try:
            inc_i, Rdisk_i, _ = get_table(gal_i)
            if use_dens: rad_i, SBdisk_i = get_SBdisk(gal_i, inc_i)
            else: rad_i, SBdisk_i = jnp.array(SPARC_data[gal_i]["r"]), jnp.array(SPARC_data[gal_i]["data"]["SBdisk"]) * 1e6

            surface_density_i = pdisk * SBdisk_i
            v_disk_i = vcdisk(rad_i, surface_density_i, Rdisk_i, rhoz='exp')

            r_i = jnp.array(SPARC_data[gal_i]["r"])
            data_i = SPARC_data[gal_i]["data"]

            # Replace the single axes with two stacked axes (main + residual) using GridSpecFromSubplotSpec
            old_ax = axes1[i]
            old_spec = old_ax.get_subplotspec()
            old_ax.remove()

            gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=old_spec, height_ratios=[7, 3], hspace=0.05)
            ax_main = fig1.add_subplot(gs[0])
            ax_res = fig1.add_subplot(gs[1], sharex=ax_main)

            # --- Main panel: g_disk ---
            ax = ax_main
            ax.plot(r_i, vel2acc((data_i["Vdisk"].values) ** 2 * pdisk, r_i), 'o', c='tab:blue', label="Data g_disk", alpha=0.5)
            ax.plot(rad_i, vel2acc(v_disk_i**2, rad_i), '--', marker='o', c='tab:orange', label="Model g_disk", linewidth=1.5, alpha=0.5)

            if use_dens and fit_const:
                SBdisk_i_adjusted = SBdisk_i * SB_scales[gal_i]  # Apply best-fit multiplicative constant
                surface_density_i_adjusted = pdisk * SBdisk_i_adjusted
                v_disk_i_adjusted = vcdisk(rad_i, surface_density_i_adjusted, Rdisk_i, rhoz='exp')
                ax.plot(rad_i, vel2acc(v_disk_i_adjusted**2, rad_i), '-', c='tab:green', label="Model g_disk (adjusted)", linewidth=1.5)

            ax.set_title(gal_i, fontsize=10)
            if i >= 4: ax.set_xlabel("Radius (kpc)", fontsize=9)
            ax.set_ylabel("g_disk (m/s^2)", fontsize=9)
            ax.set_yscale("log")
            ax.grid(True)
            ax.legend(fontsize=8)

            # --- Residual panel: fractional residuals (model_adjusted - data) / data ---
            # extract plotted lines by label (these were created above)
            lines = {ln.get_label(): ln for ln in ax.get_lines()}
            data_line = lines.get("Data g_disk", None)
            if fit_const: adjusted_line = lines.get("Model g_disk (adjusted)", None)
            else: adjusted_line = lines.get("Model g_disk", None)

            if (data_line is not None) and (adjusted_line is not None):
                x_data = _np.asarray(data_line.get_xdata())
                y_data = _np.asarray(data_line.get_ydata())
                x_model = _np.asarray(adjusted_line.get_xdata())
                y_model = _np.asarray(adjusted_line.get_ydata())

                # interpolate adjusted model onto data radii
                sm_interp = _np.interp(x_data, x_model, y_model)

                # fractional residuals (model - data) / data, avoid divide-by-zero
                nonzero = y_data != 0
                
                # avoid extrapolation: only interpolate where data radii lie within model radius range
                max_allowed = min(_np.max(x_data), _np.max(x_model))
                in_range = (x_data >= _np.min(x_model)) & (x_data <= max_allowed)
                valid = in_range & nonzero

                frac = _np.full_like(x_data, _np.nan, dtype=float)
                if valid.any():
                    sm_interp_valid = _np.interp(x_data[valid], x_model, y_model)
                    frac[valid] = (sm_interp_valid - y_data[valid]) / y_data[valid]

                ax_res.plot(x_data, frac, c="tab:blue", marker="o", ms=3, label="(model - data) / data", alpha=0.7)
                ax_res.axhline(0.0, ls="--", color='k', alpha=0.5, lw=0.8)
                valid = _np.isfinite(frac)
                if valid.any():
                    vmax = _np.nanmax(_np.abs(frac[valid]))
                    ax_res.set_ylim(-1.1 * vmax, 1.1 * vmax if vmax != 0 else 1.0)
                else:
                    ax_res.set_ylim(-1, 1)
                ax_res.legend(fontsize=7, loc="upper right")
            else:
                ax_res.text(0.5, 0.5, "No adjusted model", ha='center', va='center', fontsize=8)
                ax_res.set_ylim(-1, 1)

            # formatting
            ax_res.grid(True, which="both", ls="--", lw=0.5)
            if i >= 4:
                ax_res.set_xlabel("Radius (kpc)", fontsize=9)
            if i == 0 or i == 4:
                ax_res.set_ylabel("Frac. resid.", fontsize=9)
            ax_main.xaxis.set_ticklabels([])  # hide x tick labels on the main axis

        except Exception as e:
            axes1[i].text(0.5, 0.5, f"Error: {gal_i}\n{e}", ha='center', va='center', wrap=True)
            axes1[i].set_title(gal_i, fontsize=10)
            axes1[i].grid(False)

    plt.tight_layout()
    if use_dens: fig1.savefig("/mnt/users/koe/SPARC_RAR/test_gdisk.png", dpi=300)
    else: fig1.savefig("/mnt/users/koe/SPARC_RAR/test_gdisk_rotmod.png", dpi=300)
    plt.close(fig1)


    # --- Surface brightness grid (2x4) ---
    fig2, axes2 = plt.subplots(2, 4, figsize=(16, 8), sharex=False, sharey=False)
    axes2 = axes2.flatten()
    for i, gal_i in enumerate(gals):
        try:
            inc_i, Rdisk_i, _ = get_table(gal_i)
            if use_dens: rad_i, SBdisk_i = get_SBdisk(gal_i, inc_i)
            else: rad_i, SBdisk_i = jnp.array(SPARC_data[gal_i]["r"]), jnp.array(SPARC_data[gal_i]["data"]["SBdisk"]) * 1e6

            r_i = jnp.array(SPARC_data[gal_i]["r"])
            data_i = SPARC_data[gal_i]["data"]

            # Replace the single axes with two stacked axes (main + residual) using GridSpecFromSubplotSpec

            old_ax = axes2[i]
            old_spec = old_ax.get_subplotspec()
            old_ax.remove()

            gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=old_spec, height_ratios=[7, 3], hspace=0.05)
            ax_main = fig2.add_subplot(gs[0])
            ax_res = fig2.add_subplot(gs[1], sharex=ax_main)

            # --- Main panel: surface brightness ---
            ax = ax_main
            ax.plot(r_i, jnp.array(data_i["SBdisk"]) * 1e6, 'o', c='tab:blue', label="Data SBdisk", alpha=0.5)
            ax.plot(rad_i, SBdisk_i, '--', marker='o', c='tab:orange', label="Model SBdisk", linewidth=1.5, alpha=0.5)

            if use_dens and fit_const:
                SBdisk_i_adjusted = SBdisk_i * SB_scales[gal_i]  # Apply best-fit multiplicative constant
                ax.plot(rad_i, SBdisk_i_adjusted, '-', c='tab:green', label="Model SBdisk (adjusted)", linewidth=1.5)
                
            ax.set_yscale("log")
            ax.set_title(gal_i, fontsize=10)
            if i == 0 or i == 4: ax.set_ylabel("SBdisk (Lsun/kpc^2)", fontsize=9)
            ax.grid(True, which="both", ls="--", lw=0.5)
            ax.legend(fontsize=8)

            # --- Residual panel: fractional residuals (model_adjusted - data) / data ---
            r_np = _np.asarray(r_i)
            data_SB = _np.asarray(data_i["SBdisk"]) * 1e6

            if use_dens:
                rad_np = _np.asarray(rad_i)
                if fit_const: sb_adj_np = _np.asarray(SBdisk_i * SB_scales[gal_i])
                else: sb_adj_np = _np.asarray(SBdisk_i)

                # Interpolate adjusted model onto data radii
                sm_interp = _np.interp(r_np, rad_np, sb_adj_np)
                # Avoid division by zero in data
                nonzero = data_SB != 0
                frac = _np.full_like(r_np, _np.nan, dtype=float)
                frac[nonzero] = (sm_interp[nonzero] - data_SB[nonzero]) / data_SB[nonzero]

                ax_res.plot(r_np, frac, c="tab:blue", marker="o", ms=3, label="(model - data) / data", alpha=0.7)
                ax_res.axhline(0.0, ls='--', color='k', alpha=0.5, lw=0.8)
                valid = _np.isfinite(frac)
                if valid.any():
                    vmax = _np.nanmax(_np.abs(frac[valid]))
                    ax_res.set_ylim(-1.1 * vmax, 1.1 * vmax if vmax != 0 else 1.0)
                ax_res.legend(fontsize=7, loc="upper right")
            else:
                ax_res.text(0.5, 0.5, "No adjusted model", ha='center', va='center', fontsize=8)
                ax_res.set_ylim(-1, 1)

            if i >= 4: ax_res.set_xlabel("Radius (kpc)", fontsize=9)
            if i == 0 or i == 4: ax_res.set_ylabel("Frac. resid.", fontsize=9)
            ax_res.grid(True, which="both", ls="--", lw=0.5)

            # Hide x tick labels on the main axis to avoid overlap
            ax_main.xaxis.set_ticklabels([])

        except Exception as e:
            axes2[i].text(0.5, 0.5, f"Error: {gal_i}\n{e}", ha='center', va='center', wrap=True)
            axes2[i].set_title(gal_i, fontsize=10)
            axes2[i].grid(False)

    plt.tight_layout()
    if use_dens: fig2.savefig("/mnt/users/koe/SPARC_RAR/test_SBdisk.png", dpi=300)
    else: fig2.savefig("/mnt/users/koe/SPARC_RAR/test_SBdisk_rotmod.png", dpi=300)
    plt.close(fig2)
