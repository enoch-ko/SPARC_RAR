import pandas as pd
import numpy as np
import sys
import os

import jax.numpy as jnp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from SPARC_RAR.vcdisk_differentiable import vcdisk
from utils_analysis.params import pdisk
from utils_analysis.get_SPARC import get_SPARC_data

from matplotlib import pyplot as plt


def get_table(galaxy:str):
    file = "/mnt/users/koe/SPARC_Lelli2016c.mrt.txt"
    SPARC_c = [ "Galaxy", "T", "D", "e_D", "f_D", "Inc",
            "e_Inc", "L", "e_L", "Reff", "SBeff", "Rdisk",
            "SBdisk", "MHI", "RHI", "Vflat", "e_Vflat", "Q", "Ref."]
    table = pd.read_fwf(file, skiprows=98, names=SPARC_c)
    i_table = np.where(table["Galaxy"] == galaxy)[0][0]

    inc = table["Inc"][i_table]     # in degrees
    Rdisk = table["Rdisk"][i_table] # in kpc

    return inc, Rdisk


def get_SBdisk(galaxy:str, inc:float):
    file = f"/mnt/users/koe/SPARC_RAR/BulgeDiskDec_LTG/{galaxy}.dens"
    columns = [ "Rad", "SBdisk", "SBbul" ]
    data = pd.read_csv(file, names=columns, skiprows=1, sep="\t")

    rad = jnp.array(data["Rad"])    # in kpc
    SBdisk = jnp.array(data["SBdisk"]) * 1e6 * jnp.cos(inc*jnp.pi/180)    # in Lsun / kpc^2, corrected for inclination

    return rad, SBdisk


if __name__ == "__main__":
    SPARC_data, _, _ = get_SPARC_data()

    gals = [
        "DDO170", "NGC4100", "NGC0024", "NGC5585",
        "NGC0247", "UGC02259", "NGC3877", "UGC04325"
        ]   # 8 'upward hooked' galaxies from https://arxiv.org/pdf/2307.09507

    # --- Rotation curve grid (2x4) ---
    fig1, axes1 = plt.subplots(2, 4, figsize=(16, 8), sharex=False, sharey=False)
    axes1 = axes1.flatten()
    for i, gal_i in enumerate(gals):
        try:
            inc_i, Rdisk_i = get_table(gal_i)
            rad_i, SBdisk_i = get_SBdisk(gal_i, inc_i)
            surface_density_i = pdisk * SBdisk_i
            v_disk_i = vcdisk(rad_i, surface_density_i, Rdisk_i, rhoz='exp')

            r_i = jnp.array(SPARC_data[gal_i]["r"])
            data_i = SPARC_data[gal_i]["data"]

            ax = axes1[i]
            ax.plot(r_i, data_i["Vdisk"].values * jnp.sqrt(pdisk), 'o', label="Data Vdisk", markersize=4)
            ax.plot(rad_i, v_disk_i, '-', label="Model Vdisk", linewidth=1.5)
            ax.set_title(gal_i, fontsize=10)
            if i >= 4: ax.set_xlabel("Radius (kpc)", fontsize=9)
            if i == 0 or i == 4: ax.set_ylabel("Vdisk (km/s)", fontsize=9)
            ax.grid(True)
            ax.legend(fontsize=8)
        except Exception as e:
            axes1[i].text(0.5, 0.5, f"Error: {gal_i}\n{e}", ha='center', va='center', wrap=True)
            axes1[i].set_title(gal_i, fontsize=10)
            axes1[i].grid(False)

    plt.tight_layout()
    fig1.savefig("/mnt/users/koe/SPARC_RAR/test_vcdisk.png", dpi=300)
    plt.close(fig1)

    # --- Surface brightness grid (2x4) ---
    fig2, axes2 = plt.subplots(2, 4, figsize=(16, 8), sharex=False, sharey=False)
    axes2 = axes2.flatten()
    for i, gal_i in enumerate(gals):
        try:
            inc_i, Rdisk_i = get_table(gal_i)
            rad_i, SBdisk_i = get_SBdisk(gal_i, inc_i)

            r_i = jnp.array(SPARC_data[gal_i]["r"])
            data_i = SPARC_data[gal_i]["data"]

            ax = axes2[i]
            ax.plot(r_i, jnp.array(data_i["SBdisk"]) * 1e6, 'o', label="Data SBdisk", markersize=4)
            ax.plot(rad_i, SBdisk_i, '-', label="Model SBdisk", linewidth=1.5)
            ax.set_yscale("log")
            ax.set_title(gal_i, fontsize=10)
            if i >= 4: ax.set_xlabel("Radius (kpc)", fontsize=9)
            if i == 0 or i == 4: ax.set_ylabel("SBdisk (Lsun/kpc^2)", fontsize=9)
            ax.grid(True, which="both", ls="--", lw=0.5)
            ax.legend(fontsize=8)
        except Exception as e:
            axes2[i].text(0.5, 0.5, f"Error: {gal_i}\n{e}", ha='center', va='center', wrap=True)
            axes2[i].set_title(gal_i, fontsize=10)
            axes2[i].grid(False)

    plt.tight_layout()
    fig2.savefig("/mnt/users/koe/SPARC_RAR/test_SBdisk.png", dpi=300)
    plt.close(fig2)
