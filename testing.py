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


def plot_RCs(r, Vdisk_data, rad, vcdisk):
    plt.plot(r, Vdisk_data, 'o', label="Data Vdisk", markersize=6)
    plt.plot(rad, vcdisk, '-', label="Model Vdisk", linewidth=2)
    plt.xlabel("Radius (kpc)", fontsize=14)
    plt.ylabel("Vdisk (km/s)", fontsize=14)
    plt.title("Disk Rotation Curve", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid()
    plt.savefig("/mnt/users/koe/SPARC_RAR/test_vcdisk.png", dpi=300)
    plt.close()


def plot_SBdisk(r, SBdisk_data, rad, SBdisk_model):
    plt.plot(r, SBdisk_data, 'o', label="Data SBdisk", markersize=6)
    plt.plot(rad, SBdisk_model, '-', label="Model SBdisk", linewidth=2)
    plt.xlabel("Radius (kpc)", fontsize=14)
    plt.ylabel("SBdisk (Lsun/kpc^2)", fontsize=14)
    plt.title("Disk Surface Brightness Profile", fontsize=16)
    plt.legend(fontsize=12)
    plt.yscale("log")
    plt.grid()
    plt.savefig("/mnt/users/koe/SPARC_RAR/test_SBdisk.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    SPARC_data, _, _ = get_SPARC_data()
    gal = "DDO170"

    r = jnp.array(SPARC_data[gal]["r"])
    data = SPARC_data[gal]["data"]
    inc, Rdisk = get_table(gal)
    rad, SBdisk = get_SBdisk(gal, inc)
    # SBdisk = jnp.array(data["SBdisk"]) * 1e6    # in Lsun / kpc^2

    surface_density = pdisk * SBdisk    # in Msun / kpc^2

    v_disk = vcdisk(rad, surface_density, Rdisk, rhoz='exp')   # in km/s

    plot_RCs(r, data["Vdisk"].values * jnp.sqrt(pdisk), rad, v_disk)
    plot_SBdisk(r, jnp.array(data["SBdisk"]) * 1e6, rad, SBdisk)
