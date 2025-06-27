#!/usr/bin/env python3
"""
Plot all numeric branches of a hard-coded ROOT file tree as red outline histograms
with an overlayed legend showing the number of entries.

Inputs/Outputs are hard-coded:
  INPUT_FILE = "flattuple-znunu.root"
  TREE_NAME  = "znunu_delphes"
  OUTPUT_DIR = "plots"

Integer-valued branches (nJets, nBjets, nCjets) use bin edges at integer boundaries.
All histograms are drawn with histtype="step", line color red, and a legend in the upper right.
"""

import os
import numpy as np
import uproot
import matplotlib.pyplot as plt

# Hard-coded parameters
INPUT_FILE   = "flattuple-znunu.root"
TREE_NAME    = "znunu_delphes"
OUTPUT_DIR   = "plots"
DEFAULT_BINS = 100

# Which branches are integer-valued
INT_BRANCHES = {"nJets", "nBjets", "nCjets"}

def plot_branches():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with uproot.open(INPUT_FILE) as f:
        tree = f[TREE_NAME]
        for branch in tree.keys():
            data = tree[branch].array(library="np")

            # skip empty or non-numeric
            if data.size == 0 or not np.issubdtype(data.dtype, np.number):
                continue

            # choose bins
            if branch in INT_BRANCHES:
                lo = int(data.min())
                hi = int(data.max())
                bins = np.arange(lo, hi + 2, 1)
            else:
                bins = DEFAULT_BINS

            plt.figure(figsize=(8, 6), dpi=300)
            plt.hist(
                data,
                bins=bins,
                histtype="step",
                linewidth=1.5,
                color="red"
            )
            plt.title(f"Histogram of {branch}")
            plt.xlabel(branch)
            plt.ylabel("Entries")
            plt.grid(True)
            plt.legend([f"Entries = {data.size}"], loc="upper right")
            plt.tight_layout()

            outpath = os.path.join(OUTPUT_DIR, f"{branch}.png")
            plt.savefig(outpath)
            plt.close()

if __name__ == "__main__":
    plot_branches()
    print(f"Saved histograms to directory '{OUTPUT_DIR}/'.")
