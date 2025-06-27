#!/usr/bin/env python3
"""
Process a Delphes ROOT file (ATLAS PILEUP card) and write out a flat ntuple
(.root) containing per-event variables for the two leading jets, MET, and jet counts
including flavor-based counts for b- and c-jets.

Inputs/Outputs are hard-coded:
  INPUT_FILE  = "/home/sgoswami/mg5-tutorial/madgraph_tutorial/MG5_aMC_v3_6_3/Delphes/delphes_znunu.root"
  OUTPUT_FILE = "flattuple-znunu.root"  (will be overwritten)

Outputs a TTree 'znunu_delphes' with branches:
  - Jet_pt1           : float32, pT of the leading jet
  - Jet_pt2           : float32, pT of the subleading jet
  - jet1_eta          : float32, eta of the leading jet
  - jet2_eta          : float32, eta of the subleading jet
  - jet1_phi          : float32, phi of the leading jet
  - jet2_phi          : float32, phi of the subleading jet
  - met_pt            : float32, missing transverse energy
  - jet1_met_dphi     : float32, Delta phi between leading jet and MET
  - met_significance  : float32, MET / \sqrt(HT)
  - nJets             : int32, total number of jets in event
  - nBjets            : int32, number of b-flavored jets (by jet flavor ID)
  - nCjets            : int32, number of c-flavored jets (by jet flavor ID)
"""

import math
import numpy as np
import uproot
import awkward as ak

# Hard-coded filenames
INPUT_FILE  = "/home/sgoswami/mg5-tutorial/madgraph_tutorial/MG5_aMC_v3_6_3/Delphes/delphes_znunu.root"
OUTPUT_FILE = "flattuple-znunu.root"


def compute_leading_jets(jets_pt, jets_eta, jets_phi):
    """
    Select events with ≥2 jets, sort each event's jets by descending pT,
    and return NumPy arrays for the two leading jets and a mask:
      pt1, pt2  : leading and subleading jet pT
      eta1, eta2: leading and subleading jet eta
      phi1, phi2: leading and subleading jet phi
      mask      : awkward Boolean mask of events with ≥2 jets
    """
    mask = ak.num(jets_pt, axis=1) >= 2
    pt_m  = jets_pt[mask]
    eta_m = jets_eta[mask]
    phi_m = jets_phi[mask]

    idx   = ak.argsort(pt_m, axis=1, ascending=False)
    pt_s  = pt_m[idx]
    eta_s = eta_m[idx]
    phi_s = phi_m[idx]

    pt1   = pt_s[:, 0].to_numpy()
    pt2   = pt_s[:, 1].to_numpy()
    eta1  = eta_s[:, 0].to_numpy()
    eta2  = eta_s[:, 1].to_numpy()
    phi1  = phi_s[:, 0].to_numpy()
    phi2  = phi_s[:, 1].to_numpy()

    return pt1, pt2, eta1, eta2, phi1, phi2, mask


def compute_dphi(phi1, phi2):
    """
    Compute delta phi between two 1-D angle arrays, wrapped into [0, pi].
    """
    d = (phi1 - phi2 + math.pi) % (2 * math.pi) - math.pi
    return np.abs(d)


def main():
    # Open the Delphes file and get the Delphes TTree
    f_in = uproot.open(INPUT_FILE)
    tree = f_in["Delphes"]

    # Load jagged arrays for jets, MET, and HT
    jets_pt     = tree["Jet.PT"].array(library="ak")
    jets_eta    = tree["Jet.Eta"].array(library="ak")
    jets_phi    = tree["Jet.Phi"].array(library="ak")
    met_ak      = tree["MissingET.MET"].array(library="ak")
    met_phi_ak  = tree["MissingET.Phi"].array(library="ak")
    ht_ak       = tree["ScalarHT.HT"].array(library="ak")

    # Attempt to read the flavor-ID branch (requires JetFlavorAssociation in your Delphes card)
    try:
        flavor_ak = tree["Jet.Flavor"].array(library="ak")
    except uproot.KeyInFileError:
        try:
            flavor_ak = tree["Jet.FlavorAlgo"].array(library="ak")
        except uproot.KeyInFileError:
            print("Warning: no Jet.Flavor or Jet.FlavorAlgo branch found; setting all to 0")
            flavor_ak = ak.zeros_like(jets_pt, dtype=int)

    # Compute total counts for all events
    nJets_all   = ak.num(jets_pt, axis=1)
    nB_all      = ak.sum(flavor_ak == 5, axis=1)
    nC_all      = ak.sum(flavor_ak == 4, axis=1)

    # Compute leading/subleading jets and mask for events with ≥2 jets
    pt1, pt2, eta1, eta2, phi1, phi2, mask = compute_leading_jets(
        jets_pt, jets_eta, jets_phi)

    # Apply mask to MET, HT, and counts; convert to NumPy and squeeze out any extra dims
    met      = met_ak[mask].to_numpy().squeeze()
    met_phi  = met_phi_ak[mask].to_numpy().squeeze()
    ht       = ht_ak[mask].to_numpy().squeeze()
    nJets    = nJets_all[mask].to_numpy().astype(np.int32)
    nBjets   = nB_all[mask].to_numpy().astype(np.int32)
    nCjets   = nC_all[mask].to_numpy().astype(np.int32)

    # Compute derived quantities
    met_pt            = met
    jet1_met_dphi     = compute_dphi(phi1, met_phi)
    met_significance  = np.where(ht > 0, met / np.sqrt(ht), 0.0)

    # Prepare output dictionary; ensure float32 for floats
    out = {
        "Jet_pt1":          pt1.astype(np.float32),
        "Jet_pt2":          pt2.astype(np.float32),
        "jet1_eta":         eta1.astype(np.float32),
        "jet2_eta":         eta2.astype(np.float32),
        "jet1_phi":         phi1.astype(np.float32),
        "jet2_phi":         phi2.astype(np.float32),
        "met_pt":           met_pt.astype(np.float32),
        "jet1_met_dphi":    jet1_met_dphi.astype(np.float32),
        "met_significance": met_significance.astype(np.float32),
        "nJets":            nJets,
        "nBjets":           nBjets,
        "nCjets":           nCjets,
    }

    # Debug: print shapes to confirm all branches are 1-D and consistent
    print("Branch shapes before writing:")
    for name, arr in out.items():
        print(f"  {name:15s} -> {arr.shape}")

    # Write out the flat ntuple, overwriting if the file exists
    with uproot.recreate(OUTPUT_FILE) as f_out:
        f_out.mktree("znunu_delphes", {
            "Jet_pt1":          "float32",
            "Jet_pt2":          "float32",
            "jet1_eta":         "float32",
            "jet2_eta":         "float32",
            "jet1_phi":         "float32",
            "jet2_phi":         "float32",
            "met_pt":           "float32",
            "jet1_met_dphi":    "float32",
            "met_significance": "float32",
            "nJets":            "int32",
            "nBjets":           "int32",
            "nCjets":           "int32",
        })
        f_out["znunu_delphes"].extend(out)

    print(f"Wrote {pt1.shape[0]} events to '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    main()