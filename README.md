# Kinase Substrate Localization UMAP

This script visualizes predicted subcellular localizations of kinase substrates using UMAP embeddings based on GO compartment annotations.

---

## 🔧 Overview

* Builds protein-compartment vectors using `human_compartment_integrated_full.tsv` from https://compartments.jensenlab.org/Search.
* Uses UMAP to embed proteins in 2D.
* Labels proteins by compartment with confidence tiers.
* Overlays kinase substrates from activation/inhibition datasets.
* Outputs UMAP plots for each condition and kinase.

---

## 📁 Required Files

* `human_compartment_integrated_full.tsv`: GO-based compartment scores
* List of input genes for plotting
* Condition folders (e.g., `HG_vs_LG/`) with:

  * `activated_ser_thr_percentile_ranks_thresh_10.xlsx`
  * `inhibited_ser_thr_percentile_ranks_thresh_10.xlsx`

---

## ▶️ Run

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn umap-learn scikit-learn openpyxl
```
---

## 📤 Output

UMAP plots saved to:

* `output/[condition]/[kinase]_activated.png`
* `output/[condition]/[kinase]_inhibited.png`

