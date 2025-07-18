import os
import pandas as pd
import numpy as np
from collections import defaultdict
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors

# -------------------------------------------------------------------
# 1. Constants
# -------------------------------------------------------------------
conds =["LG_vs_TUG891", "LG_vs_PKA", "LG_vs_P2Y", "LG_vs_PKC", "LG_vs_TG",
        "AC_vs_LG", "CAM_vs_LG", "CK2_vs_LG", "EX_vs_LG", "HG_vs_LG", 
         "HG_vs_LG",
            "EX_vs_HG",
            "TUG891_vs_HG",
            "TG_vs_HG"]
kinases = [
    "PKACA", "CAMK1B", "CAMK2A", "P70S6K", "PDK1",
    "AMPKA2", "MARK3", "MARK4", "CK1A2", "CDK16",
    "JNK1", "JNK2", "ERK2", "ERK1", "ERK5"
]

go_to_compartment = {
        "GO:0005856": "Cytoskeleton",
        "GO:0005783": "Endoplasmic reticulum",
        "GO:0005768": "Endosome",
        "GO:0005794": "Golgi apparatus",
        "GO:0005764": "Lysosome",
        "GO:0005739": "Mitochondrion",
        "GO:0005634": "Nucleus",
        "GO:0005777": "Peroxisome",
        "GO:0005886": "Plasma membrane",
        "GO:0005929": "Cilium",
        "GO:0031982": "Vesicle",
        "GO:0099503":"Secretory vesicle",
        "GO:0005813": "Centrosome"

    }
compartments = sorted(set(go_to_compartment.values()))

# -------------------------------------------------------------------
# 2. Helpers
# -------------------------------------------------------------------
def build_vectors(df: pd.DataFrame, id_col: str):
    vec_dict = defaultdict(lambda: defaultdict(float))
    for _, row in df.iterrows():
        vec_dict[row[id_col]][row["Compartment"]] += row["Score"]

    vectors, ids = [], []
    for pid, score_dict in vec_dict.items():
        vec = [score_dict.get(c, 0.0) for c in compartments]
        tot = sum(vec)
        if tot:
            vectors.append(vec)
            ids.append(pid)
    return vectors, ids

# -------------------------------------------------------------------
# 3. Load and prepare reference database
# -------------------------------------------------------------------
df_OG_1 = pd.read_csv("human_compartment_integrated_full.tsv", sep="\t", header=None,
                      names=["ProteinID", "Gene", "GO_ID", "GO_Name", "Score"])
df_OG_1 = df_OG_1.rename(columns={"GO_Name": "Compartment"})
df_OG_1 = df_OG_1[df_OG_1["Compartment"].isin(go_to_compartment.values())]

df_gene = pd.read_excel("PANC_Input.xlsx")
genes = set(df_gene["Gene"].str.upper())
df_OG = df_OG_1[df_OG_1["Gene"].isin(genes)].copy()
df_OG["Score"] = pd.to_numeric(df_OG["Score"], errors="coerce")

all_vectors, all_ids = build_vectors(df_OG, "ProteinID")
score_by_protein_and_comp = df_OG.groupby(["ProteinID", "Compartment"])["Score"].mean()
print("Sample vector:", all_vectors[0])

# -------------------------------------------------------------------
# 4. Label compartments
# -------------------------------------------------------------------
labels, tiers = [], []
for pid, vec in zip(all_ids, all_vectors):
    vec = np.array(vec)
    total = vec.sum()
    if total == 0:
        label, tier = "Uncertain", "Low confidence"
    else:
        max_idx = np.argmax(vec)
        dominant_score = vec[max_idx]
        dominant_frac = dominant_score / total
        dominant_comp = compartments[max_idx]
        #the conditions defined for high and low confidence labeling
        if dominant_frac >= 0.3 and score_by_protein_and_comp.get((pid, dominant_comp), 0) >= 4:
            if dominant_comp != "Nucleus":
                label, tier = dominant_comp, "High confidence"
            #further filtering for nucleas due to its high abundance
            elif dominant_frac >= 0.5 and score_by_protein_and_comp.get((pid, dominant_comp), 0) >= 4.5:
                label, tier = dominant_comp, "High confidence"
            else:
                label, tier = "Uncertain", "Low confidence"
        else:
            label, tier = "Uncertain", "Low confidence"

    labels.append(label)
    tiers.append(tier)

labeled_vecs = [v for v, l in zip(all_vectors, labels) if l != "Uncertain"]
labeled_ids = [pid for pid, l in zip(all_ids, labels) if l != "Uncertain"]
labels_used = [l for l in labels if l != "Uncertain"]
tiers_used = [t for l, t in zip(labels, tiers) if l != "Uncertain"]

reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, spread=1.5, random_state=42)
hc_emb = reducer.fit_transform(labeled_vecs)
# Embed uncertain vectors using kNN and append to the full dataset
remaining_vecs = [v for v, l in zip(all_vectors, labels) if l == "Uncertain"]
remaining_ids = [pid for pid, l in zip(all_ids, labels) if l == "Uncertain"]
remaining_tiers = [t for l, t in zip(labels, tiers) if l == "Uncertain"]

if len(remaining_vecs) > 0:
    remaining_emb = reducer.transform(remaining_vecs)
    knn = NearestNeighbors(n_neighbors=3).fit(hc_emb)
    _, idx = knn.kneighbors(remaining_emb)
    inferred_labels = [labels_used[i[0]] for i in idx]

    df_unc = pd.DataFrame(remaining_emb, columns=["UMAP1", "UMAP2"])
    df_unc["ProteinID"] = remaining_ids
    df_unc["Label"] = inferred_labels
    df_unc["Tier"] = remaining_tiers

    # Combine high confidence and inferred uncertain embeddings
    df_all = pd.DataFrame(hc_emb, columns=["UMAP1", "UMAP2"])
    df_all["ProteinID"] = labeled_ids
    df_all["Label"] = labels_used
    df_all["Tier"] = tiers_used

    df_all = pd.concat([df_all, df_unc], ignore_index=True)
else:
    df_all = pd.DataFrame(hc_emb, columns=["UMAP1", "UMAP2"])
    df_all["ProteinID"] = labeled_ids
    df_all["Label"] = labels_used
    df_all["Tier"] = tiers_used
# -------------------------------------------------------------------
# 5. Main loop over conditions and kinases
# -------------------------------------------------------------------
# ==============================
# CONFIGURABLE STYLE PARAMETERS
# ==============================
circle_scale = 75 
circle_cap = 8*circle_scale   
circle_min = 10    # Minimum size to keep it visible
font_scale = 35   
font_family = "Arial"  # Set all fonts to Arial
background_s =25   # Background data dot size
thickness = 1.5
  
# Override colors for specific labels
custom_colors = {
    "Centrosome": "lightsteelblue",        # Same as 'cilia'
    "Plasma membrane": "lightgray"   # Light gray override
}

# ========== FOR ACTIVATED ==========
for cond in conds:
    cond_dir = cond
    #percentile ranking file generated from PhosphositePlus
    excel_path = os.path.join(cond_dir, "activated_ser_thr_percentile_ranks_thresh_10.xlsx")

    if not os.path.exists(excel_path):
        print(f"Excel file not found for {cond}. Skipping...")
        continue

    print(f"Processing: activated {cond}")
    for kinase in kinases:
        try:
            df_kinase = pd.read_excel(excel_path, sheet_name=kinase)
        except Exception as e:
            print(f"Could not load sheet {kinase} for {cond}: {e}")
            continue

        log2fc_col = f"{cond}_log2 fold change"
        pval_col = f"{cond}_p.val"
       
        if log2fc_col not in df_kinase.columns or pval_col not in df_kinase.columns:
            print(f"{log2fc_col} or {pval_col} missing in {kinase}-{cond}. Skipping...")
            continue
        #threshold for p_val and log2FC
        df_kinase_filt = df_kinase[(df_kinase[log2fc_col] > 1) & (df_kinase[pval_col] < 0.001)].copy()
        if df_kinase_filt.empty:
            print(f"No substrates passed threshold for {kinase} in {cond}.")
            continue

        df_kinase_filt["Gene"] = df_kinase_filt["Gene"].astype(str).str.strip().str.upper()
        prots = df_kinase_filt["Gene"].dropna().tolist()
        df_sub_OG = df_OG_1[df_OG_1["Gene"].isin(prots)].copy()
        if df_sub_OG.empty:
            print(f"No matching genes in compartment DB for {kinase} in {cond}.")
            continue

        sub_vectors, sub_ids = build_vectors(df_sub_OG, "ProteinID")
        if not sub_vectors:
            continue

        sub_emb = reducer.transform(sub_vectors)
        df_sub_emb = pd.DataFrame(sub_emb, columns=["UMAP1", "UMAP2"])
        df_sub_emb["Gene"] = df_sub_OG["Gene"].unique()[:len(df_sub_emb)]
        df_sub_emb["Gene"] = df_sub_emb["Gene"].astype(str).str.strip().str.upper()

        fc_map = df_kinase_filt.set_index("Gene")[log2fc_col].to_dict()
        df_sub_emb["log2FC"] = df_sub_emb["Gene"].map(fc_map).fillna(0)
        df_sub_emb["log2FC"] = pd.to_numeric(df_sub_emb["log2FC"], errors="coerce")
        df_sub_emb["size"] = np.clip(df_sub_emb["log2FC"].abs() * circle_scale, circle_min, circle_cap)

        knn = NearestNeighbors(n_neighbors=2).fit(hc_emb)
        _, idx = knn.kneighbors(sub_emb)
        pred_labels = [labels_used[i[0]] for i in idx]
        df_sub_emb["Predicted"] = pred_labels

        unique_labels = sorted(df_all["Label"].unique())
        palette = {}
        tab20_palette = sns.color_palette("tab20", len(unique_labels))
        for lab, col in zip(unique_labels, tab20_palette):
            if lab == "Centrosome":
                palette[lab] = custom_colors["Centrosome"]
            elif lab == "Plasma membrane":
                palette[lab] = custom_colors["Plasma membrane"]
            else:
                palette[lab] = col

        os.makedirs(f"output/{cond}", exist_ok=True)  # ✳️ Ensure output directory exists

        plt.figure(figsize=(8, 8))

        sns.scatterplot(
            data=df_all[df_all["Tier"] == "Low confidence"],
            x="UMAP1", y="UMAP2",
            hue="Label", palette=palette,
            s=background_s, alpha=0.3, legend=False, linewidth=0  # ✳️ Background point size
        )
        sns.scatterplot(
            data=df_all[df_all["Tier"] == "High confidence"],
            x="UMAP1", y="UMAP2",
            hue="Label", palette=palette,
            s=background_s, alpha=0.3, legend=False, linewidth=0  # ✳️ Background point size
        )

        for label in df_sub_emb["Predicted"].unique():
            temp = df_sub_emb[df_sub_emb["Predicted"] == label]
            pos = temp[temp["log2FC"] > 0]
            plt.scatter(
                pos["UMAP1"], pos["UMAP2"],
                s=pos["size"],
                color=palette.get(label, "black"),
                edgecolor="black",
                linewidth=0.5,
                alpha=0.9,
                label=f"{label}"
            )

        for size_val in [1, 2, 4]:
            plt.scatter([], [], s=size_val * circle_scale, edgecolor='black', facecolor='gray', label=f"log2FC ≈ {size_val}")

        plt.title(f"{cond} - {kinase}", fontsize=font_scale/7, family=font_family)
        #plt.legend(title="Legend", bbox_to_anchor=(1.05, 1), loc="upper left", prop={'family': font_family, 'size': font_scale * 0.6})
       
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_linewidth(thickness)
        plt.xlabel("UMAP1", fontsize=font_scale/3, family=font_family)
        plt.ylabel("UMAP2", fontsize=font_scale/3, family=font_family)
        plt.tick_params(axis='both', which='major', labelsize=font_scale)
        plt.tight_layout()
       
        plt.savefig(f"output/{cond}/{kinase}_activated.png", dpi=300)
        plt.close()


# ========== FOR INHIBITED ==========
for cond in conds:
    cond_dir = cond
    excel_path = os.path.join(cond_dir, "inhibited_ser_thr_percentile_ranks_thresh_10.xlsx")

    if not os.path.exists(excel_path):
        print(f"Excel file not found for {cond}. Skipping...")
        continue

    print(f"Processing: inhibition {cond}")
    for kinase in kinases:
        try:
            df_kinase = pd.read_excel(excel_path, sheet_name=kinase)
        except Exception as e:
            print(f"Could not load sheet {kinase} for {cond}: {e}")
            continue

        log2fc_col = f"{cond}_log2 fold change"
        pval_col = f"{cond}_p.val"

        if log2fc_col not in df_kinase.columns or pval_col not in df_kinase.columns:
            print(f"{log2fc_col} or {pval_col} missing in {kinase}-{cond}. Skipping...")
            continue

        df_kinase_filt = df_kinase[(df_kinase[log2fc_col] < -1) & (df_kinase[pval_col] < 0.001)].copy()
        if df_kinase_filt.empty:
            print(f"No substrates passed threshold for {kinase} in {cond}.")
            continue

        df_kinase_filt["Gene"] = df_kinase_filt["Gene"].astype(str).str.strip().str.upper()
        prots = df_kinase_filt["Gene"].dropna().tolist()
        df_sub_OG = df_OG_1[df_OG_1["Gene"].isin(prots)].copy()
        if df_sub_OG.empty:
            print(f"No matching genes in compartment DB for {kinase} in {cond}.")
            continue

        sub_vectors, sub_ids = build_vectors(df_sub_OG, "ProteinID")
        if not sub_vectors:
            continue

        sub_emb = reducer.transform(sub_vectors)
        df_sub_emb = pd.DataFrame(sub_emb, columns=["UMAP1", "UMAP2"])
        df_sub_emb["Gene"] = df_sub_OG["Gene"].unique()[:len(df_sub_emb)]
        df_sub_emb["Gene"] = df_sub_emb["Gene"].astype(str).str.strip().str.upper()

        fc_map = df_kinase_filt.set_index("Gene")[log2fc_col].to_dict()
        df_sub_emb["log2FC"] = df_sub_emb["Gene"].map(fc_map).fillna(0)
        df_sub_emb["log2FC"] = pd.to_numeric(df_sub_emb["log2FC"], errors="coerce")
        df_sub_emb["size"] = np.clip(df_sub_emb["log2FC"].abs() * circle_scale, circle_min, circle_cap)

        knn = NearestNeighbors(n_neighbors=2).fit(hc_emb)
        _, idx = knn.kneighbors(sub_emb)
        pred_labels = [labels_used[i[0]] for i in idx]
        df_sub_emb["Predicted"] = pred_labels

        unique_labels = sorted(df_all["Label"].unique())
        palette = {}
        tab20_palette = sns.color_palette("tab20", len(unique_labels))
        for lab, col in zip(unique_labels, tab20_palette):
            if lab == "Centrosome":
                palette[lab] = custom_colors["Centrosome"]
            elif lab == "Plasma membrane":
                palette[lab] = custom_colors["Plasma membrane"]
            else:
                palette[lab] = col

        os.makedirs(f"output/{cond}", exist_ok=True)
        os.makedirs(f"output_paper_iteration/{cond}", exist_ok=True)

        plt.figure(figsize=(8, 8))

        sns.scatterplot(
            data=df_all[df_all["Tier"] == "Low confidence"],
            x="UMAP1", y="UMAP2",
            hue="Label", palette=palette,
            s=background_s, alpha=0.3, legend=False, linewidth=0
        )
        sns.scatterplot(
            data=df_all[df_all["Tier"] == "High confidence"],
            x="UMAP1", y="UMAP2",
            hue="Label", palette=palette,
            s=background_s, alpha=0.3, legend=False, linewidth=0
        )

        for label in df_sub_emb["Predicted"].unique():
            temp = df_sub_emb[df_sub_emb["Predicted"] == label]
            pos = temp[temp["log2FC"] < 0]
            plt.scatter(
                pos["UMAP1"], pos["UMAP2"],
                s=pos["size"],
                color=palette.get(label, "black"),
                edgecolor="black",
                linewidth=0.5,
                alpha=0.9,
                label=f"{label}"
            )

        for size_val in [1, 2, 4]:
            plt.scatter([], [], s=size_val * circle_scale, edgecolor='black', facecolor='gray', label=f"log2FC ≈ -{size_val}")

        plt.title(f"{cond} - {kinase}", fontsize=font_scale/7, family=font_family)
        #plt.legend(title="Legend", bbox_to_anchor=(1.05, 1), loc="upper left", prop={'family': font_family, 'size': font_scale * 0.6})
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_linewidth(thickness)
        plt.xlabel("UMAP1", fontsize=font_scale/3, family=font_family)
        plt.ylabel("UMAP2", fontsize=font_scale/3, family=font_family)
        plt.tick_params(axis='both', which='major', labelsize=font_scale)
        plt.tight_layout()
        
        
        plt.savefig(f"output/{cond}/{kinase}_inhibited.png", dpi=300)
        plt.close()
