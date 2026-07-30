import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
mpl.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

res = pd.read_parquet("data_processed/rf_scores.parquet")
fam = pd.read_parquet("data_processed/family_divergence.parquet")
print(len(res), "score rows,", len(fam), "families")

ORDER  = ["esm2_8m","esm2_35m","esm2_150m","esm2_650m","esmc_300m","esmc_600m","hamming"]
LABEL  = {"esm2_8m":"ESM-2 8M","esm2_35m":"ESM-2 35M","esm2_150m":"ESM-2 150M",
          "esm2_650m":"ESM-2 650M","esmc_300m":"ESM-C 300M","esmc_600m":"ESM-C 600M",
          "hamming":"Hamming"}
PARAMS = {"esm2_8m":8,"esm2_35m":35,"esm2_150m":150,"esm2_650m":650,
          "esmc_300m":300,"esmc_600m":600}
COLOR  = {"esm2_8m":"#c6dbef","esm2_35m":"#9ecae1","esm2_150m":"#6baed6",
          "esm2_650m":"#2171b5","esmc_300m":"#fdae6b","esmc_600m":"#e6550d",
          "hamming":"#333333"}

fig, ax = plt.subplots(figsize=(8, 4.2))
x = np.arange(len(ORDER)); w = 0.38

for ref, off, alpha, lab in [("species", -w/2, 1.0, "Species tree (independent)"),
                             ("ml",      +w/2, 0.5, "ML tree (shares alignment)")]:
    g  = res[res.ref == ref].groupby("method")["norm_rf"]
    m  = [g.mean()[t] for t in ORDER]
    se = [g.sem()[t]  for t in ORDER]
    ax.bar(x + off, m, w, yerr=se, capsize=3, label=lab,
           color=[COLOR[t] for t in ORDER], alpha=alpha, edgecolor="white")


hb = res.groupby(["ref","method"])["norm_rf"].mean()
ax.axhline(hb["species","hamming"], ls="--", c=COLOR["hamming"], lw=1, alpha=0.7,
           label="Hamming (species)")
#ax.axhline(hb["ml","hamming"], ls=":", c=COLOR["hamming"], lw=1, alpha=0.7,
#           label="Hamming (ML)")
ax.set_xticks(x); ax.set_xticklabels([LABEL[t] for t in ORDER], rotation=30, ha="right")
ax.set_ylabel("Normalized RF distance\n(lower is closer to true tree)")
ax.set_title("Protein-language-model embeddings do not recover phylogeny")
ax.legend(frameon=False, fontsize=9, loc="lower left")



fig.savefig("figures/fig1_overall.png"); fig.savefig("figures/fig1_overall.pdf")
plt.show()

dcorr = pd.read_parquet("data_processed/dist_corr.parquet")

mc = dcorr.groupby("method")["corr"].mean()                       # distance correlation
mr = res[res.ref == "ml"].groupby("method")["norm_rf"].mean()     # RF vs ML tree

fig, ax = plt.subplots(figsize=(8, 6))
for m in ORDER:
    ax.scatter(mc[m], mr[m], color=COLOR[m], s=100, zorder=5)

# label offsets in points (dx, dy); tweak if any still overlap
off = {
    "esmc_600m": (40, -5),
    "esm2_650m": (30, -20),
    "esm2_8m":   (-15, -20),
    "esm2_35m":  (50, 5),
    "esmc_300m": (10, 10),
    "esm2_150m": (55, -15),
    "hamming":   (0, 10),
}
for m in ORDER:
    ax.annotate(LABEL[m], (mc[m], mr[m]),
                textcoords="offset points", xytext=off[m], fontsize=8,
                color=COLOR[m], ha="center",
                arrowprops=dict(arrowstyle="-", color=COLOR[m],
                                lw=0.6, shrinkA=0, shrinkB=3))

ax.set_ylim(mr.min() - 0.01, mr.max() + 0.05)

ax.set_xlabel("Distance correlation with phylogeny  (higher is better)")
ax.set_ylabel("Normalized RF  (lower is better)")
ax.set_title("Hamming dominates on both criteria;\nembeddings are moderately correlated and reconstruct poorly")
fig.savefig("figures/fig2_two_criteria.png"); fig.savefig("figures/fig2_two_criteria.pdf")
plt.show()

def clade_heatmap(ref):
    piv = (res[res.ref == ref]
           .groupby(["clade", "method"])["norm_rf"].mean()
           .unstack("method").reindex(index=clades, columns=ORDER))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(piv.values, cmap="YlOrRd", vmin=0.4, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(ORDER))); ax.set_xticklabels([LABEL[t] for t in ORDER], rotation=30, ha="right")
    ax.set_yticks(range(len(clades)))
    ax.set_yticklabels([f"{c}\n(id {clade_div[c]:.2f})" for c in clades], fontsize=8)
    for i in range(len(clades)):
        for j in range(len(ORDER)):
            ax.text(j, i, f"{piv.values[i,j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, label="Normalized RF (lower is better)")
    refname = "species tree" if ref == "species" else "ML gene tree"
    ax.set_title(f"Hamming wins in every clade; embeddings near-random\n(reference: {refname})")
    ax.grid(False)
    fig.savefig(f"figures/fig3_clade_heatmap_{ref}.png")
    fig.savefig(f"figures/fig3_clade_heatmap_{ref}.pdf")
    plt.show()

clade_div = fam.groupby("clade")["mean_pid"].median().sort_values()
clades = clade_div.index.tolist()

clade_heatmap("species")

clade_heatmap("ml")

fig, ax = plt.subplots(figsize=(7.5, 4.8))
g = res[res.ref == "species"].groupby("method")["norm_rf"]

esm2 = ["esm2_8m","esm2_35m","esm2_150m","esm2_650m"]
esmc = ["esmc_300m","esmc_600m"]

ax.plot([PARAMS[m] for m in esm2], [g.mean()[m] for m in esm2],
        "-o", color="#2171b5", markersize=8, label="ESM2")
ax.plot([PARAMS[m] for m in esmc], [g.mean()[m] for m in esmc],
        "-o", color="#e6550d", markersize=8, label="ESMC")

hb = g.mean()["hamming"]
ax.axhline(hb, ls="--", c="#333333", lw=1)
ax.text(600, hb + 0.004, "Hamming baseline", ha="right", va="bottom",
        fontsize=8, color="#333333")

for m in esm2 + esmc:
    dy = -14 if m == "esm2_650m" else 9
    va = "top" if m == "esm2_650m" else "bottom"
    ax.annotate(LABEL[m], (PARAMS[m], g.mean()[m]),
                textcoords="offset points", xytext=(0, dy),
                ha="center", va=va, fontsize=8,
                color="#2171b5" if m in esm2 else "#e6550d")

ax.set_ylim(0.60, 0.90)                      # headroom top + room for baseline
ax.set_xscale("log")
ax.set_xlabel("Model size (M parameters, log scale)")
ax.set_ylabel("Normalized RF (species ref, lower = better)")
ax.set_title("Neither scale nor metagenomic training helps")
ax.legend(frameon=False, loc="center left")   # empty mid-band
fig.savefig("figures/fig4_scaling.png"); fig.savefig("figures/fig4_scaling.pdf")
plt.show()

d = res[res.ref == "species"].merge(fam, on=["clade", "og_id"])
edges = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
d["bin"] = pd.cut(d["mean_pid"], edges)
centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(edges)-1)]

g = d.groupby(["bin", "method"], observed=True)["norm_rf"].mean().unstack("method")

fig, ax = plt.subplots(figsize=(8, 4.8))
for t in ORDER:
    lw, z = (2.8, 5) if t == "hamming" else (1.6, 3)
    ax.plot(centers, g[t].values, "-o", color=COLOR[t], lw=lw, zorder=z,
            markersize=5, label=LABEL[t])

ax.set_xlabel("Mean pairwise identity  (← more divergent      less divergent →)")
ax.set_ylabel("Normalized RF (lower = better)")
ax.set_title("Embeddings improve with divergence but never beat Hamming\n(reference: species tree)")
ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
fig.savefig("figures/fig5_divergence.png"); fig.savefig("figures/fig5_divergence.pdf")
plt.show()

pd.crosstab(d["bin"], d["clade"])

emb = (res[(res.ref=="species") & (res.method!="hamming")]
       .groupby(["clade","og_id"])["norm_rf"].mean().rename("emb_rf").reset_index())
emb = emb.merge(fam, on=["clade","og_id"])

fig, ax = plt.subplots(figsize=(8, 5))
for clade in clades:
    s = emb[emb.clade == clade]
    ax.scatter(s["mean_pid"], s["emb_rf"], s=14, alpha=0.5, label=clade)
    if len(s) > 5:
        b, a = np.polyfit(s["mean_pid"], s["emb_rf"], 1)   # slope, intercept
        xs = np.array([s["mean_pid"].min(), s["mean_pid"].max()])
        ax.plot(xs, a + b*xs, lw=2)
ax.set_xlabel("Mean pairwise identity")
ax.set_ylabel("Embedding RF (mean of 6 models, species ref)")
ax.set_title("Within-clade: does RF actually track divergence?")
ax.legend(frameon=False, fontsize=8, ncol=2)
fig.savefig("figures/fig6_clade_divergence.png"); fig.savefig("figures/fig6_clade_divergence.pdf")
plt.show()

from scipy.stats import wilcoxon

sp = (res[res.ref == "species"]
      .pivot_table(index=["clade","og_id"], columns="method", values="norm_rf"))

print("— Claim 1: does Hamming beat each embedding? (one-sided, 175 families) —")
for m in [x for x in ORDER if x != "hamming"]:
    stat, p = wilcoxon(sp["hamming"], sp[m], alternative="less")
    dmed = (sp[m] - sp["hamming"]).median()
    print(f"Hamming < {m:10s}:  median Δ={dmed:+.3f}  p={p:.1e}")

print("\n— Claim 2: ESMC vs size-matched ESM2 (two-sided) —")
for a, b in [("esmc_300m","esm2_150m"), ("esmc_600m","esm2_650m")]:
    stat, p = wilcoxon(sp[a], sp[b])
    print(f"{a} vs {b}:  median Δ={(sp[a]-sp[b]).median():+.3f}  p={p:.3f}")

