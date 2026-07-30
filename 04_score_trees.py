import pandas as pd
import numpy as np
import json
import glob

EMB_DIR   = "data_processed/embeddings"
ALIGN_DIR = "data_processed/align"
ML_DIR    = "data_processed/trees_max_likelihood"
SP_DIR    = "data_processed/trees_species"

bench = json.load(open("data_processed/benchmark_families.json"))
# clade -> orthologous families
families = bench["families"]
index = pd.read_parquet(f"{EMB_DIR}/index.parquet").reset_index(drop=True)

MODELS = [
    "esmc_300m",
    "esmc_600m",
    "esm2_8m",
    "esm2_35m",
    "esm2_150m",
    "esm2_650m"
]

emb = {model: np.load(f"{EMB_DIR}/{model}.npy") for model in MODELS}

# returns row indexes and organism_id list of a given (clade, orthologous family) pair
def family_rows(clade, og):
  mask = (index["clade"] == clade) & (index["og_id"] == og)
  sub = index[mask]

  return sub.index.to_numpy(), sub["organism_id"].tolist()

from scipy.spatial.distance import pdist, squareform

# calculates cosine distance for every pair of vectors in a given array of vector
# minimum value is 0, means vectors point to the same direction
# value 1 means vectors are orthognal
# maximum value is 2, means vectors point to oposite directions
def cosine_dm(vectors):
  return squareform(pdist(vectors, metric="cosine"))

def dist_dm(vectors, metric):
    return squareform(pdist(vectors, metric=metric))

from skbio import DistanceMatrix
from skbio.tree import nj
from ete3 import Tree

# builds newick tree given distance matrix
def nj_tree(D, tips):
  # converts D into a DistanceMatrix object of skbio
  dm = DistanceMatrix(D, ids=tips)

  # creates a tree using Neighbor-Joining algorithm
  skbio_tree_node = nj(dm)

  # converts the tree to a ete3 newick tree
  tree = Tree(str(skbio_tree_node))

  for leaf in tree:
    # remove skbio quotes
    leaf.name = leaf.name.strip("'")

  return tree

from Bio import AlignIO
import itertools

# calculates Hamming distance matrix for every pair of (aligned) sequences
# of a given (clade, orthologus family) pair
def hamming_dm(clade, og):
  aln = AlignIO.read(f"{ALIGN_DIR}/{clade}/{og}.aln", "fasta")
  ids = [r.id for r in aln]
  seqs = [str(r.seq) for r in aln]

  n = len(seqs)
  D = np.zeros((n,n))
  for i, j in itertools.combinations(range(n),2):
    pos = [(x,y) for x,y in zip(seqs[i],seqs[j]) if x != "-" and y != "-"]
    d = sum(x != y for x,y in pos) / len(pos) if pos else 0.0
    D[i,j] = D[j,i] = d
  return D, ids

# calculates Normalized Robson-Fould distance between two trees
def norm_rf(t1, t2):
  common = set(t1.get_leaf_names()) & set(t2.get_leaf_names())
  a, b = t1.copy(), t2.copy()
  a.prune(common)
  b.prune(common)
  return a.compare(b, unrooted=True)["norm_rf"]

# returns Normalized Robinson-Foulds between tree given by (clade, og, distance method)
# against a reference tree (maximum likelihood based or species based)
def score_family(clade, og, ml_ref, sp_ref):
  # row indexes, organism_id list
  rows, tips = family_rows(clade, og)

  # hamming distance matrix, organism_id list
  Dh, ids_h = hamming_dm(clade, og)

  # trees dictionary
  # first entry is NJ tree based on hamming distance matrix
  trees = {"hamming": nj_tree(Dh, ids_h)}

  # NJ trees based on cosine similiarty of ESM models
  for model in MODELS:
    trees[model] = nj_tree(cosine_dm(emb[model][rows]), tips)

  out = []
  for method, tr in trees.items():
    for refname, ref in [("ml", ml_ref), ("species", sp_ref)]:
      out.append({
          "clade": clade,
          "og_id": og,
          "method": method,
          "ref": refname,
          "norm_rf": norm_rf(tr, ref)
      })
  return out

from tqdm import tqdm

sp_refs = {c: Tree(f"{SP_DIR}/{c}.nwk", format=1) for c in families}
# (clade, orthologous family) pairs
pairs = [(c,o) for c in families for o in families[c]]

records = []
for clade, og in tqdm(pairs, total=len(pairs)):
  ml_ref = Tree(f"{ML_DIR}/{clade}/{og}.treefile")
  sp_ref = sp_refs[clade]
  records += score_family(clade, og, ml_ref, sp_ref)

res = pd.DataFrame(records)
print(len(res), "rows (expect 175 * 14 = 2450)")
res.to_parquet("data_processed/rf_scores.parquet")

from scipy.stats import spearmanr
from skbio import TreeNode

# calculates patristic distance over tips of a newick tree
def patristic_dm(treefile, tips):
  t = TreeNode.read(treefile)
  for tip in t.tips():
    # undo Newick exchanging underscore for space
    tip.name = tip.name.replace(" ", "_")
  dm = t.tip_tip_distances()
  return dm.filter(tips).data

# calculates the Spearman corretion between the entries of two distance matrices
# (diagonals are excluded since they are always zero)
def dm_corr(D1, D2):
  iu = np.triu_indices(D1.shape[0], k=1)
  return spearmanr(D1[iu], D2[iu]).correlation

patr = {}
for clade, og in pairs:
  rows, tips = family_rows(clade, og)
  Dp = patristic_dm(f"{ML_DIR}/{clade}/{og}.treefile", tips)
  patr[(clade, og)] = (rows, tips, Dp)

rows_out = []
for model in tqdm(MODELS):
    for metric in ["cosine", "euclidean"]:
        corrs, rfs = [], []
        for (clade, og), (rows, tips, Dp) in patr.items():   # rows/tips/Dp all aligned
            D = dist_dm(emb[model][rows], metric)
            corrs.append(dm_corr(D, Dp))
            t = nj_tree(D, tips)
            rfs.append(norm_rf(t, sp_refs[clade]))
        rows_out.append({"model": model,
                         "metric": metric,
                         "mean_corr": np.mean(corrs),
                         "mean_rf": np.mean(rfs)})

abl = pd.DataFrame(rows_out)
print(abl.pivot_table(index="model",
                      columns="metric",
                      values=["mean_corr", "mean_rf"]).round(3))

corr_records = []
for clade, og in tqdm(pairs, total=len(pairs)):
  rows, tips = family_rows(clade, og)
  Dp = patristic_dm(f"{ML_DIR}/{clade}/{og}.treefile", tips)
  Dh, ids_h = hamming_dm(clade, og)
  assert ids_h == tips
  corr_records.append({
      "clade": clade,
      "og_id": og,
      "method": "hamming",
      "corr": dm_corr(Dh, Dp)
  })

  for model in MODELS:
    De = cosine_dm(emb[model][rows])
    corr_records.append({
        "clade": clade,
        "og_id": og,
        "method": model,
        "corr": dm_corr(De, Dp)
    })

dcorr = pd.DataFrame(corr_records)
dcorr.to_parquet("data_processed/dist_corr.parquet")

print(dcorr.groupby("method")["corr"].mean().round(3))

# given a set of vectors X it moves its mean to the origin
# and then projets out the top-d principal directions
def all_but_the_top(X,d):
  Xc = X - X.mean(axis=0)
  if d>0:
    # Vt is the Vt in X = USVt of singular value decompostion
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    # top d directions
    V = Vt[:d]
    Xc = Xc - (Xc @V.T) @ V
  return Xc

# raw mean no correction
# d=0 correspond to just mean-centering
D_VALUES = ["raw", 0, 1, 5, 10, 20, 50, 100, 200]

rows_out = []
for model in tqdm(MODELS):
  for d in D_VALUES:
    Xc = emb[model] if d == "raw" else all_but_the_top(emb[model], d=d)
    cs = [dm_corr(cosine_dm(Xc[rows]), Dp) for (rows, tips, Dp) in patr.values()]
    rows_out.append({"model": model, "d": d, "mean_corr": np.mean(cs)})

sweep = pd.DataFrame(rows_out)

print(sweep.pivot_table(index="model", columns="d", values="mean_corr").round(3))

BEST_D = 100

emb_corr = {model: all_but_the_top(emb[model], BEST_D) for model in MODELS}

rf_rows = []
for clade, og in tqdm(pairs, total=len(pairs)):
  rows, tips = family_rows(clade, og)
  sp_ref = sp_refs[clade]
  for model in MODELS:
    for tag, X in [("raw", emb[model]), ("corr", emb_corr[model])]:
      t = nj_tree(cosine_dm(X[rows]), tips)
      rf_rows.append({"clade": clade,
                      "og_id": og,
                      "model": model,
                      "variant": tag,
                      "rf_species": norm_rf(t, sp_ref)})

rfc = pd.DataFrame(rf_rows)

print(rfc.groupby("variant")["rf_species"].mean().round(3))
print(rfc.groupby(["model", "variant"])["rf_species"].mean().round(3).unstack())

from scipy.stats import wilcoxon

# 1. Is corrected significantly better than raw? (paired over family x model)
piv = rfc.pivot_table(index=["clade","og_id","model"], columns="variant", values="rf_species")
d = (piv["raw"] - piv["corr"])
print("raw vs corr:  median improvement", round(d.median(), 3),
      " p =", wilcoxon(piv["raw"], piv["corr"], alternative="greater").pvalue)

# 2. Is corrected still significantly worse than Hamming? (per family)
ham = (res[(res.ref=="species") & (res.method=="hamming")]
       .set_index(["clade","og_id"])["norm_rf"])
cor = rfc[rfc.variant=="corr"].groupby(["clade","og_id"])["rf_species"].mean()
comp = pd.DataFrame({"ham": ham, "cor": cor}).dropna()
print("corr vs Hamming:  median gap", round((comp["cor"]-comp["ham"]).median(), 3),
      " p =", wilcoxon(comp["cor"], comp["ham"], alternative="greater").pvalue)

print(res[res.ref=="species"].groupby("method")["norm_rf"].mean())

print(dcorr.groupby("method")["corr"].mean())

from scipy.stats import wilcoxon
import numpy as np

EMB = MODELS  # the six embedding models
ORDER = EMB + ["hamming"]

print("="*60)
print("1. SPECIES-tree RF, per method (mean)")
print(res[res.ref=="species"].groupby("method")["norm_rf"].mean().reindex(ORDER).round(3))

print("\n2. ML-tree RF, per method (mean)")
print(res[res.ref=="ml"].groupby("method")["norm_rf"].mean().reindex(ORDER).round(3))

print("\n3. Claim 1 — Hamming vs each embedding (species ref, one-sided)")
sp = res[res.ref=="species"].pivot_table(index=["clade","og_id"],
                                         columns="method", values="norm_rf")
for m in EMB:
    p = wilcoxon(sp["hamming"], sp[m], alternative="less").pvalue
    print(f"  Hamming < {m:10s}  medianΔ={ (sp[m]-sp['hamming']).median():+.3f}  p={p:.1e}")

print("\n4. Claim 2 — ESM-C vs size-matched ESM-2 (species ref, two-sided)")
for a,b in [("esmc_300m","esm2_150m"), ("esmc_600m","esm2_650m")]:
    p = wilcoxon(sp[a], sp[b]).pvalue
    print(f"  {a} vs {b}  meanΔ={sp[a].mean()-sp[b].mean():+.3f}  p={p:.3f}")

print("\n5. Distance correlation, per method (mean)")
print(dcorr.groupby("method")["corr"].mean().reindex(ORDER).round(3))

print("\n6. Isotropy correction (d=100), species RF")
print("  raw vs corr mean RF:")
print(rfc.groupby("variant")["rf_species"].mean().round(3).to_string())
piv = rfc.pivot_table(index=["clade","og_id","model"], columns="variant", values="rf_species")
print(f"  raw->corr improvement: mean {(piv['raw']-piv['corr']).mean():+.3f}"
      f"  p={wilcoxon(piv['raw'], piv['corr'], alternative='greater').pvalue:.1e}")
ham = res[(res.ref=='species')&(res.method=='hamming')].set_index(['clade','og_id'])['norm_rf']
cor = rfc[rfc.variant=='corr'].groupby(['clade','og_id'])['rf_species'].mean()
comp = pd.DataFrame({"ham":ham,"cor":cor}).dropna()
print(f"  corr vs Hamming: mean gap {(comp['cor']-comp['ham']).mean():+.3f}"
      f"  median {(comp['cor']-comp['ham']).median():+.3f}"
      f"  p={wilcoxon(comp['cor'], comp['ham'], alternative='greater').pvalue:.1e}")
print("="*60)

sp_refs

from ete3 import Tree
import numpy as np

species = bench["species"]

rng = np.random.default_rng(0)
rand_rf = []
for clade in families:
    tips = list(species[clade])
    ref = sp_refs[clade]
    for _ in range(10000):
        t = Tree()
        t.populate(len(tips), names_library=tips)
        t.unroot()
        rand_rf.append(norm_rf(t, ref))

rand_rf = np.array(rand_rf)
print(f"random-tree normRF: mean {rand_rf.mean():.3f}  "
      f"5-95% [{np.percentile(rand_rf,5):.3f}, {np.percentile(rand_rf,95):.3f}]")

import pandas as pd, numpy as np
from scipy.stats import binomtest, wilcoxon
from statsmodels.stats.multitest import multipletests

rf = pd.read_parquet("data_processed/rf_scores.parquet")
rf = rf[rf["ref"] == "species"]   # match column/value to your data; run BEFORE the loop

# adjust these names to match your columns
KEY   = ["clade", "og_id", "ref"]   # what pairs a Hamming row to an embedding row
SRC   = "method"                   # column holding 'hamming' / model names
VALUE = "norm_rf"

ham = rf[rf[SRC] == "hamming"].set_index(KEY)[VALUE]
models = [m for m in rf[SRC].unique() if m != "hamming"]

rows = []
for m in models:
    emb = rf[rf[SRC] == m].set_index(KEY)[VALUE]
    d = (emb - ham).dropna()                      # positive => Hamming better (lower RF)
    n_eff = (d != 0).sum()                         # sign test ignores exact ties
    wins  = (d > 0).sum()                          # families where Hamming wins
    p_sign = binomtest(wins, n_eff, 0.5, alternative="two-sided").pvalue
    p_wilc = wilcoxon(d, zero_method="wilcox").pvalue
    rows.append({"model": m, "n": len(d), "median_dRF": d.median(),
                 "win_rate": wins / n_eff, "p_sign": p_sign, "p_wilcoxon": p_wilc})

out = pd.DataFrame(rows)
out["p_sign_holm"]     = multipletests(out["p_sign"],     method="holm")[1]
out["p_wilcoxon_holm"] = multipletests(out["p_wilcoxon"], method="holm")[1]
print(out.to_string(index=False))

import inspect
import ete3

print(inspect.getsource(ete3.Tree.populate))

sp = rf[rf["ref"] == "species"]
ham = sp[sp["method"]=="hamming"].set_index(["clade","og_id"])["norm_rf"]
for m in ["esmc_300m","esmc_600m","esm2_8m","esm2_35m","esm2_150m","esm2_650m"]:
    d = (sp[sp["method"]==m].set_index(["clade","og_id"])["norm_rf"] - ham).dropna()
    print(m, "mean dRF %.4f" % d.mean(), "median %.4f" % d.median())
    print("   dRF value counts:", d.round(4).value_counts().sort_index().to_dict())