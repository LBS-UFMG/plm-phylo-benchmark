import pandas as pd
import json

df_bench = pd.read_parquet("data_processed/df_bench.parquet")

bench = json.load(open("data_processed/benchmark_families.json"))
species  = bench["species"]     # clade -> list of organism_id  (15 each)
families = bench["families"]    # clade -> list of og_id        (25 each)

assert len(df_bench) == 2625
assert (df_bench.groupby(["clade", "og_id"])["organism_id"].nunique() == 15).all()
print(df_bench["clade"].value_counts())

import subprocess

ALIGN_DIR = "data_processed/align"

def align_family(clade, og, seqs):
  # creates alignment file
  output = f"{ALIGN_DIR}/{clade}/{og}.aln"
  if os.path.exists(output):
    return output
  os.makedirs(os.path.dirname(output), exist_ok=True)

  # writes a FASTA file with sequences from a given clade-orthologous family pair
  temp = f"{ALIGN_DIR}/{clade}/{og}.faa"
  with open(temp, "w") as fh:
    for sp, seq in seqs:
      fh.write(f">{sp}\n{seq}\n")

  # runs MAFFT to align the sequences on the created FASTA file
  # saves on the alignment file
  tmp = output + ".tmp"
  with open(tmp, "w") as fh:
    subprocess.run(["mafft", "--auto", "--quiet", temp], stdout=fh, check=True)
  os.rename(tmp, output)

  os.remove(temp)
  return output

fail = []
for clade, ogs in families.items():
  for og in ogs:
    seqs = [
        (r.organism_id, r.seq)
        for r in df_bench[(df_bench.clade == clade) & (df_bench.og_id == og)].itertuples()
    ]

    try:
      align_family(clade, og, seqs)
    except Exception as e:
      print(e)
      fail.append((clade, og, str(e)[:60]))

  print(f"{clade:20s} done")

  print("failures:", len(fail))

import glob
print(len(glob.glob(f"{ALIGN_DIR}/*/*.aln")), "alignments (expect 175)")

from Bio import AlignIO
import itertools

# calculates mean pairwise identity
def mean_pid(path):
  aln = AlignIO.read(path, "fasta")
  seqs = [str(r.seq) for r in aln]
  pids = []
  for a,b in itertools.combinations(seqs,2):
    # comparison exclude gaps
    pos = [(x,y) for x,y in zip(a,b) if x != "-" and y != "-"]
    if pos:
      pids.append(sum(x==y for x,y in pos) / len(pos))
  return sum(pids) / len(pids)

rows = []
for clade, ogs in families.items():
  for og in ogs:
    rows.append({
        "clade": clade,
        "og_id": og,
        "mean_pid": mean_pid(f"{ALIGN_DIR}/{clade}/{og}.aln")
    })

fam = pd.DataFrame(rows)
print(fam.groupby("clade")["mean_pid"].agg(["min", "median", "max"]).round(3))

fam.to_parquet("data_processed/family_divergence.parquet")

import os

TREE_DIR = "data_processed/trees_max_likelihood"

def build_max_likelihood_tree(clade, og):
  # creates tree file
  output = f"{TREE_DIR}/{clade}/{og}.treefile"
  if os.path.exists(output):
    return output
  os.makedirs(os.path.dirname(output), exist_ok=True)

  # alignment file
  aln = f"{ALIGN_DIR}/{clade}/{og}.aln"
  # prefix for tree files
  pre = f"{TREE_DIR}/{clade}/{og}"

  # IQ-TREE uses the alingment to build a tree for each family of genes in a given clade
  subprocess.run([
      "iqtree2", "-s", aln, "-m", "MFP", "-pre", pre, "-quiet", "-nt", "AUTO"],
                 check=True)

  return output

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# number of available CPU threads
N_WORKERS = 2

pairs = [(clade, og) for clade, ogs in families.items() for og in ogs]

def run(p):
  try:
    build_max_likelihood_tree(*p)
    return None
  except Exception as e:
    return (*p, str(e)[:60])

with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
  futs = [ex.submit(run, p) for p in pairs]
  fail = [f.result() for f in tqdm(as_completed(futs), total=len(pairs)) if f.result()]

print("failures:", len(fail))

taxid_map = pd.read_parquet("data_processed/taxid_map.parquet")

# organism_id -> ncbi_taxid
org2taxid = dict(zip(taxid_map["organism_id"], taxid_map["ncbi_taxid"]))

from ete3 import NCBITaxa
ncbi = NCBITaxa()

# creates folder for species-based trees
SP_TREE_DIR = "data_processed/trees_species"
os.makedirs(SP_TREE_DIR, exist_ok=True)

def build_species_tree(clade):
  # species of a given clade
  sps = species[clade]
  # NCBI taxid for species in clade
  taxids = [org2taxid[sp] for sp in sps]

  # NCBI topological tree selected species of the clade
  t = ncbi.get_topology(taxids)
  # ncbi_taxid -> organism_id
  taxid2org = {v:k for k,v in org2taxid.items()}

  for leaf in t.get_leaves():
    leaf.name = taxid2org[int(leaf.name)]

  out = f"{SP_TREE_DIR}/{clade}.nwk"
  t.write(format=1, outfile=out)
  return out, t

sp_trees = {}
for clade in families:
  out, t = build_species_tree(clade)
  sp_trees[clade] = t
  print(f"{clade:20s} {len(t)} tips")

orgs = species["Debaryomycetaceae"]
taxids = [org2taxid[o] for o in orgs]
print(len(taxids), "taxids,", len(set(taxids)), "unique")

leaf_taxids = {int(l.name.split("_")[0]) if "_" in l.name else l.name
               for l in sp_trees["Debaryomycetaceae"].get_leaves()}
print(len(sp_trees["Debaryomycetaceae"]), "leaves")

tree_orgs = {l.name for l in sp_trees["Debaryomycetaceae"].get_leaves()}
missing = set(species["Debaryomycetaceae"]) - tree_orgs
print("missing:", missing)
print(taxid_map[taxid_map.organism_id.isin(missing)][["organism_id","ncbi_taxid","name"]])

from ete3 import Tree
import numpy as np

rng = np.random.default_rng(0)
rand_rf = []
for clade in families:                     # over all clades
    tips = species[clade]
    ref = sp_refs[clade]
    for _ in range(1000):
        t = Tree()
        t.populate(len(tips), names_library=list(tips))   # random topology
        t.unroot()
        rand_rf.append(norm_rf(t, ref))

rand_rf = np.array(rand_rf)
print(f"random-tree normRF: mean {rand_rf.mean():.3f}, "
      f"5–95% [{np.percentile(rand_rf,5):.3f}, {np.percentile(rand_rf,95):.3f}]")