import pandas as pd

# OrthoDB levels table. For each level there is an id, name, number of genes,
# number of ortholog groups and number of species
df_levels = pd.read_csv("data/odb12v2_levels.tab", sep="\t", header=None,
                       names=["level_tax_id", "level_name", "n_genes", "n_OGs", "n_species"])

print(df_levels.sample(3))

# naming columns according to README.txt of OrthoDB
for col in ["level_tax_id", "n_genes", "n_OGs", "n_species"]:
  df_levels[col] = pd.to_numeric(df_levels[col], errors="coerce").astype('Int64')

from ete3 import NCBITaxa
ncbi = NCBITaxa()

DOMAINS = {2: "Bacteria", 2157: "Archaea", 2759: "Eukaryota", 10239: "Viruses"}

# returns domain and rank for a given taxid
def annotate(taxid):
  try:
    # gets lineage from root of current taxid
    lineage = ncbi.get_lineage(taxid)
  except ValueError:
    return None, None
  if not lineage:
    return None, None

  # ensures valid, updated taxid
  resolved = lineage[-1]
  # gets taxonomic rank of current taxid
  rank = ncbi.get_rank([resolved]).get(resolved)
  # gets domain of current taxid
  domain = next((n for t, n in DOMAINS.items() if t in lineage), None)

  return domain, rank

# use previous function to map each level to its domain and rank
df_levels[["domain", "rank"]] = pd.DataFrame(
    df_levels["level_tax_id"].apply(annotate).tolist(),
    index=df_levels.index
)

# given warnings are about NCBI Taxonomy ID that changed,
# they are not harmful and automatically handled by ete3

# only viruses have NaN values, they will be excluded anyway
df_levels[df_levels["domain"].isna()]["level_name"]

# we select candidate families/orders having between 50 and 150 species
cand = df_levels[
    df_levels["n_species"].between(50, 150)
    & df_levels["rank"].isin(["family", "order"])
]

print(cand["domain"].value_counts())

# viruses are excluded
cand = cand[cand["domain"] != "Viruses"]

# Eukaryotic, Bacterial and Archaeal families/orders are considered
euk = cand[cand["domain"] == "Eukaryota"].sort_values("n_OGs", ascending=False)
bac = cand[cand["domain"].isin(["Bacteria"])].sort_values("n_OGs", ascending=False)
arc = cand[cand["domain"].isin(["Archaea"])].sort_values("n_OGs", ascending=False)

print(euk[["level_tax_id", "level_name", "rank", "n_species", "n_OGs"]].to_string())

print(bac[["level_tax_id", "level_name", "rank", "n_species", "n_OGs"]].to_string())

print(arc[["level_tax_id", "level_name", "rank", "n_species", "n_OGs"]].to_string())

# selected clades
CLADES = {
    9443:   "Primates",
    7214:   "Drosophilidae",
    766764: "Debaryomycetaceae",
    135620: "Oceanospirillaceae",
    563835: "Chitinophagaceae",
    90964:  "Staphylococcaceae",
    2235:   "Halobacteriales",
}

# maps each organism to its sequence of levels in OrthoDB,
# it allows query which organisms belong to a given level
df_level2species = pd.read_csv("data/odb12v2_level2species.tab", sep="\t", header=None)
df_level2species.columns = ["top_level", "organism_id", "hops", "level_path"]

# each row correspond to a species. The columns are NCBI Taxonomy ID, OrthoDB ID,
# scientific name and genome assembly.
df_species = pd.read_csv("data/odb12v2_species.tab",       sep="\t", header=None)
df_species.columns = ["ncbi_taxid", "organism_id", "name", "assembly", "n_genes", "n_OGs", "mapping_type"]

# last three columns are empty, we named according to README.txt.
assert df_species[["n_genes", "n_OGs", "mapping_type"]].notna().sum().sum() == 0
df_species = df_species.drop(columns=["n_genes", "n_OGs", "mapping_type"])

# source of genome assemblies
df_species["asm_source"] = df_species["assembly"].str[:3]
print(df_species["asm_source"].value_counts())

# only a virus does not come from GFC or GCA
df_species[df_species["asm_source"] == "NC_"]

# converts string to set for faster search
def path_to_set(s):
  return {int(x) for x in s.strip("{}").split(",") if x}

df_level2species["level_set"] = df_level2species["level_path"].apply(path_to_set)

# maps each selected family/order to its list of species
pools = {
    name: df_level2species[df_level2species["level_set"].apply(lambda s: tax in s)]["organism_id"].tolist()
    for tax, name in CLADES.items()
}

# expected number of species in each selected clade
expected = (df_levels.set_index("level_tax_id")
                     .loc[list(CLADES.keys()), "n_species"]
                     .rename(index=CLADES).to_dict())

for name, orgs in pools.items():
    print(f"{name:20s} got {len(orgs):4d}   expected {expected[name]:4d}")

# assigns each organism to one of the selected clades
df_pool = pd.concat(
  [df_species[df_species["organism_id"].isin(orgs)].assign(clade=name)
  for name, orgs in pools.items()],
  ignore_index=True
)

print(len(df_pool), "should be 586")
print(pd.crosstab(df_pool["clade"], df_pool["asm_source"]))

import requests, time

API = "https://data.orthodb.org/v12-2/search"
HEADERS = {"User-Agent": "BSB2026-research/0.1 (arcanjomjr@gmail.com)"}

# retrieves count of orthologous groups for a given clade
def og_count(tax, universal, singlecopy):
  r = requests.get(API,
                   params={"level":tax,
                           "species":tax,
                           "universal":universal,
                           "singlecopy":singlecopy},
                   headers=HEADERS,
                   timeout=60)
  r.raise_for_status()
  return int(r.json()["count"])

# number of orthologous families that have a single copy in at least
# 90% of the species of a given clade
for tax, name in CLADES.items():
  print(f"{name:20s}  90/90: {og_count(tax, 0.9, 0.9):6d}")
  time.sleep(0.5)

import os, json

os.makedirs("cache", exist_ok=True)

DEFAULT_TARGET = 80
FETCH_TARGET = {"Debaryomycetaceae": 800} #needs more candidates since many are flagged as being of the same species

# fetches and caches Orthologous Groups IDs from OrthoDB given a set of clades
def fetch_og_ids(tax, target):
    path = f"cache/ogids_{tax}.json"
    ids = json.load(open(path)) if os.path.exists(path) else []
    skip = len(ids)
    while len(ids) < target:
        r = requests.get(API, params={"level": tax,
                                      "species": tax,
                                      "universal": 0.9,
                                      "singlecopy": 0.9,
                                      "skip": skip},
                         headers=HEADERS, timeout=60)
        r.raise_for_status()
        page = r.json().get("data") or []
        if not page:
            break
        ids.extend(page); skip += len(page)
        time.sleep(0.3)
    ids = list(dict.fromkeys(ids))
    json.dump(ids, open(path, "w"))
    return ids[:target]

og_ids = {name: fetch_og_ids(tax, FETCH_TARGET.get(name, DEFAULT_TARGET))
          for tax, name in CLADES.items()}
for name, ids in og_ids.items():
    print(f"{name:20s} {len(ids):4d} candidate IDs")

FASTA_API = "https://data.orthodb.org/v12-2/fasta"
os.makedirs("cache/fasta", exist_ok=True)

for name, ids in og_ids.items():
    target = FETCH_TARGET.get(name, DEFAULT_TARGET)
    todo = ids[:target]
    for og in todo:
        path = f"cache/fasta/{og}.fa"
        if os.path.exists(path):
            continue
        try:
            r = requests.get(FASTA_API, params={"id": og}, headers=HEADERS, timeout=120)
            r.raise_for_status()
            tmp = f"cache/fasta/{og}.fa.tmp"
            with open(tmp, "w") as fh:
                fh.write(r.text)
            os.rename(tmp, f"cache/fasta/{og}.fa")
        except Exception as e:
            print(name, og, str(e)[:60]); time.sleep(10)
        time.sleep(1.0)
    have = sum(1 for og in todo if os.path.exists(f"cache/fasta/{og}.fa"))
    print(f"{name:20s} {have}/{target}")

# parses FASTA files from OrthoDB
# these FASTA files have JSON-formatted information in their header
def parse_orthodb_fasta(text):
    recs = []
    for block in text.split(">")[1:]:
        lines = block.split("\n")
        gene_id, meta_json = lines[0].split(" ", 1)
        meta = json.loads(meta_json)
        recs.append({
            "gene_id":       gene_id,
            "organism_id":   meta["organism_taxid"],
            "organism_name": meta["organism_name"],
            "seq":           "".join(lines[1:]).strip(),
        })
    return recs

rows = []
for name, ids in og_ids.items():
  for og in ids:
    path = f"cache/fasta/{og}.fa"
    if not os.path.exists(path):
      continue
    try:
      recs = parse_orthodb_fasta(open(path).read())
    except Exception as e:
      print("BAD:", path, str(e)[:60])
      os.remove(path)
      continue
    for rec in recs:
      rec["clade"] = name
      rec["og_id"] = og
      rows.append(rec)

df_genes = pd.DataFrame(rows)

print(len(df_genes), "sequences")

print(df_genes.groupby("clade")["og_id"].nunique())

# no duplicated genes whithin a single orthologous group
assert df_genes.duplicated(["og_id", "gene_id"]).sum() == 0

# number of orthologous groups by clade
totals = df_genes.groupby("clade")["og_id"].nunique()

# number of orthologous groups by organism (identified by organism_id)
cov = df_genes.groupby(["clade", "organism_id"])["og_id"].nunique().rename("n_ogs").reset_index()

# fraction of orthologous groups a given organism has a gene (in relation to the total number of the corresponding clade)
cov["frac"] = cov["n_ogs"] / cov["clade"].map(totals)

for name in CLADES.values():
  sub = cov[cov["clade"] == name]["frac"]
  print(f"{name:20s} n={len(sub):3d} min={sub.min():.2f}  "
        f"q25={sub.quantile(0.25):.2f}  med={sub.median():.2f}  "
        f"q75={sub.quantile(0.75):.2f}  max={sub.max():.2f}")

N_SPECIES = 15
MAX_PER_GENUS = 5

# excludes halophilic/environmental, keep clinical arm host-associated
EXCLUDE_GENERA = {"Staphylococcaceae": {"Salinicoccus"}}

cov2 = cov.merge(df_species[["organism_id", "ncbi_taxid", "name"]],
                 on="organism_id", how="left")

# returns the species-leves in the hierachy of a given taxid
# specially useful when dealing with strains
def species_taxid(tx):
  lin = ncbi.get_lineage(tx)
  ranks = ncbi.get_rank(lin)
  return next((t for t in reversed(lin) if ranks.get(t) == "species"), tx)

selected = {}
for name in CLADES.values():
  sub = (cov2[cov2["clade"] == name]
         .assign(genus=lambda d: d["name"].str.split().str[0], sp_taxid=lambda d: d["ncbi_taxid"].apply(species_taxid))
         .sort_values("frac", ascending=False)
         .drop_duplicates("sp_taxid")) #single assembly per species
  sub = sub[~sub["genus"].isin(EXCLUDE_GENERA.get(name, set()))]

  chosen = sub.groupby("genus", group_keys=False).head(MAX_PER_GENUS).head(N_SPECIES)
  if len(chosen) < N_SPECIES:
    rest = sub[~sub["organism_id"].isin(chosen["organism_id"])]
    chosen = pd.concat([chosen, rest.head(N_SPECIES - len(chosen))])

  selected[name] = chosen["organism_id"].tolist()

  print(f"\n{name:20s} n={len(chosen)}  frac {chosen['frac'].min():.2f}-{chosen['frac'].max():.2f}")
  print("  ", dict(chosen["genus"].value_counts()))

org2tax = dict(zip(df_species["organism_id"], df_species["ncbi_taxid"]))
for name in CLADES.values():
    sp = [species_taxid(org2tax[o]) for o in selected[name]]
    assert len(set(sp)) == N_SPECIES, f"{name}: duplicate species in selection"

# maps clade to respective set of selected organisms ids
sel_sets = {name: set(ids) for name, ids in selected.items()}

# keeps only selected organisms
mask = [org in sel_sets[cl]
        for org, cl in zip(df_genes["organism_id"], df_genes["clade"])]
df_sel = df_genes[mask]

# counts the number of times each combinatation clade-orthologous group-organsim appears
cnt = (df_sel.groupby(["clade", "og_id", "organism_id"])
             .size().rename("n").reset_index())
# counts two quantitites, for each combination of clade-orthologous group:
# the number of species in that clade that has a gene in that orthologous group
# the maximum number of genes in that orthologous group that a species in that clade has
qual = (cnt.groupby(["clade", "og_id"])
           .agg(n_species=("organism_id", "nunique"),
                max_copies=("n", "max"))
           .reset_index())

qual["ok"] = (qual["n_species"] == N_SPECIES) & (qual["max_copies"] == 1)

print(qual.groupby("clade")["ok"].agg(["sum", "size"]))

ok = qual[(qual["clade"] == "Debaryomycetaceae") & qual["ok"]]
print(len(ok), "ok families (need ≥25)")

bad = qual[~qual["ok"]]
print(bad["max_copies"].gt(1).sum(), "failed on duplicates")
print((bad["n_species"] < N_SPECIES).sum(), "failed on missing species")

dup = qual[~qual["ok"] & (qual["max_copies"] > 1)]
print(dup.groupby("clade").size())

for name, ids in selected.items():
    tax = df_species[df_species["organism_id"].isin(ids)]["ncbi_taxid"]
    print(f"{name:20s} {len(ids)} orgs, {tax.nunique()} distinct taxids")

import numpy as np

N_FAM = 25
NP_SEED = 0

benchmark = {}
for i, name in enumerate(CLADES.values()):
  r = np.random.default_rng([NP_SEED, i])
  # og_id list for a given clade that is "ok" (present as a single copy in all species)
  ok_list = qual[(qual["clade"] == name) & qual["ok"]]["og_id"].tolist()
  # choses N_FAM og_ids from the ok_list
  benchmark[name] = sorted(r.choice(ok_list, N_FAM, replace=False).tolist())

print({k:len(v) for k,v in benchmark.items()})

os.makedirs("data_processed", exist_ok=True)
json.dump({"seed": NP_SEED,
           "n_fam": N_FAM,
           "n_species": N_SPECIES,
           "species": {k:v for k,v in selected.items()},
           "families": benchmark},
           open("data_processed/benchmark_families.json", "w"), indent=2)

for name in CLADES.values():
    ok_list = qual[(qual["clade"] == name) & qual["ok"]]["og_id"].tolist()
    print(f"{name:20s} {len(ok_list):3d} ok families")

# set of selected clade-orthologous families pairs
sel_pairs = {(cl,og) for cl, ogs in benchmark.items() for og in ogs}

# set of selected clade-species pairs
sel_species = {(cl,sp) for cl, sps in selected.items() for sp in sps}

keep = [
    ((cl,og) in sel_pairs) and ((cl,sp) in sel_species)
    for cl, og, sp in zip(df_genes["clade"], df_genes["og_id"], df_genes["organism_id"])
]

df_bench = df_genes[keep].copy()

# Total number of sequences is equal to the expected
assert len(df_bench) == N_FAM * N_SPECIES * len(CLADES) == 2625

# each pair clade-orthologous family has exactly 15 organisms
per = df_bench.groupby(["clade", "og_id"])["organism_id"].nunique()
assert (per == N_SPECIES).all(), "some family lacks exactly 15 species"

# save benchmark dataframe
df_bench.reset_index(drop=True).to_parquet("data_processed/df_bench.parquet")

# maps each organism to its NCBI taxonomy ID
taxid_map = (
    df_bench[["clade", "organism_id"]]
    .drop_duplicates()
    .merge(df_species[["organism_id", "ncbi_taxid", "name"]],
           on="organism_id", how="left")
)

# every organism is mapped to a NCBI Taxonomy ID
assert taxid_map["ncbi_taxid"].notna().all()

# 105 organisms
assert len(taxid_map) == 15 * len(CLADES)

# save taxid map
taxid_map.to_parquet("data_processed/taxid_map.parquet")

old = json.load(open("data_processed/benchmark_families_OLD.json"))["families"]
new = json.load(open("data_processed/benchmark_families.json"))["families"]
for clade in new:
    print(f"{clade:20s} {'CHANGED' if set(old[clade]) != set(new[clade]) else 'same'}")

len(qual[(qual.clade=="Debaryomycetaceae") & qual.ok])

qual.groupby("clade")["ok"].agg(["sum","size"])
