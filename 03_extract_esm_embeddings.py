import pandas as pd
import torch

print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")

df_bench = pd.read_parquet("data_processed/df_bench.parquet")
print(len(df_bench), "proteins")

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

model = ESMC.from_pretrained("esmc_300m").to("cuda").eval()

@torch.no_grad()
def embed_esmc(model, seq):
  prot = model.encode(ESMProtein(sequence=seq))
  out = model.logits(prot, LogitsConfig(sequence=True, return_embeddings=True))

  # ignores BOS/EOS tokes, averages along amino acids
  mean_emb = out.embeddings[0, 1:-1].mean(0)

  return mean_emb.float().cpu().numpy()

import numpy as np
from tqdm import tqdm

# ESM2 was trained on sequences up to 1022 amino acids
MAX_LEN = 1022

EMB_DIR = "data_processed/embeddings"
os.makedirs(EMB_DIR, exist_ok=True)

def extract(model, tag, embed_fn):
  path = f"{EMB_DIR}/{tag}.npy"
  if os.path.exists(path):
    print(tag, "cached")
    return np.load(path)

  vecs = [embed_fn(model, s[:MAX_LEN]) for s in tqdm(df_bench["seq"], desc=tag)]
  arr = np.stack(vecs).astype("float32")
  np.save(path, arr)

  return arr

emb_300 = extract(model, "esmc_300m", embed_esmc)

print(emb_300.shape)

df_bench[["clade","og_id","organism_id","gene_id"]].to_parquet(f"{EMB_DIR}/index.parquet")

del model
torch.cuda.empty_cache()

model = ESMC.from_pretrained("esmc_600m").to("cuda").eval()
emb_600 = extract(model, "esmc_600m", embed_esmc)

print(emb_600.shape)

del model
torch.cuda.empty_cache()

ESM2_MODELS = {
    "esm2_8m":   "facebook/esm2_t6_8M_UR50D",
    "esm2_35m":  "facebook/esm2_t12_35M_UR50D",
    "esm2_150m": "facebook/esm2_t30_150M_UR50D",
    "esm2_650m": "facebook/esm2_t33_650M_UR50D",
}

from transformers import AutoTokenizer, AutoModel

@torch.no_grad()
def embed_esm2(model, seq, tokenizer):
  enc = tokenizer(seq, return_tensors="pt").to("cuda")
  out = model(**enc).last_hidden_state
  return out[0, 1:-1].mean(0).float().cpu().numpy()

for tag, ckpt in ESM2_MODELS.items():
  path = f"{EMB_DIR}/{tag}.npy"
  if os.path.exists(path):
    print(tag, "cached")
    continue

  tok = AutoTokenizer.from_pretrained(ckpt)
  esm2  = AutoModel.from_pretrained(ckpt).to("cuda").eval()

  arr = extract(esm2, tag, lambda m, s: embed_esm2(m, s, tok))
  print(f"{tag:12s} {arr.shape}")

  del esm2, tok
  torch.cuda.empty_cache()

L = df_bench["seq"].str.len()
print(L.describe())
print((L > 1022).sum(), "proteins over ~1022 aa")

idx = pd.read_parquet(f"{EMB_DIR}/index.parquet")
for tag in ["esmc_300m","esmc_600m","esm2_8m","esm2_35m","esm2_150m","esm2_650m"]:
    a = np.load(f"{EMB_DIR}/{tag}.npy")
    assert a.shape[0] == len(idx) == 2625, tag
    print(f"{tag:12s} {a.shape}")