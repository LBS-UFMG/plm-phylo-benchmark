# Correlation Is Not Reconstruction: A Controlled Benchmark of Phylogenetic Signal in Protein Language Model Embeddings

A controlled benchmark addressing a single question: do protein language model embeddings encode phylogeny?

Two lines of work disagree. Some report that embedding distances correlate with phylogenetic distance, while others find that trees reconstructed from those same embeddings fail to surpass a trivial baseline. The two criteria had never been measured on the same data, which is what this benchmark does.

We assembled 175 single-copy orthologous families across seven clades spanning bacteria, archaea, and eukaryotes, with one sequence per species and an independent species-tree reference, holding everything fixed except the distance source. We then scored six models (ESM-2 at 8M/35M/150M/650M, ESM-C at 300M/600M) against a simple Hamming distance, under both criteria on identical data.

**The two criteria agree.** Embedding distances are moderately correlated with phylogeny (ρ ≈ 0.55) but reconstruct trees only somewhat better than random. A Hamming distance is both more correlated (ρ ≈ 0.90) and more accurate at reconstruction, in every clade and at every model scale. Correcting the embeddings' anisotropy helps but does not close the gap, and neither ESM-2 scaling nor ESM-C's metagenomic-scale training improves the metrics. For reconstructing phylogenies, standard single-sequence embeddings are not yet a substitute for alignment-based distances.

## The pipeline

The five notebooks run in order:

| | |
|---|---|
| `01_build_benchmark` | Select clades, species, and single-copy families from OrthoDB v12v2 |
| `02_alignment_and_reference_trees` | MAFFT alignments, IQ-TREE 2 gene trees, NCBI species trees |
| `03_extract_esm_embeddings` | Mean-pooled last-layer embeddings for all six models |
| `04_score_trees` | Cosine/Hamming distances, neighbor joining, RF and correlation scoring |
| `05_figures` | The result figures in the paper |

The overview schematic (Figure 1) is not produced by a notebook. It is drawn directly in the LaTeX source with TikZ.

The benchmark is fully deterministic, with fixed random seeds per clade, so that a clean run exactly reproduces reported results.

## Data

Three OrthoDB v12v2 tables (`levels`, `level2species`, `species`) plus ortholog groups and sequences pulled live from the OrthoDB API. Notebook `01` handles the download.

## Citing

Arcanjo, A., Lemos, R. P., Mariano, D., and de Melo-Minardi, R. C. *Correlation Is Not Reconstruction: A Controlled Benchmark of Phylogenetic Signal in Protein Language Model Embeddings.* To be Published, 2026.
