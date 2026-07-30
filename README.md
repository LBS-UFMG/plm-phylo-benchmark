# Correlation Is Not Reconstruction: A Controlled Cross-Domain Benchmark of Phylogenetic Signal in Protein Language Model Embeddings

A controlled benchmark addressing a single question: do protein language model embeddings encode phylogeny?

Two lines of work disagree. Some report that embedding distances correlate with phylogenetic distance, others find that trees built from those same embeddings can't beat a trivial baseline. Nobody had measured both on the same data. So we did.

We assembled 175 single-copy orthologous families across seven clades spanning bacteria, archaea, and eukaryotes, one sequence per species, an independent species-tree reference, everything held fixed except the distance source. Then we scored six models (ESM-2 at 8M/35M/150M/650M, ESM-C at 300M/600M) against a plain Hamming distance, under both criteria at once.

**The two criteria agree.** Embedding distances are moderately correlated with phylogeny (ρ ≈ 0.55) but reconstruct trees only somewhat better than random. A Hamming distance is both more correlated (ρ ≈ 0.90) and more accurate at reconstruction, in every clade, at every model scale. Correcting the embeddings' anisotropy helps but doesn't close the gap, neither ESM-2 scaling nor ESM-C's metagenomic-scale training improves the metrics. For building phylogenies, standard single-sequence embeddings are not yet a substitute for alignment-based distances.

## The pipeline

The five notebooks run in order:

| | |
|---|---|
| `01_build_benchmark` | Select clades, species, and single-copy families from OrthoDB v12v2 |
| `02_alignment_and_reference_trees` | MAFFT alignments, IQ-TREE 2 gene trees, NCBI species trees |
| `03_extract_esm_embeddings` | Mean-pooled last-layer embeddings for all six models |
| `04_score_trees` | Cosine/Hamming distances, neighbor joining, RF and correlation scoring |
| `05_figures` | Every figure in the paper |

The benchmark is fully deterministic, with fixed random seeds per clade, so that a clean run exactly reproduces reported results.

## Data

Three OrthoDB v12v2 tables (`levels`, `level2species`, `species`) plus ortholog groups and sequences pulled live from the OrthoDB API. Notebook `01` handles the download.

## Citing

Arcanjo, A. and de Melo-Minardi, R. C. *Correlation Is Not Reconstruction: A Controlled Cross-Domain Benchmark of Phylogenetic Signal in Protein Language Model Embeddings.* To be Published, 2026.
