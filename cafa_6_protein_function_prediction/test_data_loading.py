"""Quick test script to verify data loading works correctly."""

from pathlib import Path
from src.data import load_sequences_from_fasta, load_go_terms_long_format, build_samples

# Define paths
data_dir = Path(__file__).parent / "data" / "raw" / "cafa-6-protein-function-prediction" / "Train"
fasta_path = data_dir / "train_sequences.fasta"
terms_path = data_dir / "train_terms.tsv"

print("=" * 60)
print("Testing CAFA 6 Data Loading")
print("=" * 60)

# Load sequences
print(f"\n1. Loading sequences from {fasta_path.name}...")
sequences = load_sequences_from_fasta(fasta_path)
print(f"   ✓ Loaded {len(sequences)} protein sequences")

# Show sample
sample_id = list(sequences.keys())[0]
print(f"   Sample: {sample_id}")
print(f"   Sequence length: {len(sequences[sample_id])}")
print(f"   First 50 chars: {sequences[sample_id][:50]}...")

# Load GO terms
print(f"\n2. Loading GO terms from {terms_path.name}...")
annotations = load_go_terms_long_format(terms_path)
print(f"   ✓ Loaded annotations for {len(annotations)} proteins")

# Show sample
if sample_id in annotations:
    print(f"   Sample: {sample_id}")
    print(f"   GO terms: {annotations[sample_id][:5]}...")
    print(f"   Total terms: {len(annotations[sample_id])}")

# Build samples
print(f"\n3. Building protein samples...")
samples = build_samples(sequences, annotations)
print(f"   ✓ Built {len(samples)} samples")

# Show sample
print(f"\n4. Sample protein:")
sample = samples[0]
print(f"   Accession: {sample.accession}")
print(f"   Sequence length: {len(sample.sequence)}")
print(f"   GO terms: {len(sample.go_terms)}")
print(f"   First 5 GO terms: {sample.go_terms[:5]}")

# Statistics
import numpy as np
terms_per_protein = [len(s.go_terms) for s in samples]
seq_lengths = [len(s.sequence) for s in samples]

print(f"\n5. Dataset Statistics:")
print(f"   Total proteins: {len(samples)}")
print(f"   Avg GO terms per protein: {np.mean(terms_per_protein):.2f}")
print(f"   Median GO terms per protein: {np.median(terms_per_protein):.0f}")
print(f"   Avg sequence length: {np.mean(seq_lengths):.2f}")
print(f"   Median sequence length: {np.median(seq_lengths):.0f}")

print("\n" + "=" * 60)
print("✓ All data loading tests passed!")
print("=" * 60)

