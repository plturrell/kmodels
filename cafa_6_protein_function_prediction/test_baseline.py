"""Test the baseline model on a small subset of data."""

import sys
from pathlib import Path

# Test with a small subset first
print("=" * 60)
print("Testing Baseline Model (Small Subset)")
print("=" * 60)

# Import after setting up paths
from src.modeling.baseline import run_training

# Define paths
data_dir = Path(__file__).parent / "data" / "raw" / "cafa-6-protein-function-prediction" / "Train"
fasta_path = data_dir / "train_sequences.fasta"
terms_path = data_dir / "train_terms.tsv"
output_dir = Path(__file__).parent / "outputs" / "baseline_test"

print(f"\nData paths:")
print(f"  FASTA: {fasta_path}")
print(f"  Terms: {terms_path}")
print(f"  Output: {output_dir}")

# Create a small subset for testing
print(f"\nCreating small test subset...")
from src.data import load_sequences_from_fasta, load_go_terms_long_format, build_samples

sequences = load_sequences_from_fasta(fasta_path)
annotations = load_go_terms_long_format(terms_path)
all_samples = build_samples(sequences, annotations)

# Filter to proteins with at least 3 GO terms for meaningful testing
filtered_samples = [s for s in all_samples if len(s.go_terms) >= 3]
print(f"  Total samples: {len(all_samples)}")
print(f"  Samples with ≥3 GO terms: {len(filtered_samples)}")

# Take first 500 for quick test
test_samples = filtered_samples[:500]
print(f"  Using {len(test_samples)} samples for test")

# Create temporary files with subset
import tempfile
import csv

with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)
    
    # Write subset FASTA
    subset_fasta = tmpdir / "subset.fasta"
    with subset_fasta.open("w") as f:
        for sample in test_samples:
            f.write(f">{sample.accession}\n")
            f.write(f"{sample.sequence}\n")
    
    # Write subset terms
    subset_terms = tmpdir / "subset_terms.tsv"
    with subset_terms.open("w") as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['EntryID', 'term', 'aspect'])
        for sample in test_samples:
            for term in sample.go_terms:
                writer.writerow([sample.accession, term, 'P'])  # Aspect doesn't matter for this test
    
    print(f"\nRunning baseline training (without embeddings)...")
    print(f"This may take a few minutes...")
    
    try:
        metrics = run_training(
            fasta_path=subset_fasta,
            annotation_path=subset_terms,
            output_dir=output_dir,
            val_fraction=0.2,
            seed=42,
            max_iter=100,  # Reduced for faster testing
            use_embeddings=False,  # Skip embeddings for quick test
        )
        
        print(f"\n" + "=" * 60)
        print("✓ Baseline Training Completed Successfully!")
        print("=" * 60)
        print(f"\nMetrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        
        print(f"\nModel saved to: {output_dir}")
        
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed!")
print("=" * 60)

