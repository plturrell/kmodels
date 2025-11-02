"""Generate comprehensive training results report."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

def generate_report():
    """Generate comprehensive training report."""
    
    print("="*70)
    print("FULL PRODUCTION TRAINING RUN - COMPREHENSIVE RESULTS")
    print("="*70)
    print()
    
    # Load base model metrics
    print("📊 BASE MODEL PERFORMANCE")
    print("-"*70)
    
    models_metrics = {}
    models_oof = {}
    models_test = {}
    
    for i in range(1, 5):
        model_dir = Path(f"outputs/full_production_run/model{i}")
        run_dirs = sorted(model_dir.glob("run-*"))
        
        if run_dirs:
            run_dir = run_dirs[-1]
            
            # Load metrics
            metrics_file = run_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    models_metrics[f"Model {i}"] = json.load(f)
            
            # Load OOF
            oof_file = run_dir / "oof_predictions.csv"
            if oof_file.exists():
                models_oof[f"Model {i}"] = pd.read_csv(oof_file)
            
            # Load test predictions
            test_file = run_dir / "submission.csv"
            if test_file.exists():
                models_test[f"Model {i}"] = pd.read_csv(test_file)
    
    # Display metrics
    print("\nCross-Validation RMSE:")
    sorted_models = sorted(
        models_metrics.items(),
        key=lambda x: x[1].get('rmse_mean', 999)
    )
    
    for model_name, metrics in sorted_models:
        rmse_mean = metrics.get('rmse_mean', 0)
        rmse_std = metrics.get('rmse_std', 0)
        print(f"  {model_name}: {rmse_mean:.6f} ± {rmse_std:.6f}")
    
    # Load training data for OOF analysis
    train_df = pd.read_csv("data/raw/train.csv")
    
    print("\n📈 OOF PREDICTION ANALYSIS")
    print("-"*70)
    
    for model_name, oof_df in models_oof.items():
        merged = train_df.merge(oof_df, on="date_id")
        if "forward_returns_oof" in merged.columns:
            pred = merged["forward_returns_oof"]
            actual = merged["forward_returns"]
            
            rmse = ((pred - actual) ** 2).mean() ** 0.5
            mae = (pred - actual).abs().mean()
            corr = pred.corr(actual)
            
            print(f"\n{model_name}:")
            print(f"  RMSE: {rmse:.6f}")
            print(f"  MAE:  {mae:.6f}")
            print(f"  Corr: {corr:.4f}")
            print(f"  Range: [{pred.min():.6f}, {pred.max():.6f}]")
    
    # Ensemble results
    print("\n🎯 ENSEMBLE RESULTS")
    print("-"*70)
    
    ensemble_file = Path("outputs/full_production_run/ensemble_final.csv")
    if ensemble_file.exists():
        ensemble_df = pd.read_csv(ensemble_file)
        print(f"\nFinal Ensemble Predictions: {len(ensemble_df)} samples")
        print(f"  Range: [{ensemble_df['prediction'].min():.6f}, {ensemble_df['prediction'].max():.6f}]")
        print(f"  Mean:  {ensemble_df['prediction'].mean():.6f}")
        print(f"  Std:   {ensemble_df['prediction'].std():.6f}")
    
    # Submission package
    print("\n📦 SUBMISSION PACKAGE")
    print("-"*70)
    
    submission_dir = Path("outputs/full_production_run/submission_package")
    if submission_dir.exists():
        files = list(submission_dir.glob("*"))
        print(f"\nGenerated {len(files)} files:")
        for f in sorted(files):
            size = f.stat().st_size
            print(f"  - {f.name} ({size:,} bytes)")
    
    # Model configurations
    print("\n⚙️  MODEL CONFIGURATIONS")
    print("-"*70)
    
    configs = {
        "Model 1": "min_samples_leaf=5, max_iter=300, lr=0.1, depth=8",
        "Model 2": "min_samples_leaf=10, max_iter=400, lr=0.05, depth=10",
        "Model 3": "min_samples_leaf=20, max_iter=500, lr=0.03, depth=12",
        "Model 4": "min_samples_leaf=15, max_iter=350, lr=0.07, depth=9",
    }
    
    for model, config in configs.items():
        print(f"  {model}: {config}")
    
    print("\n" + "="*70)
    print("✅ FULL TRAINING RUN COMPLETE")
    print("="*70)
    print(f"\n📁 Results saved to: outputs/full_production_run/")
    print(f"📦 Submission package: outputs/full_production_run/submission_package/")
    print(f"🎯 Final predictions: outputs/full_production_run/ensemble_final.csv")

if __name__ == "__main__":
    generate_report()

