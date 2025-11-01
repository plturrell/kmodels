import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd

def find_best_checkpoint(model_output_dir: Path) -> Path:
    checkpoints = list(model_output_dir.glob('**/best_model.pt'))
    if not checkpoints:
        raise FileNotFoundError(f'No checkpoints found in {model_output_dir}')
    # A simple heuristic: return the first one found. A more robust approach would be to parse logs.
    return checkpoints[0]

def run_training_stage(args, train_csv_path, output_dir_suffix='', dropout_override=None, aug_policy_override=None):
    base_output_dir = Path(args.output_dir)

    for model_name in args.models:
        model_output_dir = base_output_dir / f'{model_name}{output_dir_suffix}'
        print(f'--- Running Cross-Validation for model: {model_name} ---')

        common_args = [
            sys.executable, '-m',
            'competitions.csiro_biomass.src.train',
            '--train-csv', str(train_csv_path),
            '--test-csv', args.test_csv,
            '--sample-submission', args.sample_submission,
            '--image-dir', args.image_dir,
            '--epochs', str(args.epochs),
            '--batch-size', str(args.batch_size),
            '--snapshot-count', str(args.snapshot_count),
            '--model', model_name,
            '--n-folds', str(args.n_folds),
            '--cv-group-column', args.cv_group_column,
        ]

        if dropout_override is not None:
            common_args.extend(['--backbone-dropout', str(dropout_override)])
        if aug_policy_override is not None:
            common_args.extend(['--aug-policy', aug_policy_override])

        if args.device:
            common_args.extend(['--device', args.device])

        # Run training for each fold
        for fold in range(args.n_folds):
            fold_output_dir = model_output_dir / f'fold-{fold}'
            fold_args = common_args + ['--output-dir', str(fold_output_dir), '--fold', str(fold)]
            print(f'--- Starting training for fold {fold} of {model_name} ---')
            subprocess.run(fold_args, check=True)
            print(f'--- Finished training for fold {fold} of {model_name} ---\n')

def run_cv_ensemble(args):
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(exist_ok=True, parents=True)

    # Stage 1: Initial training
    print('--- Stage 1: Initial Training ---')
    run_training_stage(args, args.train_csv, output_dir_suffix='_initial')

    if args.pseudo_label or args.noisy_student:
        print('--- Stage 2: Pseudo-Labeling and Retraining ---')
        best_model_dir = base_output_dir / f'{args.models[0]}_initial'
        best_checkpoint = find_best_checkpoint(best_model_dir)
        
        pseudo_csv_path = base_output_dir / 'pseudo_labeled_train.csv'

        pseudo_label_args = [
            sys.executable, '-m',
            'competitions.csiro_biomass.src.pseudo_label',
            '--checkpoint-path', str(best_checkpoint),
            '--train-csv', args.train_csv,
            '--test-csv', args.test_csv,
            '--image-dir', args.image_dir,
            '--output-csv', str(pseudo_csv_path),
            '--confidence-threshold', str(args.confidence_threshold),
        ]
        subprocess.run(pseudo_label_args, check=True)

        if args.noisy_student:
            print('--- Running Noisy Student Training ---')
            run_training_stage(args, pseudo_csv_path, output_dir_suffix='_noisy_student', dropout_override=args.student_dropout, aug_policy_override=args.student_aug_policy)
        else:
            run_training_stage(args, pseudo_csv_path, output_dir_suffix='_pseudo')

    # Ensemble all submissions from all models and folds
    print('--- Starting final ensembling ---')
    
    if args.tta:
        print('--- Using Test-Time Augmentation ---')
        # When using TTA, we need to load each model and run prediction
        # This is slower but more accurate.
        all_predictions = []
        device = torch.device(args.device)
        test_df = pd.read_csv(args.test_csv)

        checkpoint_paths = list(base_output_dir.glob('**/best_model.pt'))
        for checkpoint_path in checkpoint_paths:
            model = BiomassLightningModule.load_from_checkpoint(str(checkpoint_path)).to(device)
            test_loader = create_inference_loader(
                test_df,
                image_dir=Path(args.image_dir),
                image_column='image_path', # Assuming 'image_path'
                metadata_columns=[],
                metadata_mean=None,
                metadata_std=None,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                is_tta=True, # Important!
            )
            _, preds, _ = run_prediction_loop(model, test_loader, device, use_tta=True)
            all_predictions.append(preds)
        
        # Average all TTA predictions
        final_predictions = torch.stack(all_predictions).mean(dim=0)
        
        # Create submission dataframe
        sample_submission = pd.read_csv(args.sample_submission)
        predictions_df = pd.DataFrame(final_predictions.numpy(), columns=sample_submission.columns[1:])
        predictions_df['sample_id'] = sample_submission['sample_id']

    else:
        submission_files = list(base_output_dir.glob('**/submission.csv'))
        if not submission_files:
            print('No submission files found. Cannot create ensemble.')
            return

        print(f'Found {len(submission_files)} submission files for ensembling.')

        # Read and average predictions
        sample_submission = pd.read_csv(args.sample_submission)
        predictions_df = pd.DataFrame(sample_submission[['sample_id']])
        target_columns = [col for col in sample_submission.columns if col != 'sample_id']

        for col in target_columns:
            predictions_df[col] = 0

        for sub_file in submission_files:
            sub_df = pd.read_csv(sub_file)
            for col in target_columns:
                predictions_df[col] += sub_df[col] / len(submission_files)

    ensemble_submission_path = base_output_dir / 'ensemble_submission.csv'
    predictions_df.to_csv(ensemble_submission_path, index=False)
    print(f'Ensemble submission saved to: {ensemble_submission_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Cross-Validation and Model Ensemble')
    parser.add_argument('--train-csv', type=str, required=True)
    parser.add_argument('--test-csv', type=str, required=True)
    parser.add_argument('--sample-submission', type=str, required=True)
    parser.add_argument('--image-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--snapshot-count', type=int, default=5)
    parser.add_argument('--models', type=str, nargs='+', default=['efficientnet_b3', 'convnext_tiny'])
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--cv-group-column', type=str, default='State')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--pseudo-label', action='store_true', help="Enable basic pseudo-labeling.")
    parser.add_argument('--confidence-threshold', type=float, default=0.9)
    parser.add_argument('--noisy-student', action='store_true', help="Enable Noisy Student training (implies --pseudo-label).")
    parser.add_argument('--student-dropout', type=float, default=0.5, help="Dropout for the student model in Noisy Student training.")
    parser.add_argument('--student-aug-policy', type=str, default='randaugment', help="Augmentation policy for the student model.")
    parser.add_argument('--tta', action='store_true', help="Enable Test-Time Augmentation for the final ensemble.")

    args = parser.parse_args()
    run_cv_ensemble(args)
