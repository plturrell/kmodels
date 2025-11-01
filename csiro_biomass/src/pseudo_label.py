import argparse
import pandas as pd
import torch
from pathlib import Path
from .training.lightning_module import BiomassLightningModule
from .data.dataset import create_inference_loader
from .train import run_prediction_loop

def create_pseudo_labels(args):
    device = torch.device(args.device)
    model = BiomassLightningModule.load_from_checkpoint(args.checkpoint_path, model=None, optimizer_cfg=None, target_names=None, huber_beta=None, train_sampler=None).to(device)

    test_df = pd.read_csv(args.test_csv)
    test_loader = create_inference_loader(
        test_df,
        image_dir=Path(args.image_dir),
        image_column=args.image_column,
        metadata_columns=[],
        metadata_mean=None,
        metadata_std=None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    _, predictions, _ = run_prediction_loop(model, test_loader, device)

    # This is a simple confidence threshold. More sophisticated methods could be used.
    confidence = predictions.max(dim=1).values
    high_confidence_mask = confidence > args.confidence_threshold

    pseudo_labeled_preds = predictions[high_confidence_mask]
    pseudo_labeled_ids = test_df[high_confidence_mask][args.id_column].values

    # Create a new dataframe for the pseudo-labels
    pseudo_df = pd.DataFrame(pseudo_labeled_preds.numpy(), columns=model.target_names)
    pseudo_df[args.id_column] = pseudo_labeled_ids

    # Combine with original training data
    train_df = pd.read_csv(args.train_csv)
    combined_df = pd.concat([train_df, pseudo_df], ignore_index=True)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)
    print(f'Saved combined training data with {len(pseudo_df)} pseudo-labels to {output_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create Pseudo-Labels')
    parser.add_argument('--checkpoint-path', type=str, required=True)
    parser.add_argument('--train-csv', type=str, required=True)
    parser.add_argument('--test-csv', type=str, required=True)
    parser.add_argument('--image-dir', type=str, required=True)
    parser.add_argument('--output-csv', type=str, required=True)
    parser.add_argument('--image-column', type=str, default='image_path')
    parser.add_argument('--id-column', type=str, default='sample_id')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--confidence-threshold', type=float, default=0.9)
    args = parser.parse_args()
    create_pseudo_labels(args)
