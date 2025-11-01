import argparse
import pandas as pd
import torch
from pathlib import Path
from .training.lightning_module import BiomassLightningModule
from .data.dataset import create_inference_loader
from .train import run_prediction_loop
import torch.nn.functional as F

def find_hard_samples(args):
    device = torch.device(args.device)
    model = BiomassLightningModule.load_from_checkpoint(args.checkpoint_path, model=None, optimizer_cfg=None, target_names=None, huber_beta=None, train_sampler=None).to(device)
    model.eval()

    train_df = pd.read_csv(args.train_csv)
    train_loader = create_inference_loader(
        train_df,
        image_dir=Path(args.image_dir),
        image_column=args.image_column,
        metadata_columns=[],
        metadata_mean=None,
        metadata_std=None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    identifiers, predictions, targets = run_prediction_loop(model, train_loader, device, return_targets=True)

    # Calculate per-sample loss
    losses = F.mse_loss(predictions, targets, reduction='none').mean(dim=1)

    loss_df = pd.DataFrame({
        'identifier': identifiers,
        'loss': losses.cpu().numpy()
    })

    # Merge with original dataframe to get file paths
    merged_df = pd.merge(loss_df, train_df, left_on='identifier', right_on=args.id_column)
    hardest_samples = merged_df.sort_values(by='loss', ascending=False).head(args.top_k)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hardest_samples.to_csv(output_path, index=False)
    print(f'Saved {len(hardest_samples)} hardest samples to {output_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find Hard Samples in Training Data')
    parser.add_argument('--checkpoint-path', type=str, required=True)
    parser.add_argument('--train-csv', type=str, required=True)
    parser.add_argument('--image-dir', type=str, required=True)
    parser.add_argument('--output-csv', type=str, required=True)
    parser.add_argument('--image-column', type=str, default='image_path')
    parser.add_argument('--id-column', type=str, default='sample_id')
    parser.add_argument('--top-k', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    find_hard_samples(args)
