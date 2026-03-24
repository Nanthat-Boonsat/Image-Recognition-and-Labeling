import argparse
import os

def get_args():
    """Define and return all CLI options used for training."""
    parser = argparse.ArgumentParser(description="Model training options")

    # Backbone choice impacts speed vs accuracy.
    parser.add_argument('--backbone', type=str, default='fasterrcnn_resnet50_fpn', 
                        choices=['fasterrcnn_resnet50_fpn', 'fasterrcnn_mobilenet_v3'])
    
    parser.add_argument('--num_classes', type=int, default=1)

    default_csv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CSVs')
    default_out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    parser.add_argument('--csv_dir', type=str, default=default_csv_dir)
    # Output folder for model checkpoint, loss CSV, and learning curve figure.
    parser.add_argument('--outdir', type=str, default=default_out_dir)
    # Print progress every N batches.
    parser.add_argument('--log_every', type=int, default=1)

    # Core training hyperparameters.
    parser.add_argument('--batch_size', type=int, default=8, choices=[8, 16, 32, 64])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=1e-4)

    return parser.parse_args()