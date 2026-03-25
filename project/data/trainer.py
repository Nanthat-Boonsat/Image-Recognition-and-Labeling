from args import get_args
import os
import torch
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt


def save_learning_curve(train_losses, val_losses, outdir):
    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label='Training Loss', marker='o', linewidth=2)
    plt.plot(epochs, val_losses, label='Validation Loss', marker='o', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'learning_curve.png'))
    plt.close()

    history_df = pd.DataFrame(
        {
            'epoch': epochs,
            'train_loss': train_losses,
            'val_loss': val_losses,
        }
    )
    history_df.to_csv(os.path.join(outdir, 'loss_history.csv'), index=False)

def train_model(model, train_loader, val_loader, device):
    args = get_args()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    outdir = os.path.abspath(args.outdir)
    log_every = max(1, args.log_every)
    os.makedirs(outdir, exist_ok=True)

    print(f"Training on device: {device}", flush=True)
    print(f"Output directory: {outdir}", flush=True)
    print(f"Train batches per epoch: {len(train_loader)}", flush=True)
    print(f"Validation batches per epoch: {len(val_loader)}", flush=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        print(f"Starting epoch {epoch + 1}/{args.epochs}...", flush=True)
        
        for batch_idx, (images, targets) in enumerate(train_loader, start=1):
            images = [image.to(device=device, dtype=torch.float32) for image in images]
            targets = [
                {
                    'boxes': target['boxes'].to(device=device, dtype=torch.float32),
                    'labels': target['labels'].to(device=device, dtype=torch.int64)
                }
                for target in targets
            ]
            
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            loss = sum(loss_value for loss_value in loss_dict.values())
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * len(images)

    
            if batch_idx % log_every == 0 or batch_idx == len(train_loader):
                print(
                    f"Epoch {epoch + 1}/{args.epochs} | Batch {batch_idx}/{len(train_loader)} | "
                    f"Batch Loss: {loss.item():.4f}",
                    flush=True,
                )
            
        train_epoch_loss = running_loss / len(train_loader.dataset)
        
        val_loss = validate_model(model, val_loader, device)
        train_losses.append(train_epoch_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(outdir, 'best_model.pth'))
            
        print(f"Epoch {epoch + 1}/{args.epochs} | "
              f"Train Loss: {train_epoch_loss:.4f} | "
              f"Val Loss: {val_loss:.4f}")

    save_learning_curve(train_losses, val_losses, outdir)
    print(f"Saved learning curve to {os.path.join(outdir, 'learning_curve.png')}")
    print(f"Saved loss history to {os.path.join(outdir, 'loss_history.csv')}")


def validate_model(model, val_loader, device):
    model.train()
    val_loss_sum = 0.0
    val_count = 0
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = [image.to(device=device, dtype=torch.float32) for image in images]
            targets = [
                {
                    'boxes': target['boxes'].to(device=device, dtype=torch.float32),
                    'labels': target['labels'].to(device=device, dtype=torch.int64)
                }
                for target in targets
            ]
            
            loss_dict = model(images, targets)
            loss = sum(loss_value for loss_value in loss_dict.values())
            
            val_loss_sum += loss.item() * len(images)
            val_count += len(images)
            
    val_epoch_loss = val_loss_sum / val_count
    return val_epoch_loss