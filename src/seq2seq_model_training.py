import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from tqdm import tqdm
import json
import argparse
from typing import Dict

from seq2seq_model import GestureLoss, create_model
from seq2seq_preprocessing import create_data_loaders, load_processed_data
from constants import PROCESSED_GESTURE_DATA_PATH, SEQ2SEQ_CONFIG


class GestureTrainer:
    """Complete training pipeline for gesture interpolation"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict,
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Training on device: {self.device}")

        # Setup optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        # Setup loss function
        self.criterion = GestureLoss(
            mse_weight=config.get("mse_weight", 1.0),
            smoothness_weight=config.get("smoothness_weight", 0.1),
            endpoint_weight=config.get("endpoint_weight", 2.0),
        )
        
        self.epochs = config["epochs"]

        # Setup logging
        self.setup_logging()

        # Training state
        self.best_val_loss = float("inf")
        self.train_losses = []
        self.val_losses = []

    def setup_logging(self):
        """Setup logging and checkpointing"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_name = f"gesture_interpolation_{timestamp}"
        self.log_dir = os.path.join("logs", self.exp_name)
        self.checkpoint_dir = os.path.join("checkpoints", self.exp_name)

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.writer = SummaryWriter(self.log_dir)

        # Save config
        with open(os.path.join(self.log_dir, "config.json"), "w") as f:
            json.dump(self.config, f, indent=2)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {"total": [], "mse": [], "smoothness": [], "endpoint": []}

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            source = batch["source"].to(self.device)
            target = batch["target"].to(self.device)
            ground_truth = batch["ground_truth"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(
                source,
                target,
                ground_truth.size(1),
                teacher_forcing_ratio=self.config.get("teacher_forcing_ratio", 0.5),
            )

            # Calculate loss with endpoint information
            source_end = source[:, -1:, :]  # Last frame of source
            target_start = target[:, :1, :]  # First frame of target

            loss_dict = self.criterion(output, ground_truth, source_end, target_start)

            # Backward pass
            loss_dict["total_loss"].backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.get("grad_clip", 1.0)
            )

            self.optimizer.step()

            # Record losses
            for key in epoch_losses.keys():
                if key == "total":
                    epoch_losses[key].append(loss_dict["total_loss"].item())
                else:
                    epoch_losses[key].append(loss_dict[f"{key}_loss"].item())

            # Update progress bar
            pbar.set_postfix(
                {
                    "Loss": f"{loss_dict['total_loss'].item():.4f}",
                    "MSE": f"{loss_dict['mse_loss'].item():.4f}",
                    "LR": f"{self.optimizer.param_groups[0]['lr']:.6f}",
                }
            )

            # Log to tensorboard every N steps
            if batch_idx % self.config.get("log_every", 100) == 0:
                step = epoch * len(self.train_loader) + batch_idx
                for key, value in loss_dict.items():
                    if isinstance(value, torch.Tensor):
                        self.writer.add_scalar(f"Train/{key}", value.item(), step)

        # Calculate epoch averages
        epoch_avg_losses = {
            key: np.mean(values) for key, values in epoch_losses.items()
        }
        return epoch_avg_losses

    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        epoch_losses = {"total": [], "mse": [], "smoothness": [], "endpoint": []}

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Validation"):
                # Move data to device
                source = batch["source"].to(self.device)
                target = batch["target"].to(self.device)
                ground_truth = batch["ground_truth"].to(self.device)

                # Forward pass (no teacher forcing during validation)
                output = self.model.generate(source, target, ground_truth.size(1))

                # Calculate loss
                source_end = source[:, -1:, :]
                target_start = target[:, :1, :]
                loss_dict = self.criterion(
                    output, ground_truth, source_end, target_start
                )

                # Record losses
                for key in epoch_losses.keys():
                    if key == "total":
                        epoch_losses[key].append(loss_dict["total_loss"].item())
                    else:
                        epoch_losses[key].append(loss_dict[f"{key}_loss"].item())

        # Calculate epoch averages
        epoch_avg_losses = {
            key: np.mean(values) for key, values in epoch_losses.items()
        }

        # Log to tensorboard
        for key, value in epoch_avg_losses.items():
            self.writer.add_scalar(f"Val/{key}", value, epoch)

        return epoch_avg_losses

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "config": self.config,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }

        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"
        )
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"New best model saved with validation loss: {val_loss:.6f}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.best_val_loss = checkpoint.get("val_loss", float("inf"))

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint["epoch"]

    def visualize_sample_predictions(self, epoch: int, num_samples: int = 3):
        """Visualize sample predictions for debugging"""
        self.model.eval()

        with torch.no_grad():
            # Get a batch from validation set
            batch = next(iter(self.test_loader))
            source = batch["source"][:num_samples].to(self.device)
            target = batch["target"][:num_samples].to(self.device)
            ground_truth = batch["ground_truth"][:num_samples].to(self.device)

            # Generate predictions
            predictions = self.model.generate(source, target, ground_truth.size(1))

            # Move to CPU for plotting
            source = source.cpu().numpy()
            target = target.cpu().numpy()
            ground_truth = ground_truth.cpu().numpy()
            predictions = predictions.cpu().numpy()

            # Plot first few features for visualization
            fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 3))
            if num_samples == 1:
                axes = axes.reshape(1, -1)

            for i in range(num_samples):
                for j in range(3):  # Plot first 3 features
                    ax = axes[i, j]

                    # Plot ground truth
                    ax.plot(
                        ground_truth[i, :, j], "b-", label="Ground Truth", linewidth=2
                    )

                    # Plot prediction
                    ax.plot(
                        predictions[i, :, j], "r--", label="Prediction", linewidth=2
                    )

                    # Add source/target endpoints for reference
                    ax.axhline(
                        y=source[i, -1, j],
                        color="g",
                        linestyle=":",
                        alpha=0.7,
                        label="Source End",
                    )
                    ax.axhline(
                        y=target[i, 0, j],
                        color="orange",
                        linestyle=":",
                        alpha=0.7,
                        label="Target Start",
                    )

                    ax.set_title(f"Sample {i+1}, Feature {j+1}")
                    ax.legend()
                    ax.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save plot
            plot_path = os.path.join(self.log_dir, f"predictions_epoch_{epoch}.png")
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()

            # Log to tensorboard
            self.writer.add_figure("Predictions", fig, epoch)

    def plot_training_curves(self):
        """Plot and save training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        epochs = range(1, len(self.train_losses) + 1)

        # Total loss
        axes[0, 0].plot(
            epochs, [loss["total"] for loss in self.train_losses], "b-", label="Train"
        )
        axes[0, 0].plot(
            epochs, [loss["total"] for loss in self.val_losses], "r-", label="Val"
        )
        axes[0, 0].set_title("Total Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # MSE loss
        axes[0, 1].plot(
            epochs, [loss["mse"] for loss in self.train_losses], "b-", label="Train"
        )
        axes[0, 1].plot(
            epochs, [loss["mse"] for loss in self.val_losses], "r-", label="Val"
        )
        axes[0, 1].set_title("MSE Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Smoothness loss
        axes[1, 0].plot(
            epochs,
            [loss["smoothness"] for loss in self.train_losses],
            "b-",
            label="Train",
        )
        axes[1, 0].plot(
            epochs, [loss["smoothness"] for loss in self.val_losses], "r-", label="Val"
        )
        axes[1, 0].set_title("Smoothness Loss")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Endpoint loss
        axes[1, 1].plot(
            epochs,
            [loss["endpoint"] for loss in self.train_losses],
            "b-",
            label="Train",
        )
        axes[1, 1].plot(
            epochs, [loss["endpoint"] for loss in self.val_losses], "r-", label="Val"
        )
        axes[1, 1].set_title("Endpoint Loss")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(self.log_dir, "training_curves.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

    def train(self, resume_from: str = None):
        """Complete training loop"""
        start_epoch = 0

        if resume_from:
            start_epoch = self.load_checkpoint(resume_from) + 1

        print(f"Starting training for {self.epochs} epochs...")
        print(f"Model has {sum(p.numel() for p in self.model.parameters())} parameters")

        for epoch in range(start_epoch, self.epochs):
            print(f"\nEpoch {epoch}/{self.epochs-1}")
            print("-" * 50)

            # Training phase
            train_losses = self.train_epoch(epoch)
            self.train_losses.append(train_losses)

            # Validation phase
            val_losses = self.validate_epoch(epoch)
            self.val_losses.append(val_losses)

            # Learning rate scheduling
            self.scheduler.step(val_losses["total"])

            # Print epoch summary
            print(
                f"Train Loss: {train_losses['total']:.6f} | Val Loss: {val_losses['total']:.6f}"
            )
            print(
                f"MSE: {val_losses['mse']:.6f} | Smoothness: {val_losses['smoothness']:.6f} | Endpoint: {val_losses['endpoint']:.6f}"
            )

            # Save checkpoint
            is_best = val_losses["total"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses["total"]

            self.save_checkpoint(epoch, val_losses["total"], is_best)

            # Visualize predictions periodically
            if epoch % self.config.get("vis_every", 10) == 0:
                self.visualize_sample_predictions(epoch)

            # Early stopping
            if self.config.get("early_stopping", False):
                patience = self.config.get("patience", 20)
                if len(self.val_losses) > patience:
                    recent_losses = [
                        loss["total"] for loss in self.val_losses[-patience:]
                    ]
                    if all(loss >= self.best_val_loss for loss in recent_losses):
                        print(f"Early stopping triggered after epoch {epoch}")
                        break

        # Final visualization
        self.plot_training_curves()
        self.writer.close()

        print(f"\nTraining completed! Best validation loss: {self.best_val_loss:.6f}")
        print(f"Logs and checkpoints saved to: {self.log_dir}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train Gesture Interpolation Model")
    parser.add_argument(
        "--data_path",
        type=str,
        default=PROCESSED_GESTURE_DATA_PATH,
        help="Path to processed gesture data (.pkl file)",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config JSON file"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")

    args = parser.parse_args()

    # Default configuration
    config = SEQ2SEQ_CONFIG

    # Load custom config if provided
    if args.config:
        with open(args.config, "r") as f:
            custom_config = json.load(f)
            config.update(custom_config)

    # Load processed data
    print("Loading processed data...")
    processed_data = load_processed_data(args.data_path)

    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        processed_data, config["batch_size"]
    )

    # Create model
    print("Creating model...")
    feature_dim = processed_data["feature_dim"]
    model = create_model(feature_dim, config)

    # Initialize trainer
    trainer = GestureTrainer(model, train_loader, test_loader, config)

    # Start training
    trainer.train(args.resume)


if __name__ == "__main__":
    main()

# Example usage without command line:
"""
# Load your processed data
processed_data = load_processed_data('gesture_data_processed.pkl')

# Create data loaders
train_loader, test_loader = create_data_loaders(processed_data, batch_size=16)

# Configuration
config = {
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'batch_size': 16,
    'hidden_dim': 256,
    'num_layers': 3,
    'use_transformer': True,
    'dropout': 0.1,
    'mse_weight': 1.0,
    'smoothness_weight': 0.1,
    'endpoint_weight': 2.0,
    'teacher_forcing_ratio': 0.5,
    'grad_clip': 1.0,
    'log_every': 50,
    'vis_every': 5,
    'early_stopping': True,
    'patience': 15
}

# Create model
model = create_model(processed_data['feature_dim'], config)

# Initialize trainer
trainer = GestureTrainer(model, train_loader, test_loader, config)

# Train
trainer.train(num_epochs=50)
"""
