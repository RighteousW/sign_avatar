"""
Gloss-to-Text Translation Using Sequence-to-Sequence Attention Models

This implementation is based on:

PRIMARY PAPER:
    Arvanitis, N., Constantinopoulos, C., & Kosmopoulos, D. (2019).
    "Translation of Sign Language Glosses to Text Using Sequence-to-Sequence Attention Models"
    2019 15th International Conference on Signal-Image Technology & Internet-Based Systems (SITIS)
    DOI: 10.1109/SITIS.2019.00056

REFERENCE PAPERS:
    - Zhou, F., & Van de Cruys, T. (2025). "Non-autoregressive Modeling for
      Sign-gloss to Texts Translation." Proceedings of Machine Translation Summit XX.
    - Luong, M.-T., Pham, H., & Manning, C. (2015). "Effective Approaches to
      Attention-based Neural Machine Translation." EMNLP 2015.
    - Bahdanau, D., Cho, K., & Bengio, Y. (2014). "Neural Machine Translation by
      Jointly Learning to Align and Translate." arXiv:1409.0473v7

DATASETS:
    - Othman, A., & Jemni, M. (2012). "English-ASL Gloss Parallel Corpus 2012: ASLG-PC12"
      5th Workshop on the Representation and Processing of Sign Languages LREC12
    - https://huggingface.co/datasets/agentlans/high-quality-english-sentences/tree/main

EVALUATION:
    Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2002). "BLEU: a method for
    automatic evaluation of machine translation." ACL 2002.

For full citations and references, see the accompanying CITATIONS.md file.

Implementation by: [Your Name/Organization]
Date: 2025
License: [Your License]
"""

import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter, defaultdict
import random
import json
import os
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
import sys

from tqdm import tqdm

from ..constants import LOGS_DIR


class LuongAttention(nn.Module):
    """
    Implements three types of Luong attention:
    - dot: h_t^T * h_s
    - general: h_t^T * W_a * h_s
    - concat: v_a^T * tanh(W_a * [h_t; h_s])
    """

    def __init__(self, hidden_size, attention_type="dot"):
        super(LuongAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_type = attention_type

        if attention_type == "general":
            self.W_a = nn.Linear(hidden_size, hidden_size, bias=False)
        elif attention_type == "concat":
            self.W_a = nn.Linear(hidden_size * 2, hidden_size, bias=False)
            self.v_a = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        """
        Args:
            decoder_hidden: (batch_size, hidden_size)
            encoder_outputs: (batch_size, seq_len, hidden_size)
        Returns:
            context: (batch_size, hidden_size)
            attention_weights: (batch_size, seq_len)
        """
        seq_len = encoder_outputs.size(1)

        # Calculate attention scores
        if self.attention_type == "dot":
            # h_t^T * h_s
            scores = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2))
            scores = scores.squeeze(2)  # (batch_size, seq_len)

        elif self.attention_type == "general":
            # h_t^T * W_a * h_s
            energy = self.W_a(encoder_outputs)  # (batch_size, seq_len, hidden_size)
            scores = torch.bmm(energy, decoder_hidden.unsqueeze(2))
            scores = scores.squeeze(2)  # (batch_size, seq_len)

        elif self.attention_type == "concat":
            # v_a^T * tanh(W_a * [h_t; h_s])
            decoder_hidden_expanded = decoder_hidden.unsqueeze(1).expand(
                -1, seq_len, -1
            )
            concat = torch.cat([decoder_hidden_expanded, encoder_outputs], dim=2)
            energy = torch.tanh(self.W_a(concat))
            scores = self.v_a(energy).squeeze(2)  # (batch_size, seq_len)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=1)  # (batch_size, seq_len)

        # Calculate context vector as weighted sum
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)  # (batch_size, hidden_size)

        return context, attention_weights


class GRUEncoder(nn.Module):
    def __init__(
        self, input_size, embedding_size, hidden_size, num_layers=2, dropout=0.25
    ):
        super(GRUEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru = nn.GRU(
            embedding_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_seq):
        """
        Args:
            input_seq: (batch_size, seq_len)
        Returns:
            outputs: (batch_size, seq_len, hidden_size)
            hidden: (num_layers, batch_size, hidden_size)
        """
        embedded = self.dropout(self.embedding(input_seq))
        outputs, hidden = self.gru(embedded)
        return outputs, hidden


class GRUDecoderWithAttention(nn.Module):
    def __init__(
        self,
        output_size,
        embedding_size,
        hidden_size,
        num_layers=2,
        dropout=0.25,
        attention_type="dot",
    ):
        super(GRUDecoderWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(output_size, embedding_size)
        self.attention = LuongAttention(hidden_size, attention_type)
        self.gru = nn.GRU(
            embedding_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.concat_layer = nn.Linear(hidden_size * 2, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, encoder_outputs):
        """
        Args:
            input_token: (batch_size, 1)
            hidden: (num_layers, batch_size, hidden_size)
            encoder_outputs: (batch_size, seq_len, hidden_size)
        Returns:
            output: (batch_size, output_size)
            hidden: (num_layers, batch_size, hidden_size)
            attention_weights: (batch_size, seq_len)
        """
        # Embed input token
        embedded = self.dropout(
            self.embedding(input_token)
        )  # (batch_size, 1, embedding_size)

        # Pass through GRU
        gru_output, hidden = self.gru(
            embedded, hidden
        )  # gru_output: (batch_size, 1, hidden_size)
        gru_output = gru_output.squeeze(1)  # (batch_size, hidden_size)

        # Calculate attention
        context, attention_weights = self.attention(gru_output, encoder_outputs)

        # Combine context and GRU output
        concat_input = torch.cat(
            [gru_output, context], dim=1
        )  # (batch_size, hidden_size * 2)
        concat_output = torch.tanh(
            self.concat_layer(concat_input)
        )  # (batch_size, hidden_size)

        # Generate output
        output = self.output_layer(concat_output)  # (batch_size, output_size)

        return output, hidden, attention_weights


class Seq2SeqWithAttention(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2SeqWithAttention, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        Args:
            src: (batch_size, src_len)
            trg: (batch_size, trg_len)
            teacher_forcing_ratio: probability of using teacher forcing
        Returns:
            outputs: (batch_size, trg_len, output_size)
        """
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.output_size

        # Store outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        # Encode source
        encoder_outputs, hidden = self.encoder(src)

        # First input to decoder is SOS token
        input_token = trg[:, 0].unsqueeze(1)  # (batch_size, 1)

        for t in range(1, trg_len):
            # Decode
            output, hidden, attention_weights = self.decoder(
                input_token, hidden, encoder_outputs
            )
            outputs[:, t, :] = output

            # Teacher forcing
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1).unsqueeze(1)
            input_token = trg[:, t].unsqueeze(1) if use_teacher_forcing else top1

        return outputs


class GlossTextDataset(Dataset):
    def __init__(self, gloss_sequences, text_sequences, gloss_vocab, text_vocab):
        self.gloss_sequences = gloss_sequences
        self.text_sequences = text_sequences
        self.gloss_vocab = gloss_vocab
        self.text_vocab = text_vocab

    def __len__(self):
        return len(self.gloss_sequences)

    def __getitem__(self, idx):
        gloss = self.gloss_sequences[idx]
        text = self.text_sequences[idx]

        # Convert to indices
        gloss_indices = [
            self.gloss_vocab.get(g, self.gloss_vocab["<UNK>"]) for g in gloss
        ]
        text_indices = (
            [self.text_vocab["<SOS>"]]
            + [self.text_vocab.get(t, self.text_vocab["<UNK>"]) for t in text]
            + [self.text_vocab["<EOS>"]]
        )

        return torch.LongTensor(gloss_indices), torch.LongTensor(text_indices)


class TrainingLogger:
    """Comprehensive training logger for research documentation"""

    def __init__(self, experiment_name, log_dir=LOGS_DIR):
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(log_dir, f"{experiment_name}_{self.timestamp}")

        # Create directories
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "checkpoints"), exist_ok=True)

        # Initialize metrics storage
        self.metrics = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "bleu_1": [],
            "bleu_2": [],
            "bleu_3": [],
            "bleu_4": [],
            "learning_rate": [],
            "epoch_time": [],
        }

        # Error analysis
        self.error_analysis = defaultdict(list)

        print(f"Logging to: {self.run_dir}")

    def log_epoch(self, epoch, train_loss, val_loss, bleu_scores, lr, epoch_time):
        """Log metrics for one epoch"""
        self.metrics["epoch"].append(epoch)
        self.metrics["train_loss"].append(train_loss)
        self.metrics["val_loss"].append(val_loss)
        self.metrics["bleu_1"].append(bleu_scores.get("BLEU-1", 0))
        self.metrics["bleu_2"].append(bleu_scores.get("BLEU-2", 0))
        self.metrics["bleu_3"].append(bleu_scores.get("BLEU-3", 0))
        self.metrics["bleu_4"].append(bleu_scores.get("BLEU-4", 0))
        self.metrics["learning_rate"].append(lr)
        self.metrics["epoch_time"].append(epoch_time)

    def log_error(self, gloss_input, predicted, reference, error_type="mismatch"):
        """Log translation errors for analysis"""
        self.error_analysis[error_type].append(
            {
                "gloss": (
                    " ".join(gloss_input)
                    if isinstance(gloss_input, list)
                    else gloss_input
                ),
                "predicted": predicted,
                "reference": reference,
            }
        )

    def save_metrics_csv(self):
        """Save metrics to CSV for easy analysis"""
        df = pd.DataFrame(self.metrics)
        csv_path = os.path.join(self.run_dir, "training_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"Metrics saved to {csv_path}")
        return csv_path

    def save_metrics_json(self):
        """Save complete metrics including error analysis to JSON"""
        full_data = {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "metrics": self.metrics,
            "error_analysis": dict(self.error_analysis),
        }
        json_path = os.path.join(self.run_dir, "training_log.json")
        with open(json_path, "w") as f:
            json.dump(full_data, f, indent=2)
        print(f"Full log saved to {json_path}")
        return json_path

    def plot_learning_curves(self):
        """Generate learning curve plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Loss curves
        ax = axes[0, 0]
        ax.plot(
            self.metrics["epoch"],
            self.metrics["train_loss"],
            label="Train Loss",
            marker="o",
            linewidth=2,
        )
        ax.plot(
            self.metrics["epoch"],
            self.metrics["val_loss"],
            label="Val Loss",
            marker="s",
            linewidth=2,
        )
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # BLEU scores
        ax = axes[0, 1]
        ax.plot(
            self.metrics["epoch"],
            self.metrics["bleu_1"],
            label="BLEU-1",
            marker="o",
            linewidth=2,
        )
        ax.plot(
            self.metrics["epoch"],
            self.metrics["bleu_2"],
            label="BLEU-2",
            marker="s",
            linewidth=2,
        )
        ax.plot(
            self.metrics["epoch"],
            self.metrics["bleu_3"],
            label="BLEU-3",
            marker="^",
            linewidth=2,
        )
        ax.plot(
            self.metrics["epoch"],
            self.metrics["bleu_4"],
            label="BLEU-4",
            marker="d",
            linewidth=2,
        )
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("BLEU Score", fontsize=12)
        ax.set_title("BLEU Score Progress", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Learning rate
        ax = axes[1, 0]
        ax.plot(
            self.metrics["epoch"],
            self.metrics["learning_rate"],
            marker="o",
            linewidth=2,
            color="green",
        )
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Learning Rate", fontsize=12)
        ax.set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        # Epoch time
        ax = axes[1, 1]
        ax.bar(self.metrics["epoch"], self.metrics["epoch_time"], color="coral")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Time (seconds)", fontsize=12)
        ax.set_title("Training Time per Epoch", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plot_path = os.path.join(self.run_dir, "plots", "learning_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Learning curves saved to {plot_path}")
        plt.close()

        return plot_path

    def plot_bleu_comparison(self):
        """Create detailed BLEU score comparison plot"""
        fig, ax = plt.subplots(figsize=(12, 6))

        x = self.metrics["epoch"]
        width = 0.2

        positions_1 = [i - 1.5 * width for i in x]
        positions_2 = [i - 0.5 * width for i in x]
        positions_3 = [i + 0.5 * width for i in x]
        positions_4 = [i + 1.5 * width for i in x]

        ax.bar(positions_1, self.metrics["bleu_1"], width, label="BLEU-1", alpha=0.8)
        ax.bar(positions_2, self.metrics["bleu_2"], width, label="BLEU-2", alpha=0.8)
        ax.bar(positions_3, self.metrics["bleu_3"], width, label="BLEU-3", alpha=0.8)
        ax.bar(positions_4, self.metrics["bleu_4"], width, label="BLEU-4", alpha=0.8)

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("BLEU Score", fontsize=12)
        ax.set_title("BLEU Scores Across All Epochs", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xticks(x)

        plot_path = os.path.join(self.run_dir, "plots", "bleu_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"BLEU comparison saved to {plot_path}")
        plt.close()

        return plot_path

    def generate_summary_report(self, config, final_results):
        """Generate a comprehensive summary report"""
        report_path = os.path.join(self.run_dir, "experiment_report.txt")

        with open(report_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write(f"EXPERIMENT REPORT: {self.experiment_name}\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write("=" * 70 + "\n\n")

            # Configuration
            f.write("MODEL CONFIGURATION\n")
            f.write("-" * 70 + "\n")
            for key, value in config.items():
                f.write(f"{key:25s}: {value}\n")
            f.write("\n")

            # Training summary
            f.write("TRAINING SUMMARY\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total epochs: {len(self.metrics['epoch'])}\n")
            f.write(
                f"Best train loss: {min(self.metrics['train_loss']):.4f} (Epoch {self.metrics['train_loss'].index(min(self.metrics['train_loss'])) + 1})\n"
            )
            f.write(
                f"Best val loss: {min(self.metrics['val_loss']):.4f} (Epoch {self.metrics['val_loss'].index(min(self.metrics['val_loss'])) + 1})\n"
            )
            f.write(
                f"Best BLEU-4: {max(self.metrics['bleu_4']):.4f} (Epoch {self.metrics['bleu_4'].index(max(self.metrics['bleu_4'])) + 1})\n"
            )
            f.write(
                f"Total training time: {sum(self.metrics['epoch_time']):.1f}s ({sum(self.metrics['epoch_time'])/60:.1f} min)\n"
            )
            f.write(f"Average epoch time: {np.mean(self.metrics['epoch_time']):.1f}s\n")
            f.write("\n")

            # Final results
            f.write("FINAL EVALUATION\n")
            f.write("-" * 70 + "\n")
            for key, value in final_results.items():
                if isinstance(value, dict):
                    f.write(f"{key}:\n")
                    for k, v in value.items():
                        if isinstance(v, float):
                            f.write(f"  {k}: {v:.4f} ({v*100:.2f}%)\n")
                        else:
                            f.write(f"  {k}: {v}\n")
                else:
                    f.write(f"{key}: {value}\n")
            f.write("\n")

            # Error analysis summary
            if self.error_analysis:
                f.write("ERROR ANALYSIS\n")
                f.write("-" * 70 + "\n")
                for error_type, errors in self.error_analysis.items():
                    f.write(f"{error_type}: {len(errors)} instances\n")
                f.write("\n")

            # System info
            f.write("SYSTEM INFORMATION\n")
            f.write("-" * 70 + "\n")
            f.write(f"PyTorch version: {torch.__version__}\n")
            f.write(f"CUDA available: {torch.cuda.is_available()}\n")
            if torch.cuda.is_available():
                f.write(f"CUDA version: {torch.version.cuda}\n")
                f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
            f.write(f"Python version: {sys.version}\n")
            f.write("\n")

        print(f"Summary report saved to {report_path}")
        return report_path


class EarlyStopping:
    """Early stopping based on BLEU score plateau"""

    def __init__(self, patience=3, min_delta=0.002):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_bleu = 0

    def should_stop(self, bleu_score):
        if bleu_score > self.best_bleu + self.min_delta:
            self.best_bleu = bleu_score
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


def create_confusion_matrix(predictions, references, top_n=100):
    """
    Create a confusion matrix for the most common prediction errors.

    Args:
        predictions: List of predicted word sequences
        references: List of reference word sequences
        vocab: Text vocabulary
        top_n: Number of most common words to include

    Returns:
        Confusion matrix as numpy array and word labels
    """
    # Get most common words
    word_counts = Counter()
    for ref in references:
        word_counts.update(ref)

    common_words = [word for word, _ in word_counts.most_common(top_n)]
    word_to_idx = {word: idx for idx, word in enumerate(common_words)}

    # Initialize confusion matrix
    n = len(common_words)
    conf_matrix = np.zeros((n, n))

    # Fill confusion matrix
    for pred_seq, ref_seq in zip(predictions, references):
        # Align sequences (simple word-by-word alignment)
        max_len = max(len(pred_seq), len(ref_seq))
        for i in range(max_len):
            true_word = ref_seq[i] if i < len(ref_seq) else None
            pred_word = pred_seq[i] if i < len(pred_seq) else None

            if true_word in word_to_idx and pred_word in word_to_idx:
                conf_matrix[word_to_idx[true_word], word_to_idx[pred_word]] += 1

    return conf_matrix, common_words


def plot_confusion_matrix(conf_matrix, labels, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(14, 12))

    # Normalize by row (true labels)
    conf_matrix_norm = conf_matrix / (conf_matrix.sum(axis=1, keepdims=True) + 1e-10)

    sns.heatmap(
        conf_matrix_norm,
        xticklabels=labels,
        yticklabels=labels,
        annot=False,
        fmt=".2f",
        cmap="YlOrRd",
        cbar_kws={"label": "Proportion"},
    )

    plt.xlabel("Predicted Word", fontsize=12)
    plt.ylabel("True Word", fontsize=12)
    plt.title("Confusion Matrix (Top 20 Words)", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def analyze_sequence_lengths(gloss_sequences, text_sequences, save_dir):
    """Analyze and visualize sequence length distributions"""
    gloss_lengths = [len(seq) for seq in gloss_sequences]
    text_lengths = [len(seq) for seq in text_sequences]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Gloss lengths
    axes[0].hist(gloss_lengths, bins=50, color="skyblue", edgecolor="black", alpha=0.7)
    axes[0].axvline(
        np.mean(gloss_lengths),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(gloss_lengths):.1f}",
    )
    axes[0].axvline(
        np.median(gloss_lengths),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {np.median(gloss_lengths):.1f}",
    )
    axes[0].set_xlabel("Sequence Length", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].set_title("Gloss Sequence Lengths", fontsize=14, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Text lengths
    axes[1].hist(
        text_lengths, bins=50, color="lightcoral", edgecolor="black", alpha=0.7
    )
    axes[1].axvline(
        np.mean(text_lengths),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(text_lengths):.1f}",
    )
    axes[1].axvline(
        np.median(text_lengths),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {np.median(text_lengths):.1f}",
    )
    axes[1].set_xlabel("Sequence Length", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_title("Text Sequence Lengths", fontsize=14, fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "sequence_length_distribution.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Sequence length analysis saved to {plot_path}")
    plt.close()

    # Return statistics
    stats = {
        "gloss": {
            "min": min(gloss_lengths),
            "max": max(gloss_lengths),
            "mean": np.mean(gloss_lengths),
            "median": np.median(gloss_lengths),
            "std": np.std(gloss_lengths),
        },
        "text": {
            "min": min(text_lengths),
            "max": max(text_lengths),
            "mean": np.mean(text_lengths),
            "median": np.median(text_lengths),
            "std": np.std(text_lengths),
        },
    }

    return stats


def evaluate_model_detailed(
    model, dataloader, gloss_vocab, text_vocab, device, logger=None
):
    """
    Enhanced evaluation with error logging and detailed metrics.
    """
    model.eval()
    predictions = []
    references = []
    gloss_inputs = []

    idx_to_text = {v: k for k, v in text_vocab.items()}
    idx_to_gloss = {v: k for k, v in gloss_vocab.items()}

    with torch.no_grad():
        eval_pbar = tqdm(
            dataloader,
            desc="Evaluating BLEU",
            leave=False,
            bar_format="{l_bar}{bar:30}{r_bar}",
        )
        for src, trg in eval_pbar:
            batch_size = src.size(0)
            src = src.to(device)

            # Encode
            encoder_outputs, hidden = model.encoder(src)

            # Decode each sentence in batch
            for b in range(batch_size):
                trg_indices = [text_vocab["<SOS>"]]
                src_single = src[b : b + 1]
                enc_out_single = encoder_outputs[b : b + 1]
                hid_single = hidden[:, b : b + 1, :]

                # Get gloss input for error analysis
                gloss_input = [
                    idx_to_gloss.get(idx.item(), "<UNK>")
                    for idx in src[b]
                    if idx.item() != gloss_vocab["<PAD>"]
                ]

                # Use beam search for better predictions
                trg_indices = beam_search_decode(
                    model, src_single, text_vocab, device, beam_width=5, max_len=50
                )

                # Convert to words
                pred_words = [
                    idx_to_text[idx]
                    for idx in trg_indices
                    if idx in idx_to_text
                    and idx_to_text[idx] not in ["<PAD>", "<SOS>", "<EOS>"]
                ]
                ref_words = [
                    idx_to_text[idx.item()]
                    for idx in trg[b]
                    if idx.item() in idx_to_text
                    and idx_to_text[idx.item()] not in ["<PAD>", "<SOS>", "<EOS>"]
                ]

                if pred_words and ref_words:
                    predictions.append(pred_words)
                    references.append(ref_words)
                    gloss_inputs.append(gloss_input)

                    # Log errors for analysis
                    if logger and pred_words != ref_words:
                        logger.log_error(
                            gloss_input, " ".join(pred_words), " ".join(ref_words)
                        )
        eval_pbar.close()

    # Calculate BLEU scores
    bleu_scores = calculate_bleu(predictions, references)

    # Get example translations
    examples = []
    for i in range(min(10, len(predictions))):
        examples.append(
            {
                "gloss": " ".join(gloss_inputs[i]),
                "prediction": " ".join(predictions[i]),
                "reference": " ".join(references[i]),
                "match": predictions[i] == references[i],
            }
        )

    return {
        "bleu_scores": bleu_scores,
        "examples": examples,
        "num_samples": len(predictions),
        "predictions": predictions,
        "references": references,
        "gloss_inputs": gloss_inputs,
    }


def train_with_logging(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    criterion,
    scheduler,
    gloss_vocab,
    text_vocab,
    config,
    device,
    experiment_name="gloss2text_experiment",
):
    """Enhanced training loop with comprehensive logging"""

    # Initialize logger
    logger = TrainingLogger(experiment_name)

    # Log initial dataset statistics
    print("\nAnalyzing dataset statistics...")
    train_gloss = train_dataloader.dataset.gloss_sequences
    train_text = train_dataloader.dataset.text_sequences

    seq_stats = analyze_sequence_lengths(
        train_gloss, train_text, os.path.join(logger.run_dir, "plots")
    )

    # Save vocabulary info
    vocab_info = {
        "gloss_vocab_size": len(gloss_vocab),
        "text_vocab_size": len(text_vocab),
        "sequence_stats": seq_stats,
    }
    with open(os.path.join(logger.run_dir, "vocabulary_info.json"), "w") as f:
        json.dump(vocab_info, f, indent=2)

    # Training loop
    best_loss = float("inf")
    best_bleu = 0.0
    train_start_time = time.time()
    early_stopping = EarlyStopping(patience=3, min_delta=0.002)

    print(f"\nStarting training for {config['num_epochs']} epochs...")
    print("=" * 60)

    epoch_pbar = tqdm(
        range(config["num_epochs"]),
        desc="Overall Progress",
        bar_format="{l_bar}{bar:30}{r_bar}",
    )

    for epoch in epoch_pbar:
        epoch_start_time = time.time()

        # Training
        tf_ratio = get_teacher_forcing_ratio(
            epoch, config["num_epochs"], start_ratio=1.0, end_ratio=0.5
        )

        print(f"Epoch {epoch+1}: Teacher forcing ratio = {tf_ratio:.3f}")

        train_loss = train_epoch(
            model,
            train_dataloader,
            optimizer,
            criterion,
            device,
            teacher_forcing_ratio=tf_ratio,
            use_amp=False,
        )

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            val_pbar = tqdm(
                val_dataloader,
                desc="Validation",
                leave=False,
                bar_format="{l_bar}{bar:30}{r_bar}",
            )
            for src, trg in val_pbar:
                src, trg = src.to(device), trg.to(device)
                output = model(src, trg, teacher_forcing_ratio=0)
                output_dim = output.shape[-1]
                output = output[:, 1:].reshape(-1, output_dim)
                trg_flat = trg[:, 1:].reshape(-1)
                loss = criterion(output, trg_flat)
                val_loss += loss.item()
                val_pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})
            val_pbar.close()

        val_loss = val_loss / len(val_dataloader)
        scheduler.step(val_loss)

        # Evaluate BLEU
        eval_results = evaluate_model_detailed(
            model, val_dataloader, gloss_vocab, text_vocab, device, logger
        )

        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]["lr"]

        # Log metrics
        logger.log_epoch(
            epoch + 1,
            train_loss,
            val_loss,
            eval_results["bleu_scores"],
            current_lr,
            epoch_time,
        )

        # Update overall progress bar
        epoch_pbar.set_postfix(
            {
                "TLoss": f"{train_loss:.4f}",
                "VLoss": f"{val_loss:.4f}",
                "BLEU4": f'{eval_results["bleu_scores"]["BLEU-4"]:.4f}',
                "Time": f"{epoch_time:.0f}s",
            }
        )

        # Print detailed progress (still keep this for logging)
        tqdm.write(
            f"Epoch {epoch+1}/{config['num_epochs']} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"BLEU-4: {eval_results['bleu_scores']['BLEU-4']:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Save best models
        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint_path = os.path.join(
                logger.run_dir, "checkpoints", "best_loss.pth"
            )
            save_checkpoint(model, optimizer, epoch, val_loss, config, checkpoint_path)
            tqdm.write(f"   ✓ Best loss model saved")

        if eval_results["bleu_scores"]["BLEU-4"] > best_bleu:
            best_bleu = eval_results["bleu_scores"]["BLEU-4"]
            checkpoint_path = os.path.join(
                logger.run_dir, "checkpoints", "best_bleu.pth"
            )
            save_checkpoint(model, optimizer, epoch, val_loss, config, checkpoint_path)
            tqdm.write(f"   ✓ Best BLEU model saved")

        # Check early stopping
        if early_stopping.should_stop(eval_results["bleu_scores"]["BLEU-4"]):
            tqdm.write(f"\n   Early stopping triggered - BLEU not improving")
            tqdm.write(f"   Best BLEU-4: {best_bleu:.4f}")
            break
    epoch_pbar.close()

    total_train_time = time.time() - train_start_time

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation...")
    final_results = evaluate_model_detailed(
        model, val_dataloader, gloss_vocab, text_vocab, device, logger
    )

    # Create confusion matrix
    print("Generating confusion matrix...")
    conf_matrix, labels = create_confusion_matrix(
        final_results["predictions"], final_results["references"], top_n=20
    )
    conf_matrix_path = os.path.join(logger.run_dir, "plots", "confusion_matrix.png")
    plot_confusion_matrix(conf_matrix, labels, conf_matrix_path)

    # Generate all plots
    print("Generating visualizations...")
    logger.plot_learning_curves()
    logger.plot_bleu_comparison()

    # Save metrics
    logger.save_metrics_csv()
    logger.save_metrics_json()

    # Generate final report
    final_results["total_training_time"] = total_train_time
    final_results["best_val_loss"] = best_loss
    final_results["best_bleu_4"] = best_bleu
    logger.generate_summary_report(config, final_results)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"All results saved to: {logger.run_dir}")
    print("=" * 60)

    return logger, final_results


def build_vocab(sequences, min_freq=5):
    """Build vocabulary from sequences, filtering rare words"""
    counter = Counter()
    for seq in sequences:
        counter.update(seq)

    vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
    idx = 4
    for word, count in counter.items():
        if count >= min_freq:
            vocab[word] = idx
            idx += 1

    return vocab


def collate_fn(batch):
    """Collate function for DataLoader to handle variable length sequences"""
    gloss_seqs, text_seqs = zip(*batch)

    # Get max lengths
    max_gloss_len = max(len(seq) for seq in gloss_seqs)
    max_text_len = max(len(seq) for seq in text_seqs)

    # Pad sequences
    gloss_padded = torch.zeros(len(batch), max_gloss_len, dtype=torch.long)
    text_padded = torch.zeros(len(batch), max_text_len, dtype=torch.long)

    for i, (gloss, text) in enumerate(zip(gloss_seqs, text_seqs)):
        gloss_padded[i, : len(gloss)] = gloss
        text_padded[i, : len(text)] = text

    return gloss_padded, text_padded


def calculate_bleu(predictions, references, max_n=4):
    """
    Calculate BLEU scores (BLEU-1 to BLEU-4)
    Args:
        predictions: List of predicted sentences (each sentence is a list of words)
        references: List of reference sentences (each sentence is a list of words)
        max_n: Maximum n-gram to calculate (default: 4)
    Returns:
        Dictionary with BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
    """
    smoothing = SmoothingFunction().method1
    bleu_scores = {f"BLEU-{i}": [] for i in range(1, max_n + 1)}

    for pred, ref in zip(predictions, references):
        for n in range(1, max_n + 1):
            weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
            score = sentence_bleu(
                [ref], pred, weights=weights, smoothing_function=smoothing
            )
            bleu_scores[f"BLEU-{n}"].append(score)

    # Calculate mean scores
    mean_scores = {k: np.mean(v) for k, v in bleu_scores.items()}
    return mean_scores


# def evaluate_model(model, dataloader, gloss_vocab, text_vocab, device):
#     """
#     Evaluate model on test/validation set and calculate BLEU scores
#     Returns:
#         Dictionary with BLEU scores and example translations
#     """
#     model.eval()
#     predictions = []
#     references = []

#     idx_to_text = {v: k for k, v in text_vocab.items()}

#     with torch.no_grad():
#         for src, trg in dataloader:
#             batch_size = src.size(0)
#             src = src.to(device)

#             # Encode
#             encoder_outputs, hidden = model.encoder(src)

#             # Decode each sentence in batch
#             for b in range(batch_size):
#                 trg_indices = [text_vocab["<SOS>"]]
#                 src_single = src[b : b + 1]
#                 enc_out_single = encoder_outputs[b : b + 1]
#                 hid_single = hidden[:, b : b + 1, :]

#                 for _ in range(50):  # max length
#                     input_token = (
#                         torch.LongTensor([trg_indices[-1]]).unsqueeze(0).to(device)
#                     )
#                     output, hid_single, _ = model.decoder(
#                         input_token, hid_single, enc_out_single
#                     )
#                     pred_token = output.argmax(1).item()
#                     trg_indices.append(pred_token)

#                     if pred_token == text_vocab["<EOS>"]:
#                         break

#                 # Convert to words
#                 pred_words = [
#                     idx_to_text[idx]
#                     for idx in trg_indices[1:-1]
#                     if idx in idx_to_text
#                     and idx_to_text[idx] not in ["<PAD>", "<SOS>", "<EOS>"]
#                 ]
#                 ref_words = [
#                     idx_to_text[idx.item()]
#                     for idx in trg[b]
#                     if idx.item() in idx_to_text
#                     and idx_to_text[idx.item()] not in ["<PAD>", "<SOS>", "<EOS>"]
#                 ]

#                 if pred_words and ref_words:  # Only add if both are non-empty
#                     predictions.append(pred_words)
#                     references.append(ref_words)

#     # Calculate BLEU scores
#     bleu_scores = calculate_bleu(predictions, references)

#     # Get some example translations
#     examples = []
#     for i in range(min(5, len(predictions))):
#         examples.append(
#             {
#                 "prediction": " ".join(predictions[i]),
#                 "reference": " ".join(references[i]),
#             }
#         )

#     return {
#         "bleu_scores": bleu_scores,
#         "examples": examples,
#         "num_samples": len(predictions),
#     }


def save_checkpoint(
    model, optimizer, epoch, train_loss, config, filepath="checkpoint.pth"
):
    """
    Save model checkpoint with all necessary information
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "config": config,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath, device):
    """
    Load model checkpoint
    Returns:
        model, optimizer, epoch, config
    """
    checkpoint = torch.load(filepath, map_location=device)
    return checkpoint


def save_vocabularies(gloss_vocab, text_vocab, filepath="vocabularies.json"):
    """Save vocabularies to JSON file"""
    vocabs = {"gloss_vocab": gloss_vocab, "text_vocab": text_vocab}
    with open(filepath, "w") as f:
        json.dump(vocabs, f, indent=2)
    print(f"Vocabularies saved to {filepath}")


def load_vocabularies(filepath="vocabularies.json"):
    """Load vocabularies from JSON file"""
    with open(filepath, "r") as f:
        vocabs = json.load(f)
    return vocabs["gloss_vocab"], vocabs["text_vocab"]


def save_full_model(model, gloss_vocab, text_vocab, config, save_dir):
    """
    Save complete model package including weights, vocabularies, and config
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save model weights
    torch.save(model.state_dict(), os.path.join(save_dir, "model_weights.pth"))

    # Save vocabularies
    save_vocabularies(
        gloss_vocab, text_vocab, os.path.join(save_dir, "vocabularies.json")
    )

    # Save config
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Full model saved to {save_dir}/")


def load_full_model(save_dir, device):
    """
    Load complete model package
    Returns:
        model, gloss_vocab, text_vocab, config
    """
    # Load config
    with open(os.path.join(save_dir, "config.json"), "r") as f:
        config = json.load(f)

    # Load vocabularies
    gloss_vocab, text_vocab = load_vocabularies(
        os.path.join(save_dir, "vocabularies.json")
    )

    # Create model
    encoder = GRUEncoder(
        config["input_dim"],
        config["embedding_size"],
        config["hidden_size"],
        config["num_layers"],
        config["dropout"],
    )
    decoder = GRUDecoderWithAttention(
        config["output_dim"],
        config["embedding_size"],
        config["hidden_size"],
        config["num_layers"],
        config["dropout"],
        config["attention_type"],
    )
    model = Seq2SeqWithAttention(encoder, decoder, device).to(device)

    # Load weights
    model.load_state_dict(
        torch.load(os.path.join(save_dir, "model_weights.pth"), map_location=device)
    )

    print(f"Full model loaded from {save_dir}/")
    return model, gloss_vocab, text_vocab, config


def get_optimal_batch_size(model, sample_input, device, max_memory_gb=None):
    """
    Automatically determine optimal batch size based on available GPU memory
    """
    if device.type == "cpu":
        return 32  # Default for CPU

    # Start with a small batch and increase
    batch_size = 8
    model.train()

    while True:
        try:
            # Create dummy batch
            src = sample_input[0].repeat(batch_size, 1).to(device)
            trg = sample_input[1].repeat(batch_size, 1).to(device)

            # Try forward pass
            output = model(src, trg)
            loss = output.sum()
            loss.backward()

            # Clear memory
            del src, trg, output, loss
            torch.cuda.empty_cache()

            # Increase batch size
            batch_size *= 2

        except RuntimeError as e:
            if "out of memory" in str(e):
                batch_size = batch_size // 4  # Use 1/4 of max to be safe
                torch.cuda.empty_cache()
                break
            else:
                raise e

    return max(1, batch_size)


def optimize_for_hardware(device):
    """
    Configure PyTorch for optimal performance on available hardware
    Returns recommended settings
    """
    settings = {
        "device": device,
        "num_workers": 0,
        "pin_memory": False,
        "use_amp": False,  # Automatic Mixed Precision
    }

    if device.type == "cuda":
        # GPU optimizations
        torch.backends.cudnn.benchmark = True  # Auto-tune kernels
        settings["num_workers"] = 4
        settings["pin_memory"] = True

        # Check if GPU supports mixed precision (Tensor Cores on RTX/A100)
        if torch.cuda.get_device_capability()[0] >= 7:
            settings["use_amp"] = True
            print("Mixed precision training enabled (faster on RTX/A100 GPUs)")

        # Print GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.2f} GB")

    else:
        # CPU optimizations, default to 6 threads or 3/4ths of available cores
        torch.set_num_threads(4)
        cpu_count = torch.get_num_threads()
        print(f"Using CPU with {cpu_count} threads")

    return settings


def measure_inference_speed(
    model, test_sentences, gloss_vocab, text_vocab, device, num_runs=100
):
    """
    Measure inference speed (sentences per second)
    """
    model.eval()

    times = []
    for _ in range(num_runs):
        sentence = random.choice(test_sentences)

        start_time = time.time()
        _ = translate_sentence(model, sentence, gloss_vocab, text_vocab, device)
        end_time = time.time()

        times.append(end_time - start_time)

    avg_time = np.mean(times)
    std_time = np.std(times)
    sentences_per_sec = 1.0 / avg_time

    return {
        "avg_time_ms": avg_time * 1000,
        "std_time_ms": std_time * 1000,
        "sentences_per_sec": sentences_per_sec,
    }


def load_data_from_file(filepath, file_format="txt"):
    """
    General data loader - supports multiple formats

    Expected formats:

    1. TXT format (tab-separated):
        gloss1 gloss2 gloss3<TAB>word1 word2 word3

    2. JSON format:
        [
            {"gloss": ["gloss1", "gloss2"], "text": ["word1", "word2"]},
            ...
        ]

    3. CSV format (comma-separated):
        gloss,text
        "gloss1 gloss2","word1 word2"

    Args:
        filepath: Path to data file
        file_format: 'txt', 'json', or 'csv'

    Returns:
        gloss_sequences: List of gloss sequences (each is a list of words)
        text_sequences: List of text sequences (each is a list of words)
    """
    gloss_sequences = []
    text_sequences = []

    if file_format == "txt":
        with open(filepath, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Loading {filepath.split('/')[-1]}", leave=False):
                line = line.strip()
                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) == 2:
                    gloss = parts[0].split()
                    text = parts[1].split()
                    gloss_sequences.append(gloss)
                    text_sequences.append(text)

    elif file_format == "json":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                gloss_sequences.append(item["gloss"])
                text_sequences.append(item["text"])

    elif file_format == "csv":
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            # Get total rows for progress bar
            rows = list(reader)

            for row in tqdm(
                rows, desc=f"Loading {filepath.split('/')[-1]}", leave=False
            ):
                # Handle different possible column names for ASLG-PC12
                gloss_text = row.get("gloss", row.get("Gloss", row.get("sign", "")))
                text_text = row.get("text", row.get("Text", row.get("english", "")))

                if gloss_text and text_text:
                    gloss = gloss_text.strip().split()
                    text = text_text.strip().split()

                    if gloss and text:  # Only add non-empty sequences
                        gloss_sequences.append(gloss)
                        text_sequences.append(text)

    else:
        raise ValueError(f"Unsupported file format: {file_format}")

    print(f"Loaded {len(gloss_sequences)} sentence pairs from {filepath}")
    return gloss_sequences, text_sequences


def get_teacher_forcing_ratio(epoch, total_epochs, start_ratio=1.0, end_ratio=0.3):
    """
    exponential decay for better exposure to model predictions.
    """
    warmup_epochs = int(0.2 * total_epochs)

    if epoch < warmup_epochs:
        return start_ratio

    # Exponential decay
    decay_epochs = total_epochs - warmup_epochs
    progress = (epoch - warmup_epochs) / decay_epochs
    current_ratio = start_ratio * ((end_ratio / start_ratio) ** progress)

    return max(end_ratio, current_ratio)


def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    teacher_forcing_ratio=0.5,
    use_amp=False,
    accumulation_steps=4,  # Effective batch size = 32 * 4 = 128
):
    model.train()
    epoch_loss = 0

    # For mixed precision training
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    optimizer.zero_grad()

    # Add progress bar
    pbar = tqdm(
        dataloader, desc="Training", leave=False, bar_format="{l_bar}{bar:30}{r_bar}"
    )

    for batch_idx, (src, trg) in enumerate(pbar):
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()

        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(src, trg, teacher_forcing_ratio)
                output_dim = output.shape[-1]
                output = output[:, 1:].reshape(-1, output_dim)
                trg_flat = trg[:, 1:].reshape(-1)
                loss = criterion(output, trg_flat)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(src, trg, teacher_forcing_ratio)
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg_flat = trg[:, 1:].reshape(-1)
            loss = criterion(output, trg_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        epoch_loss += loss.item() * accumulation_steps

        # Update progress bar with current loss
        pbar.set_postfix({"loss": f"{loss.item() * accumulation_steps:.4f}"})

    pbar.close()
    return epoch_loss / len(dataloader)


def translate_sentence(model, sentence, gloss_vocab, text_vocab, device, max_len=50):
    """Translate a single gloss sequence to text"""
    model.eval()

    # Convert sentence to indices
    gloss_indices = [gloss_vocab.get(g, gloss_vocab["<UNK>"]) for g in sentence]
    src = torch.LongTensor(gloss_indices).unsqueeze(0).to(device)

    # Encode
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src)

    # Decode
    trg_indices = [text_vocab["<SOS>"]]

    for _ in range(max_len):
        input_token = torch.LongTensor([trg_indices[-1]]).unsqueeze(0).to(device)

        with torch.no_grad():
            output, hidden, _ = model.decoder(input_token, hidden, encoder_outputs)

        pred_token = output.argmax(1).item()
        trg_indices.append(pred_token)

        if pred_token == text_vocab["<EOS>"]:
            break

    # Convert indices back to words
    idx_to_text = {v: k for k, v in text_vocab.items()}
    output_sentence = [
        idx_to_text[idx]
        for idx in trg_indices[1:-1]
        if idx in idx_to_text and idx_to_text[idx] not in ["<PAD>", "<UNK>"]
    ]

    return output_sentence


def beam_search_decode(model, src, text_vocab, device, beam_width=5, max_len=50):
    """
    Beam search for better sequence generation during evaluation

    Args:
        model: Seq2SeqWithAttention model
        src: Source tensor (1, src_len)
        text_vocab: Text vocabulary
        device: torch device
        beam_width: Number of beams to maintain
        max_len: Maximum sequence length

    Returns:
        List of token indices (best sequence)
    """
    model.eval()

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src)

        # Initialize beam: (sequence, score, hidden)
        beams = [([text_vocab["<SOS>"]], 0.0, hidden)]

        for _ in range(max_len):
            new_beams = []

            for seq, score, hid in beams:
                if seq[-1] == text_vocab["<EOS>"]:
                    new_beams.append((seq, score, hid))
                    continue

                input_token = torch.LongTensor([seq[-1]]).unsqueeze(0).to(device)
                output, new_hid, _ = model.decoder(input_token, hid, encoder_outputs)

                # Get top k predictions
                log_probs = torch.log_softmax(output, dim=1)
                topk_probs, topk_indices = torch.topk(log_probs, beam_width)

                for prob, idx in zip(topk_probs[0], topk_indices[0]):
                    new_seq = seq + [idx.item()]
                    new_score = score + prob.item()
                    new_beams.append((new_seq, new_score, new_hid))

            # Keep top beam_width beams
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

            # Check if all beams ended
            if all(seq[-1] == text_vocab["<EOS>"] for seq, _, _ in beams):
                break

        # Return best sequence (remove SOS and EOS)
        best_seq = beams[0][0]
        if best_seq[-1] == text_vocab["<EOS>"]:
            best_seq = best_seq[1:-1]
        else:
            best_seq = best_seq[1:]

        return best_seq


def load_synthetic_datasets(hq_path, aslg_path, samples_per_dataset=30000):
    """
    Load pre-processed synthetic gloss-text pairs from CSV files.

    Args:
        hq_path: Path to high-quality English sentences synthetic CSV
        aslg_path: Path to ASLG-PC12 synthetic CSV
        samples_per_dataset: Number of samples to randomly select from each

    Returns:
        gloss_sequences, text_sequences (combined from both datasets)
    """
    all_gloss = []
    all_text = []

    # Load high-quality sentences synthetic
    print(f"   Loading high-quality synthetic from: {hq_path}")
    try:
        hq_gloss, hq_text = load_data_from_file(hq_path, "csv")

        # Randomly sample
        if len(hq_gloss) > samples_per_dataset:
            indices = random.sample(range(len(hq_gloss)), samples_per_dataset)
            hq_gloss = [hq_gloss[i] for i in indices]
            hq_text = [hq_text[i] for i in indices]

        all_gloss.extend(hq_gloss)
        all_text.extend(hq_text)
        print(f"   High-quality synthetic: {len(hq_gloss)} samples")
    except Exception as e:
        print(f"   ⚠ Warning: Could not load {hq_path}: {e}")

    # Load ASLG-PC12 synthetic
    print(f"   Loading ASLG-PC12 synthetic from: {aslg_path}")
    try:
        aslg_gloss, aslg_text = load_data_from_file(aslg_path, "csv")

        # Randomly sample
        if len(aslg_gloss) > samples_per_dataset:
            indices = random.sample(range(len(aslg_gloss)), samples_per_dataset)
            aslg_gloss = [aslg_gloss[i] for i in indices]
            aslg_text = [aslg_text[i] for i in indices]

        all_gloss.extend(aslg_gloss)
        all_text.extend(aslg_text)
        print(f"   ASLG-PC12 synthetic: {len(aslg_gloss)} samples")
    except Exception as e:
        print(f"   ⚠ Warning: Could not load {aslg_path}: {e}")

    print(f"   Total supplementary samples: {len(all_gloss)}")
    return all_gloss, all_text


def combine_datasets(
    primary_gloss,
    primary_text,
    supplemental_gloss,
    supplemental_text,
    primary_weight=2.0,
):
    """
    Combine primary (MediTOD) and supplemental datasets.

    Args:
        primary_gloss/text: Primary dataset sequences
        supplemental_gloss/text: Supplemental dataset sequences
        primary_weight: How many times to include primary data (for emphasis)

    Returns:
        Combined gloss_sequences, text_sequences
    """
    # Repeat primary data based on weight
    combined_gloss = primary_gloss * int(primary_weight)
    combined_text = primary_text * int(primary_weight)

    # Add supplemental data
    combined_gloss.extend(supplemental_gloss)
    combined_text.extend(supplemental_text)

    # Shuffle combined data
    combined = list(zip(combined_gloss, combined_text))
    random.shuffle(combined)
    combined_gloss, combined_text = zip(*combined)

    return list(combined_gloss), list(combined_text)


if __name__ == "__main__":
    from datetime import datetime

    MIN_FREQ = 2

    # Fixed hyperparameters
    BATCH_SIZE = 8
    EMBEDDING_SIZE = None
    DROPOUT = 0.5
    ATTENTION_TYPE = "general"
    LEARNING_RATE = 0.001
    max_samples = 100_000

    param_combinations = [
        (
            MIN_FREQ,
            350,
            2,
            15,
        ),
    ]

    total_experiments = len(param_combinations)

    print("=" * 70)
    print("GLOSS-TO-TEXT TRAINING - MediTOD BASELINE")
    print("=" * 70)
    print(f"\nTotal experiments to run: {total_experiments}")
    print("\nDataset:")
    print("  - MediTOD (medical domain)")
    print("  - Training: 80% split")
    print("  - Validation: 20% split")
    print(f"\nHyperparameters:")
    print(f"  Min Frequency: {MIN_FREQ}")
    print(f"  Attention Type: {ATTENTION_TYPE}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Dropout: {DROPOUT}")
    print("\n" + "=" * 70)

    # Track all results
    all_results = []
    overall_start_time = time.time()

    # ===== STEP 1: LOAD DATASET =====
    print(f"[1/8] Loading MediTOD dataset...")

    gloss_sequences, text_sequences = load_data_from_file(
        "data/dataset/MediTOD/utterances.csv", "csv"
    )
    print(f"   ✓ Loaded {len(gloss_sequences)} samples")

    # Split into train/validation
    split_idx = int(0.8 * len(gloss_sequences))
    train_gloss = gloss_sequences[:split_idx]
    train_text = text_sequences[:split_idx]
    val_gloss = gloss_sequences[split_idx:]
    val_text = text_sequences[split_idx:]

    print(f"\n   Training samples:   {len(train_gloss)}")
    print(f"   Validation samples: {len(val_gloss)}")

    # Run all experiments
    for exp_num, (
        min_freq,
        hidden_size,
        num_layers,
        num_epochs,
    ) in enumerate(param_combinations, 1):
        print(f"\n{'='*70}")
        print(f"EXPERIMENT {exp_num}/{total_experiments}")
        print(f"ARCHITECTURE {exp_num}: {num_layers} LAYERS, {hidden_size} HIDDEN SIZE")
        print(f"{'='*70}")
        print(f"Configuration:")
        print(f"  MAX_SAMPLES: {max_samples:,}")
        print(f"  MIN_FREQ: {min_freq}")
        print(f"  HIDDEN_SIZE: {hidden_size}")
        print(f"  NUM_LAYERS: {num_layers}")
        print(f"  NUM_EPOCHS: {num_epochs}")
        print(f"  EMBEDDING_SIZE: {hidden_size} (same as hidden size)")
        print(f"{'='*70}\n")

        try:

            # ===== STEP 2: BUILD VOCABULARIES =====
            print(f"\n[2/8] Building vocabularies (min_freq={min_freq})...")
            gloss_vocab = build_vocab(train_gloss, min_freq=min_freq)
            text_vocab = build_vocab(train_text, min_freq=min_freq)

            print(f"   Gloss vocabulary size: {len(gloss_vocab)}")
            print(f"   Text vocabulary size: {len(text_vocab)}")

            # ===== STEP 3: HARDWARE OPTIMIZATION =====
            print("\n[3/8] Optimizing for hardware...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            hw_settings = optimize_for_hardware(device)

            # ===== STEP 4: CONFIGURE MODEL =====
            print("\n[4/8] Configuring model...")

            # Set embedding size to match hidden size (as per paper)
            embedding_size = hidden_size

            INPUT_DIM = len(gloss_vocab)
            OUTPUT_DIM = len(text_vocab)

            config = {
                "input_dim": INPUT_DIM,
                "output_dim": OUTPUT_DIM,
                "embedding_size": embedding_size,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": DROPOUT,
                "attention_type": ATTENTION_TYPE,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "num_epochs": num_epochs,
                "min_freq": min_freq,
                "dataset": "MediTOD",
                "train_samples": len(train_gloss),
                "val_samples": len(val_gloss),
            }

            print(f"   Architecture: {num_layers} layers x {hidden_size} hidden units")
            print(f"   Attention type: {config['attention_type']}")
            print(f"   Hidden size: {config['hidden_size']}")
            print(f"   Embedding size: {config['embedding_size']}")
            print(f"   Layers: {config['num_layers']}")
            print(f"   Batch size: {config['batch_size']}")
            print(f"   Epochs: {config['num_epochs']}")
            print(f"   Optimizer: Adamax (lr={config['learning_rate']})")

            # ===== STEP 5: CREATE DATASETS AND DATALOADERS =====
            print("\n[5/8] Creating datasets and dataloaders...")

            train_dataset = GlossTextDataset(
                train_gloss, train_text, gloss_vocab, text_vocab
            )
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=config["batch_size"],
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=hw_settings["num_workers"],
                pin_memory=hw_settings["pin_memory"],
            )

            val_dataset = GlossTextDataset(val_gloss, val_text, gloss_vocab, text_vocab)
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=config["batch_size"],
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=hw_settings["num_workers"],
                pin_memory=hw_settings["pin_memory"],
            )

            print(f"   Training batches: {len(train_dataloader)}")
            print(f"   Validation batches: {len(val_dataloader)}")

            # ===== STEP 6: CREATE MODEL =====
            print("\n[6/8] Creating model...")

            encoder = GRUEncoder(
                config["input_dim"],
                config["embedding_size"],
                config["hidden_size"],
                config["num_layers"],
                config["dropout"],
            )
            decoder = GRUDecoderWithAttention(
                config["output_dim"],
                config["embedding_size"],
                config["hidden_size"],
                config["num_layers"],
                config["dropout"],
                config["attention_type"],
            )
            model = Seq2SeqWithAttention(encoder, decoder, device).to(device)

            optimizer = torch.optim.Adamax(
                model.parameters(), lr=config["learning_rate"]
            )
            criterion = nn.CrossEntropyLoss(
                ignore_index=text_vocab["<PAD>"], label_smoothing=0.1
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, "min", patience=2, factor=0.5
            )

            num_params = sum(p.numel() for p in model.parameters())
            print(f"   Model parameters: {num_params:,}")
            print(f"   Model size: ~{num_params * 4 / 1e6:.1f} MB")

            # ===== STEP 7: TRAINING WITH LOGGING =====
            print(f"\n[7/8] Training with comprehensive logging...")

            # Experiment name matching paper architecture
            experiment_name = (
                f"MediTOD_baseline_l{num_layers}_h{hidden_size}_e{num_epochs}"
            )

            logger, final_results = train_with_logging(
                model=model,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                optimizer=optimizer,
                criterion=criterion,
                scheduler=scheduler,
                gloss_vocab=gloss_vocab,
                text_vocab=text_vocab,
                config=config,
                device=device,
                experiment_name=experiment_name,
            )

            # # ===== STEP 8: STORE RESULTS =====
            experiment_result = {
                "experiment_num": exp_num,
                "architecture": f"Architecture {exp_num}",
                "max_samples": max_samples,
                "min_freq": min_freq,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "num_epochs": num_epochs,
                "embedding_size": embedding_size,
                "bleu_scores": final_results["bleu_scores"],
                "best_val_loss": final_results["best_val_loss"],
                "best_bleu_4": final_results["best_bleu_4"],
                "total_training_time": final_results["total_training_time"],
                "log_dir": logger.run_dir,
            }
            all_results.append(experiment_result)

            print("\n" + "=" * 70)
            print(f"ARCHITECTURE {exp_num} COMPLETED")
            print("=" * 70)
            print(f"BLEU-1: {final_results['bleu_scores']['BLEU-1']:.4f}")
            print(f"BLEU-2: {final_results['bleu_scores']['BLEU-2']:.4f}")
            print(f"BLEU-3: {final_results['bleu_scores']['BLEU-3']:.4f}")
            print(f"BLEU-4: {final_results['bleu_scores']['BLEU-4']:.4f}")
            print(f"Val Loss: {final_results['best_val_loss']:.4f}")
            print(f"Time: {final_results['total_training_time']/60:.1f} min")
            print("=" * 70)

            # Load checkpoints for best loss
            best_loss_checkpoint = load_checkpoint(
                os.path.join(logger.run_dir, "checkpoints", "best_loss.pth"), device
            )
            # measure inference speed for best loss model
            model.load_state_dict(best_loss_checkpoint["model_state_dict"])
            speed_results = measure_inference_speed(
                model, val_gloss[:100], gloss_vocab, text_vocab, device, num_runs=100
            )
            # save the model optimized for best loss
            save_full_model(
                model,
                gloss_vocab,
                text_vocab,
                config,
                save_dir=LOGS_DIR / experiment_name / "best_loss_model",
            )
            model_quantized = torch.quantization.quantize_dynamic(
                model, {nn.GRU, nn.Linear}, dtype=torch.qint8
            )
            # save the model optimized for best loss
            save_full_model(
                model,
                gloss_vocab,
                text_vocab,
                config,
                save_dir=LOGS_DIR / experiment_name / "best_loss_model" / "quantized",
            )

            # Load checkpoints for best BLEU
            best_bleu_checkpoint = load_checkpoint(
                os.path.join(logger.run_dir, "checkpoints", "best_bleu.pth"), device
            )
            # measure inference speed for best bleu model
            model.load_state_dict(best_bleu_checkpoint["model_state_dict"])
            speed_results = measure_inference_speed(
                model, val_gloss[:100], gloss_vocab, text_vocab, device, num_runs=100
            )
            # save the model optimized for best loss
            save_full_model(
                model,
                gloss_vocab,
                text_vocab,
                config,
                save_dir=LOGS_DIR / experiment_name / "best_bleu_model",
            )
            model_quantized = torch.quantization.quantize_dynamic(
                model, {nn.GRU, nn.Linear}, dtype=torch.qint8
            )
            save_full_model(
                model,
                gloss_vocab,
                text_vocab,
                config,
                save_dir=LOGS_DIR / experiment_name / "best_bleu_model" / "quantized",
            )

        except Exception as e:
            print(f"\n{'!'*70}")
            print(f"ERROR IN EXPERIMENT {exp_num}/{total_experiments}")
            print(f"{'!'*70}")
            print(f"Error: {str(e)}")
            print(
                f"Configuration: samples={max_samples}, freq={min_freq}, hidden={hidden_size}, layers={num_layers}"
            )
            print(f"{'!'*70}\n")

            experiment_result = {
                "experiment_num": exp_num,
                "architecture": f"Architecture {exp_num}",
                "max_samples": max_samples,
                "min_freq": min_freq,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "num_epochs": num_epochs,
                "error": str(e),
                "status": "failed",
            }
            all_results.append(experiment_result)
            continue

    # ===== FINAL SUMMARY =====
    total_time = time.time() - overall_start_time

    print("\n" + "=" * 70)
    print("REPLICATION STUDY COMPLETED!")
    print("=" * 70)
    print(f"\nTotal architectures tested: {total_experiments}")
    print(f"Total time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
    print(
        f"Average time per architecture: {total_time/total_experiments/60:.1f} minutes"
    )

    # Save summary results
    summary_path = os.path.join(
        LOGS_DIR,
        f"arvanitis2019_replication_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    os.makedirs(LOGS_DIR, exist_ok=True)

    with open(summary_path, "w") as f:
        json.dump(
            {
                "paper_reference": "Arvanitis et al. (2019) - Translation of Sign Language Glosses to Text",
                "total_experiments": total_experiments,
                "total_time_hours": total_time / 3600,
                "configurations": {
                    "architecture_1": "2 layers, 350 hidden, 5 epochs",
                    "architecture_2": "4 layers, 800 hidden, 10 epochs",
                    "dataset": "ASLG-PC12",
                    "min_freq": MIN_FREQ,
                    "attention_type": ATTENTION_TYPE,
                },
                "results": all_results,
            },
            f,
            indent=2,
        )

    print(f"\nSummary saved to: {summary_path}")
