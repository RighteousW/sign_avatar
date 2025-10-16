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

DATASET:
    Othman, A., & Jemni, M. (2012). "English-ASL Gloss Parallel Corpus 2012: ASLG-PC12"
    5th Workshop on the Representation and Processing of Sign Languages LREC12

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
from collections import Counter
import random
import json
import os
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import time

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
        batch_size = encoder_outputs.size(0)
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

def evaluate_model(model, dataloader, gloss_vocab, text_vocab, device):
    """
    Evaluate model on test/validation set and calculate BLEU scores
    Returns:
        Dictionary with BLEU scores and example translations
    """
    model.eval()
    predictions = []
    references = []

    idx_to_text = {v: k for k, v in text_vocab.items()}

    with torch.no_grad():
        for src, trg in dataloader:
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

                for _ in range(50):  # max length
                    input_token = (
                        torch.LongTensor([trg_indices[-1]]).unsqueeze(0).to(device)
                    )
                    output, hid_single, _ = model.decoder(
                        input_token, hid_single, enc_out_single
                    )
                    pred_token = output.argmax(1).item()
                    trg_indices.append(pred_token)

                    if pred_token == text_vocab["<EOS>"]:
                        break

                # Convert to words
                pred_words = [
                    idx_to_text[idx]
                    for idx in trg_indices[1:-1]
                    if idx in idx_to_text
                    and idx_to_text[idx] not in ["<PAD>", "<SOS>", "<EOS>"]
                ]
                ref_words = [
                    idx_to_text[idx.item()]
                    for idx in trg[b]
                    if idx.item() in idx_to_text
                    and idx_to_text[idx.item()] not in ["<PAD>", "<SOS>", "<EOS>"]
                ]

                if pred_words and ref_words:  # Only add if both are non-empty
                    predictions.append(pred_words)
                    references.append(ref_words)

    # Calculate BLEU scores
    bleu_scores = calculate_bleu(predictions, references)

    # Get some example translations
    examples = []
    for i in range(min(5, len(predictions))):
        examples.append(
            {
                "prediction": " ".join(predictions[i]),
                "reference": " ".join(references[i]),
            }
        )

    return {
        "bleu_scores": bleu_scores,
        "examples": examples,
        "num_samples": len(predictions),
    }

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

def save_full_model(model, gloss_vocab, text_vocab, config, save_dir="model_save"):
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
        # CPU optimizations, default to 4 threads or half of available cores
        torch.set_num_threads(min(4, int(os.cpu_count()) *.5))
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
            for line in f:
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
            for row in reader:
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

def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    teacher_forcing_ratio=0.5,
    use_amp=False,
):
    model.train()
    epoch_loss = 0

    # For mixed precision training
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    for src, trg in dataloader:
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

        epoch_loss += loss.item()

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

if __name__ == "__main__":

    # ===== STEP 1: LOAD FULL DATASET =====
    print("=" * 60)
    print("GLOSS-TO-TEXT TRANSLATOR WITH ATTENTION")
    print("=" * 60)

    # print("\n[1/8] Loading dataset...")
    MAX_SAMPLES = 100_000
    gloss_sequences, text_sequences = load_data_from_file(
        "data/dataset/ASLG-PC12 dataset/synthetic.csv", "csv"
    )
    gloss_sequences = gloss_sequences[:MAX_SAMPLES]
    text_sequences = text_sequences[:MAX_SAMPLES]
    print(f"   Loaded {len(gloss_sequences)} total samples")

    # # Split into train/validation
    split_idx = int(0.8 * len(gloss_sequences))
    train_gloss = gloss_sequences[:split_idx]
    train_text = text_sequences[:split_idx]
    val_gloss = gloss_sequences[split_idx:]
    val_text = text_sequences[split_idx:]

    print(f"   Training samples: {len(train_gloss)}")
    print(f"   Validation samples: {len(val_gloss)}")

    # ===== STEP 2: BUILD VOCABULARIES =====
    print("\n[2/8] Building vocabularies...")
    gloss_vocab = build_vocab(train_gloss, min_freq=5)
    text_vocab = build_vocab(train_text, min_freq=5)

    print(f"   Gloss vocabulary size: {len(gloss_vocab)}")
    print(f"   Text vocabulary size: {len(text_vocab)}")

    # ===== STEP 3: HARDWARE OPTIMIZATION =====
    print("\n[3/8] Optimizing for hardware...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hw_settings = optimize_for_hardware(device)

    # ===== STEP 4: CONFIGURE MODEL =====
    print("\n[4/8] Configuring model...")

    # Optimized config for full dataset on CPU
    HIDDEN_SIZE = 256
    NUM_LAYERS = 2
    NUM_EPOCHS = 10
    BATCH_SIZE = 16
    EMBEDDING_SIZE = 256
    DROPOUT = 0.25
    ATTENTION_TYPE = "dot"
    LEARNING_RATE = 0.001

    INPUT_DIM = len(gloss_vocab)
    OUTPUT_DIM = len(text_vocab)

    config = {
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "embedding_size": EMBEDDING_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "attention_type": ATTENTION_TYPE,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
    }

    print(f"   Attention type: {config['attention_type']}")
    print(f"   Hidden size: {config['hidden_size']}")
    print(f"   Embedding size: {config['embedding_size']}")
    print(f"   Layers: {config['num_layers']}")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Epochs: {config['num_epochs']}")

    # ===== STEP 5: CREATE DATASETS AND DATALOADERS =====
    print("\n[5/8] Creating datasets and dataloaders...")

    # Training dataset
    train_dataset = GlossTextDataset(train_gloss, train_text, gloss_vocab, text_vocab)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=hw_settings["num_workers"],
        pin_memory=hw_settings["pin_memory"],
    )

    # Validation dataset
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

    # Optimizer and loss
    optimizer = torch.optim.Adamax(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss(ignore_index=text_vocab["<PAD>"])

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=2, factor=0.5
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {num_params:,}")
    print(f"   Model size: ~{num_params * 4 / 1e6:.1f} MB")

    # ===== STEP 7: TRAIN MODEL =====
    print(f"\n[7/8] Training for {config['num_epochs']} epochs...")
    print("=" * 60)

    best_loss = float("inf")
    best_bleu = 0.0
    train_start_time = time.time()

    train_losses = []
    val_losses = []

    for epoch in range(config["num_epochs"]):
        epoch_start_time = time.time()

        # Training
        train_loss = train_epoch(
            model,
            train_dataloader,
            optimizer,
            criterion,
            device,
            teacher_forcing_ratio=0.5,
            use_amp=hw_settings["use_amp"],
        )
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, trg in val_dataloader:
                src, trg = src.to(device), trg.to(device)
                output = model(src, trg, teacher_forcing_ratio=0)
                output_dim = output.shape[-1]
                output = output[:, 1:].reshape(-1, output_dim)
                trg_flat = trg[:, 1:].reshape(-1)
                loss = criterion(output, trg_flat)
                val_loss += loss.item()

        val_loss = val_loss / len(val_dataloader)
        val_losses.append(val_loss)

        # Update learning rate
        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start_time

        print(
            f"Epoch {epoch+1}/{config['num_epochs']} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Save best model based on validation loss
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss,
                config,
                "models/checkpoints/gloss2text_synthetic.pth",
            )
            print(f"   ✓ New best model saved (val_loss: {val_loss:.4f})")

        # Evaluate BLEU every 2 epochs
        if (epoch + 1) % 2 == 0:
            print(f"   Evaluating BLEU score...")
            eval_results = evaluate_model(
                model, val_dataloader, gloss_vocab, text_vocab, device
            )
            current_bleu = eval_results["bleu_scores"]["BLEU-4"]
            print(f"   BLEU-4: {current_bleu:.4f} ({current_bleu*100:.2f}%)")

            if current_bleu > best_bleu:
                best_bleu = current_bleu
                save_full_model(
                    model,
                    gloss_vocab,
                    text_vocab,
                    config,
                    save_dir="models/trained_models/gloss2text_best_bleu_synthetic",
                )
                print(f"   ✓ Best BLEU model saved")

    total_train_time = time.time() - train_start_time
    print(f"\n{'='*60}")
    print(
        f"Training completed in {total_train_time:.1f}s ({total_train_time/60:.1f} minutes)"
    )
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Best BLEU-4 score: {best_bleu:.4f} ({best_bleu*100:.2f}%)")

    # ===== STEP 8: FINAL EVALUATION =====
    print("\n[8/8] Final evaluation on validation set...")
    print("=" * 60)

    # Load best model
    checkpoint = load_checkpoint(
        "models/checkpoints/gloss2text_synthetic.pth", device
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    eval_results = evaluate_model(
        model, val_dataloader, gloss_vocab, text_vocab, device
    )

    print(f"\nEvaluated on {eval_results['num_samples']} validation samples")
    print("\nBLEU Scores:")
    for metric, score in eval_results["bleu_scores"].items():
        print(f"  {metric}: {score:.4f} ({score*100:.2f}%)")

    print("\nExample Translations:")
    for i, ex in enumerate(eval_results["examples"][:5], 1):
        print(f"\n{i}. Prediction: {ex['prediction']}")
        print(f"   Reference:  {ex['reference']}")

    # ===== STEP 9: MEASURE INFERENCE SPEED =====
    print("\n" + "=" * 60)
    print("Measuring inference speed...")

    speed_results = measure_inference_speed(
        model, val_gloss[:100], gloss_vocab, text_vocab, device, num_runs=100
    )

    print(
        f"Average time per sentence: {speed_results['avg_time_ms']:.2f}ms ± {speed_results['std_time_ms']:.2f}ms"
    )
    print(f"Throughput: {speed_results['sentences_per_sec']:.1f} sentences/second")

    # ===== STEP 10: SAVE FINAL MODEL =====
    print("\n" + "=" * 60)
    print("Saving final model package...")
    save_full_model(
        model,
        gloss_vocab,
        text_vocab,
        config,
        save_dir="models/trained_models/gloss2text_final_synthetic",
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print("\nSaved models:")
    print("  - best_model_synthetic.pth (lowest validation loss)")
    print("  - gloss2text_best_bleu_synthetic/ (highest BLEU score)")
    print("  - gloss2text_final_synthetic/ (final epoch)")
    print("\nTo use the trained model:")
    print(
        "  model, gloss_vocab, text_vocab, config = load_full_model('gloss2text_best_bleu_synthetic', device)"
    )
    print(
        "  translation = translate_sentence(model, ['GLOSS1', 'GLOSS2'], gloss_vocab, text_vocab, device)"
    )
