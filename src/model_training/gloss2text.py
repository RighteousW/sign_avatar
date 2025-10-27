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
    - MediTOD: https://github.com/dair-iitd/MediTOD/tree/main

EVALUATION:
    Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2002). "BLEU: a method for
    automatic evaluation of machine translation." ACL 2002.

For full citations and references, see the accompanying CITATIONS.md file.

Implementation by: Righteous Wasambo
Date: 2025
License: ?
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import csv
import os
import json
import random
import numpy as np
from collections import Counter
from datetime import datetime
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm


# Reproducibility
def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed} for reproducibility\n")


# Model Components
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(
            embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        outputs, hidden = self.gru(embedded)
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(
            embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

        # Simple attention
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x, hidden, encoder_outputs):
        # x: (batch, 1)
        embedded = self.dropout(self.embedding(x))
        gru_out, hidden = self.gru(embedded, hidden)

        # Attention
        batch_size = encoder_outputs.size(0)
        src_len = encoder_outputs.size(1)

        gru_out_expanded = gru_out.repeat(1, src_len, 1)
        energy = torch.tanh(
            self.attn(torch.cat([gru_out_expanded, encoder_outputs], dim=2))
        )
        attention = torch.softmax(self.v(energy), dim=1)
        context = torch.sum(attention * encoder_outputs, dim=1, keepdim=True)

        output = self.fc(torch.cat([gru_out, context], dim=2).squeeze(1))
        return output, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)

        x = trg[:, 0].unsqueeze(1)

        for t in range(1, trg_len):
            output, hidden = self.decoder(x, hidden, encoder_outputs)
            outputs[:, t] = output

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            x = (
                trg[:, t].unsqueeze(1)
                if teacher_force
                else output.argmax(1).unsqueeze(1)
            )

        return outputs


# Dataset and DataLoader
class GlossDataset(Dataset):
    def __init__(self, gloss_seqs, text_seqs, gloss_vocab, text_vocab):
        self.gloss_seqs = gloss_seqs
        self.text_seqs = text_seqs
        self.gloss_vocab = gloss_vocab
        self.text_vocab = text_vocab

    def __len__(self):
        return len(self.gloss_seqs)

    def __getitem__(self, idx):
        gloss = [
            self.gloss_vocab.get(w, self.gloss_vocab["<UNK>"])
            for w in self.gloss_seqs[idx]
        ]
        text = (
            [self.text_vocab["<SOS>"]]
            + [
                self.text_vocab.get(w, self.text_vocab["<UNK>"])
                for w in self.text_seqs[idx]
            ]
            + [self.text_vocab["<EOS>"]]
        )
        return torch.LongTensor(gloss), torch.LongTensor(text)


def collate_fn(batch):
    gloss_seqs, text_seqs = zip(*batch)

    gloss_lens = [len(s) for s in gloss_seqs]
    text_lens = [len(s) for s in text_seqs]

    gloss_padded = torch.zeros(len(batch), max(gloss_lens), dtype=torch.long)
    text_padded = torch.zeros(len(batch), max(text_lens), dtype=torch.long)

    for i, (g, t) in enumerate(zip(gloss_seqs, text_seqs)):
        gloss_padded[i, : len(g)] = g
        text_padded[i, : len(t)] = t

    return gloss_padded, text_padded


def load_data_from_file(filepath, file_format="csv", max_len=50):
    """
    Load data from file. Supports CSV, TXT, JSON formats.

    Args:
        filepath: Path to data file
        file_format: 'csv', 'txt', or 'json'
        max_len: Maximum sequence length (filters out longer sequences)

    Returns:
        gloss_sequences: List of gloss sequences (each is a list of words)
        text_sequences: List of text sequences (each is a list of words)
    """
    gloss_seqs, text_seqs = [], []

    if file_format == "csv":
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                gloss_text = row.get("gloss", row.get("Gloss", ""))
                text_text = row.get("text", row.get("Text", ""))

                if gloss_text and text_text:
                    gloss = gloss_text.strip().split()
                    text = text_text.strip().split()

                    # Filter by length
                    if 1 < len(gloss) <= max_len and 1 < len(text) <= max_len:
                        gloss_seqs.append(gloss)
                        text_seqs.append(text)

    elif file_format == "txt":
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) == 2:
                    gloss = parts[0].split()
                    text = parts[1].split()
                    if 1 < len(gloss) <= max_len and 1 < len(text) <= max_len:
                        gloss_seqs.append(gloss)
                        text_seqs.append(text)

    elif file_format == "json":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                gloss = item["gloss"]
                text = item["text"]
                if 1 < len(gloss) <= max_len and 1 < len(text) <= max_len:
                    gloss_seqs.append(gloss)
                    text_seqs.append(text)

    return gloss_seqs, text_seqs


def build_vocab(sequences, max_vocab_size=3000):
    """Keep only the most frequent words"""
    counter = Counter(word for seq in sequences for word in seq)

    vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}

    # Keep only top N most frequent words
    most_common = counter.most_common(max_vocab_size - 4)  # -4 for special tokens

    for word, _ in most_common:
        vocab[word] = len(vocab)

    print(f"  Total unique words: {len(counter)}")
    print(f"  Kept top {len(most_common)} words")
    print(f"  Final vocab size: {len(vocab)}")

    return vocab


# Utilities for Saving and Loading Models


def save_full_model(model, gloss_vocab, text_vocab, config, save_dir):
    """Save complete model package including weights, vocabularies, and config"""
    os.makedirs(save_dir, exist_ok=True)

    # Save model weights
    torch.save(model.state_dict(), os.path.join(save_dir, "model_weights.pth"))

    # Save vocabularies
    vocabs = {"gloss_vocab": gloss_vocab, "text_vocab": text_vocab}
    with open(os.path.join(save_dir, "vocabularies.json"), "w") as f:
        json.dump(vocabs, f, indent=2)

    # Save config
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Model saved to {save_dir}/")


def quantize_model(model):
    """
    Apply dynamic quantization to reduce model size and improve inference speed.
    Quantizes Linear and GRU layers to int8.

    Args:
        model: Original model

    Returns:
        Quantized model (typically 4x smaller, 2-3x faster on CPU)
    """
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear, nn.GRU}, dtype=torch.qint8
    )
    return quantized_model


def save_quantized_model(model, gloss_vocab, text_vocab, config, save_dir):
    """Save quantized version of the model"""
    quantized_dir = os.path.join(save_dir, "quantized")
    os.makedirs(quantized_dir, exist_ok=True)

    # Quantize model
    print(f"\nQuantizing model...")
    quantized_model = quantize_model(model)

    # Get model sizes
    original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (
        1024 * 1024
    )
    quantized_size = sum(
        p.numel() * p.element_size() for p in quantized_model.parameters()
    ) / (1024 * 1024)

    print(f"  Original model size: {original_size:.2f} MB")
    print(f"  Quantized model size: {quantized_size:.2f} MB")
    print(f"  Compression ratio: {original_size/quantized_size:.2f}x")

    # Save quantized model weights
    torch.save(
        quantized_model.state_dict(), os.path.join(quantized_dir, "model_weights.pth")
    )

    # Copy vocabularies and config
    vocabs = {"gloss_vocab": gloss_vocab, "text_vocab": text_vocab}
    with open(os.path.join(quantized_dir, "vocabularies.json"), "w") as f:
        json.dump(vocabs, f, indent=2)

    # Mark config as quantized
    config_quantized = config.copy()
    config_quantized["quantized"] = True
    config_quantized["original_size_mb"] = original_size
    config_quantized["quantized_size_mb"] = quantized_size
    config_quantized["compression_ratio"] = original_size / quantized_size

    with open(os.path.join(quantized_dir, "config.json"), "w") as f:
        json.dump(config_quantized, f, indent=2)

    print(f"  Quantized model saved to {quantized_dir}/")

    return quantized_model


def load_full_model(
    save_dir="models/trained_models/gloss2text_logs/MediTOD+Supp_b32_h256_e10_20251026_130017",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    """
    Load complete model package

    Returns:
        model, gloss_vocab, text_vocab, config
    """
    # Load config
    with open(os.path.join(save_dir, "config.json"), "r") as f:
        config = json.load(f)

    # Load vocabularies
    with open(os.path.join(save_dir, "vocabularies.json"), "r") as f:
        vocabs = json.load(f)
    gloss_vocab = vocabs["gloss_vocab"]
    text_vocab = vocabs["text_vocab"]

    # Create model
    encoder = Encoder(
        len(gloss_vocab),
        config["embed_size"],
        config["hidden_size"],
        config["num_layers"],
        config["dropout"],
    )
    decoder = Decoder(
        len(text_vocab),
        config["embed_size"],
        config["hidden_size"],
        config["num_layers"],
        config["dropout"],
    )
    model = Seq2Seq(encoder, decoder, device).to(device)

    # Load weights
    state_dict = torch.load(
        os.path.join(save_dir, "model_weights.pth"),
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(state_dict)

    # Check if this is a quantized model
    is_quantized = config.get("quantized", False)
    if is_quantized:
        print(f"Loaded quantized model from {save_dir}/")
    else:
        print(f"Model loaded from {save_dir}/")

    return model, gloss_vocab, text_vocab, config


def load_checkpoint(filepath, device):
    """Load checkpoint for resuming training"""
    checkpoint = torch.load(filepath, map_location=device)
    return checkpoint


def translate_sentence(model, sentence, gloss_vocab, text_vocab, device, max_len=None):
    """
    Translate a single gloss sequence to text

    Args:
        model: Seq2Seq model
        sentence: List of gloss tokens, e.g., ['WHAT', 'YOU', 'NAME']
        gloss_vocab: Gloss vocabulary dict
        text_vocab: Text vocabulary dict
        device: torch device
        max_len: Maximum output length

    Returns:
        List of text tokens
    """
    model.eval()
    max_len = int(len(sentence) * 1.5) if not max_len else max_len

    # Convert sentence to indices
    gloss_indices = [gloss_vocab.get(g, gloss_vocab["<UNK>"]) for g in sentence]
    src = torch.LongTensor(gloss_indices).unsqueeze(0).to(device)

    # Encode
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src)

    # Decode
    idx_to_text = {v: k for k, v in text_vocab.items()}
    tokens = [text_vocab["<SOS>"]]

    for _ in range(max_len):
        x = torch.LongTensor([tokens[-1]]).unsqueeze(0).to(device)

        with torch.no_grad():
            output, hidden = model.decoder(x, hidden, encoder_outputs)

        pred_token = output.argmax(1).item()
        tokens.append(pred_token)

        if pred_token == text_vocab["<EOS>"]:
            break

    # Convert indices back to words
    output_sentence = [
        idx_to_text[idx]
        for idx in tokens[1:-1]
        if idx in idx_to_text
        and idx_to_text[idx] not in ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]
    ]

    return output_sentence


# Training and Evaluation Functions

def train_epoch(model, dataloader, optimizer, criterion, device, tf_ratio=0.5):
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for src, trg in pbar:
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()
        output = model(src, trg, tf_ratio)

        output = output[:, 1:].reshape(-1, output.shape[-1])
        trg = trg[:, 1:].reshape(-1)
        loss = criterion(output, trg)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(dataloader)


def evaluate(model, dataloader, gloss_vocab, text_vocab, device):
    model.eval()
    idx_to_text = {v: k for k, v in text_vocab.items()}

    predictions, references = [], []

    with torch.no_grad():
        for src, trg in tqdm(dataloader, desc="Evaluating", leave=False):
            src = src.to(device)
            batch_size = src.size(0)

            encoder_outputs, hidden = model.encoder(src)

            for b in range(batch_size):
                tokens = [text_vocab["<SOS>"]]
                enc_out = encoder_outputs[b : b + 1]
                hid = hidden[:, b : b + 1, :]

                for _ in range(50):
                    x = torch.LongTensor([tokens[-1]]).unsqueeze(0).to(device)
                    output, hid = model.decoder(x, hid, enc_out)
                    pred = output.argmax(1).item()
                    tokens.append(pred)
                    if pred == text_vocab["<EOS>"]:
                        break

                pred_words = [
                    idx_to_text[t]
                    for t in tokens[1:-1]
                    if t in idx_to_text
                    and idx_to_text[t] not in ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
                ]
                ref_words = [
                    idx_to_text[t.item()]
                    for t in trg[b]
                    if t.item() in idx_to_text
                    and idx_to_text[t.item()] not in ["<PAD>", "<SOS>", "<EOS>"]
                ]

                if pred_words and ref_words:
                    predictions.append(pred_words)
                    references.append(ref_words)

    # Calculate BLEU-1, BLEU-2, BLEU-3, BLEU-4
    smoothing = SmoothingFunction().method1
    bleu_scores = {"BLEU-1": [], "BLEU-2": [], "BLEU-3": [], "BLEU-4": []}

    for pred, ref in zip(predictions, references):
        # BLEU-1
        score_1 = sentence_bleu(
            [ref], pred, weights=(1, 0, 0, 0), smoothing_function=smoothing
        )
        bleu_scores["BLEU-1"].append(score_1)

        # BLEU-2
        score_2 = sentence_bleu(
            [ref], pred, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing
        )
        bleu_scores["BLEU-2"].append(score_2)

        # BLEU-3
        score_3 = sentence_bleu(
            [ref], pred, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing
        )
        bleu_scores["BLEU-3"].append(score_3)

        # BLEU-4
        score_4 = sentence_bleu(
            [ref], pred, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing
        )
        bleu_scores["BLEU-4"].append(score_4)

    # Calculate averages
    avg_bleu = {
        "BLEU-1": (
            sum(bleu_scores["BLEU-1"]) / len(bleu_scores["BLEU-1"])
            if bleu_scores["BLEU-1"]
            else 0.0
        ),
        "BLEU-2": (
            sum(bleu_scores["BLEU-2"]) / len(bleu_scores["BLEU-2"])
            if bleu_scores["BLEU-2"]
            else 0.0
        ),
        "BLEU-3": (
            sum(bleu_scores["BLEU-3"]) / len(bleu_scores["BLEU-3"])
            if bleu_scores["BLEU-3"]
            else 0.0
        ),
        "BLEU-4": (
            sum(bleu_scores["BLEU-4"]) / len(bleu_scores["BLEU-4"])
            if bleu_scores["BLEU-4"]
            else 0.0
        ),
    }

    return avg_bleu


# API Aliases for backward compatibility
# Match old naming for backward compatibility
GRUEncoder = Encoder
GRUDecoderWithAttention = Decoder
Seq2SeqWithAttention = Seq2Seq

# Function aliases
gloss2text_translate_sentence = translate_sentence
load_gloss2text_full_model = load_full_model


if __name__ == "__main__":
    # Set seed for reproducibility
    set_seed(42)

    # Config
    PRIMARY_DATA_PATH = "data/dataset/ASLG-PC12 dataset/train.csv"
    SUPPLEMENTARY_PATHS = []
    SAMPLES_PER_SUPPLEMENTARY = (
        10000  # Number of samples to draw from each supplementary dataset
    )

    EMBED_SIZE = 300
    HIDDEN_SIZE = 500
    NUM_LAYERS = 3
    DROPOUT = 0.3
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0005
    NUM_EPOCHS = 5
    MIN_FREQ = 20
    MAX_LEN = 50
    EARLY_STOPPING_PATIENCE = 3

    # Create timestamped save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"models/trained_models/gloss2text_logs/ASLG-PC12_{BATCH_SIZE}_h{HIDDEN_SIZE}_e{NUM_EPOCHS}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save directory: {save_dir}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load primary data (MediTOD)
    print("Loading Primary dataset...")
    primary_gloss, primary_text = load_data_from_file(
        PRIMARY_DATA_PATH, file_format="csv", max_len=MAX_LEN
    )
    print(f"  Primary: {len(primary_gloss)} samples")

    # Load supplementary data
    supplementary_gloss = []
    supplementary_text = []

    for supp_path in SUPPLEMENTARY_PATHS:
        print(f"\nLoading {supp_path.split('/')[-2]}...")
        try:
            supp_g, supp_t = load_data_from_file(
                supp_path, file_format="csv", max_len=MAX_LEN
            )

            # Sample random subset
            if len(supp_g) > SAMPLES_PER_SUPPLEMENTARY:
                indices = random.sample(range(len(supp_g)), SAMPLES_PER_SUPPLEMENTARY)
                supp_g = [supp_g[i] for i in indices]
                supp_t = [supp_t[i] for i in indices]

            supplementary_gloss.extend(supp_g)
            supplementary_text.extend(supp_t)
            print(f"  Loaded: {len(supp_g)} samples")
        except Exception as e:
            print(f"  ⚠ Warning: Could not load {supp_path}: {e}")

    print(f"\nTotal supplementary: {len(supplementary_gloss)} samples")

    # Split Primary for train/val (validation is Primary-only)
    primary_split = int(0.8 * len(primary_gloss))

    primary_train_gloss = primary_gloss[:primary_split]
    primary_train_text = primary_text[:primary_split]
    val_gloss = primary_gloss[primary_split:]
    val_text = primary_text[primary_split:]

    print(f"\nDataset composition:")
    print(f"  Primary training:   {len(primary_train_gloss)}")
    print(f"  Primary validation: {len(val_gloss)} samples")
    print(f"  Supplementary:      {len(supplementary_gloss)} samples")

    # Combine training data (increase weight of Primary if supplementary is valid)
    supplementary_valid = len(supplementary_gloss) > 0
    if supplementary_valid:
        train_gloss = primary_train_gloss * 2 + supplementary_gloss # 20% of Primary * 2
        train_text = primary_train_text * 2 + supplementary_text
    else:
        train_gloss = primary_train_gloss
        train_text = primary_train_text

    # Shuffle combined training data
    combined = list(zip(train_gloss, train_text))
    random.shuffle(combined)
    train_gloss, train_text = zip(*combined)
    train_gloss, train_text = list(train_gloss), list(train_text)

    print(f"\nFinal training set: {len(train_gloss)} samples")
    print(f"Final validation set: {len(val_gloss)} samples (Primary-only)\n")

    # Build vocabs
    print("Building vocabularies...")
    gloss_vocab = build_vocab(train_gloss, max_vocab_size=20000)
    text_vocab = build_vocab(train_text, max_vocab_size=40000)
    print(f"Gloss vocab: {len(gloss_vocab)}, Text vocab: {len(text_vocab)}\n")

    # Create datasets
    train_dataset = GlossDataset(train_gloss, train_text, gloss_vocab, text_vocab)
    val_dataset = GlossDataset(val_gloss, val_text, gloss_vocab, text_vocab)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    # Create model
    print("Creating model...")
    encoder = Encoder(len(gloss_vocab), EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
    decoder = Decoder(len(text_vocab), EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
    model = Seq2Seq(encoder, decoder, device).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}\n")

    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=text_vocab["<PAD>"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=0.5
    )

    print("Training...")
    print("=" * 60)

    best_bleu = 0
    patience_counter = 0
    training_history = []

    for epoch in range(NUM_EPOCHS):
        tf_ratio = max(0.3, 1.0 - epoch * 0.05)  # Decay from 1.0 to 0.3

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, tf_ratio
        )
        bleu_scores = evaluate(model, val_loader, gloss_vocab, text_vocab, device)

        # Learning rate scheduling (use BLEU-4)
        scheduler.step(bleu_scores["BLEU-4"])
        current_lr = optimizer.param_groups[0]["lr"]

        # Track history
        training_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "bleu_1": bleu_scores["BLEU-1"],
                "bleu_2": bleu_scores["BLEU-2"],
                "bleu_3": bleu_scores["BLEU-3"],
                "bleu_4": bleu_scores["BLEU-4"],
                "lr": current_lr,
            }
        )

        print(
            f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | Loss: {train_loss:.4f} | "
            f"BLEU-1: {bleu_scores['BLEU-1']:.4f} | BLEU-2: {bleu_scores['BLEU-2']:.4f} | "
            f"BLEU-3: {bleu_scores['BLEU-3']:.4f} | BLEU-4: {bleu_scores['BLEU-4']:.4f} | LR: {current_lr:.6f}"
        )

        if bleu_scores["BLEU-4"] > best_bleu:
            best_bleu = bleu_scores["BLEU-4"]
            patience_counter = 0

            # Save using save_full_model for API compatibility
            config = {
                "embed_size": EMBED_SIZE,
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
                "min_freq": MIN_FREQ,
                "max_len": MAX_LEN,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "timestamp": timestamp,
                "best_bleu_1": bleu_scores["BLEU-1"],
                "best_bleu_2": bleu_scores["BLEU-2"],
                "best_bleu_3": bleu_scores["BLEU-3"],
                "best_bleu_4": bleu_scores["BLEU-4"],
                "best_epoch": epoch + 1,
                "dataset": "MediTOD + Supplementary (60K)",
                "total_train_samples": len(train_gloss),
            }
            save_full_model(model, gloss_vocab, text_vocab, config, save_dir)

            print(f"  → New best BLEU-4! (saved)")
        else:
            patience_counter += 1
            print(
                f"  → No improvement (patience: {patience_counter}/{EARLY_STOPPING_PATIENCE})"
            )

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(
                    f"\n  ⚠ Early stopping triggered - no improvement for {EARLY_STOPPING_PATIENCE} epochs"
                )
                break

    # Save training history
    history_path = os.path.join(save_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)
    print(f"\nTraining history saved to: {history_path}")

    # Also save quantized version
    save_quantized_model(model, gloss_vocab, text_vocab, config, save_dir)

    print("=" * 60)
    print(f"\nBest BLEU Scores:")
    print(f"  BLEU-1: {config['best_bleu_1']:.4f}")
    print(f"  BLEU-2: {config['best_bleu_2']:.4f}")
    print(f"  BLEU-3: {config['best_bleu_3']:.4f}")
    print(f"  BLEU-4: {config['best_bleu_4']:.4f}")
    print(f"\nModel saved to: {save_dir}/")
    print(f"Quantized model saved to: {save_dir}/quantized/")

    # Test quantized model performance
    print("\n" + "=" * 60)
    print("Testing Quantized Model Performance:")
    print("=" * 60)

    quantized_dir = os.path.join(save_dir, "quantized")
    quantized_model, _, _, quant_config = load_full_model(quantized_dir, device)

    print("\nEvaluating quantized model on validation set...")
    quantized_bleu = evaluate(
        quantized_model, val_loader, gloss_vocab, text_vocab, device
    )

    print(f"\nOriginal model:")
    print(
        f"  BLEU-1: {config['best_bleu_1']:.4f} | BLEU-2: {config['best_bleu_2']:.4f} | "
        f"BLEU-3: {config['best_bleu_3']:.4f} | BLEU-4: {config['best_bleu_4']:.4f}"
    )
    print(f"Quantized model:")
    print(
        f"  BLEU-1: {quantized_bleu['BLEU-1']:.4f} | BLEU-2: {quantized_bleu['BLEU-2']:.4f} | "
        f"BLEU-3: {quantized_bleu['BLEU-3']:.4f} | BLEU-4: {quantized_bleu['BLEU-4']:.4f}"
    )
    print(
        f"BLEU-4 difference: {abs(config['best_bleu_4'] - quantized_bleu['BLEU-4']):.4f} "
        f"({abs(config['best_bleu_4'] - quantized_bleu['BLEU-4'])/config['best_bleu_4']*100:.2f}%)"
    )

    # Show some example translations
    print("\n" + "=" * 60)
    print("Example Translations:")
    print("=" * 60)
    model, gloss_vocab, text_vocab, config = load_full_model(save_dir, device)

    examples = []
    for i in range(min(5, len(val_gloss))):
        gloss = val_gloss[i]
        reference = val_text[i]
        translation = translate_sentence(model, gloss, gloss_vocab, text_vocab, device)

        example = {
            "gloss": " ".join(gloss),
            "reference": " ".join(reference),
            "translation": " ".join(translation),
        }
        examples.append(example)

        print(f"\nGloss:       {example['gloss']}")
        print(f"Reference:   {example['reference']}")
        print(f"Translation: {example['translation']}")

    # Save examples
    examples_path = os.path.join(save_dir, "example_translations.json")
    with open(examples_path, "w") as f:
        json.dump(examples, f, indent=2)

    print("=" * 60)
    print(f"\nExamples saved to: {examples_path}")
