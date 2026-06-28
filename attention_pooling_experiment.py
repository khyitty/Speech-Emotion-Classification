import argparse
import json
import os
import random
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


LABEL_MAP = {
    "Angry": 0,
    "Disgust": 1,
    "Fear": 2,
    "Happy": 3,
    "Sad": 4,
    "Neutral": 5,
}

DEFAULT_DATA_DIR_CANDIDATES = [
    Path(os.environ["P3_AUDIO_DATA_DIR"]) if "P3_AUDIO_DATA_DIR" in os.environ else None,
    Path("2025-basic-p-3-emotion-classification-via-audio"),
    Path("p3dataset"),
    Path(r"C:\Users\kahyu\Downloads\p3dataset\2025-basic-p-3-emotion-classification-via-audio"),
    Path("data/2025-basic-p-3-emotion-classification-via-audio"),
    Path("data"),
]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def extract_features(file_path, frame_length=2048, hop_length=512, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=22050, duration=3.0, offset=0.5)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=frame_length,
        hop_length=hop_length,
    ).T

    pitches, _ = librosa.piptrack(
        y=y,
        sr=sr,
        n_fft=frame_length,
        hop_length=hop_length,
    )
    pitch = np.array(
        [pitches[:, t].max() if pitches[:, t].max() > 0 else 0 for t in range(pitches.shape[1])]
    ).reshape(-1, 1)

    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length,
    ).T
    zcr = librosa.feature.zero_crossing_rate(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length,
    ).T

    return np.concatenate([mfcc, pitch, rms, zcr], axis=1)


class AudioEmotionDataset(Dataset):
    def __init__(self, dataframe, data_dir, split, label_map):
        self.df = dataframe.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.split = split.lower()
        self.label_map = label_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_filename = self.df.loc[idx, "Id"]
        audio_path = self.data_dir / self.split / audio_filename
        features = torch.tensor(extract_features(audio_path), dtype=torch.float32)

        if self.split == "test":
            return features

        label = self.label_map[self.df.loc[idx, "Emotions"]]
        return features, label


def collate_fn(batch):
    if isinstance(batch[0], tuple):
        features, labels = zip(*batch)
        padded_features = pad_sequence(features, batch_first=True).permute(0, 2, 1)
        labels = torch.tensor(labels, dtype=torch.long)
        return padded_features, labels

    return pad_sequence(batch, batch_first=True).permute(0, 2, 1)


class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()

        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1),
        )

    def forward(self, x):
        # x: (batch, channels, time)
        x = x.permute(0, 2, 1)
        # x: (batch, time, channels)

        attention_scores = self.attention(x)
        # attention_scores: (batch, time, 1)

        attention_weights = torch.softmax(attention_scores, dim=1)
        # attention_weights: (batch, time, 1)

        pooled = torch.sum(x * attention_weights, dim=1)
        # pooled: (batch, channels)

        return pooled


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(16, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            AttentionPooling(256),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 6),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def train_one_epoch(loader, model, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, labels in tqdm(loader, desc="Training"):
        data = data.float().to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def evaluate(loader, model, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels in tqdm(loader, desc="Validation"):
            data = data.float().to(device)
            labels = labels.to(device)

            logits = model(data)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total


def predict(loader, model, device):
    model.eval()
    preds = []

    with torch.no_grad():
        for data in tqdm(loader, desc="Testing"):
            data = data.float().to(device)
            logits = model(data)
            preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())

    idx_to_label = {v: k for k, v in LABEL_MAP.items()}
    return [idx_to_label[pred] for pred in preds]


def is_valid_data_dir(data_dir):
    required_paths = [
        data_dir / "train.csv",
        data_dir / "test.csv",
        data_dir / "sample_submission.csv",
        data_dir / "train",
        data_dir / "test",
    ]
    return all(path.exists() for path in required_paths)


def summarize_audio_files(data_dir, train_df=None, test_df=None, max_examples=5):
    if train_df is None:
        train_df = pd.read_csv(data_dir / "train.csv")
    if test_df is None:
        test_df = pd.read_csv(data_dir / "test.csv")

    summary = {}
    for split, dataframe in (("train", train_df), ("test", test_df)):
        missing_examples = []
        found = 0
        for audio_filename in dataframe["Id"]:
            audio_path = data_dir / split / audio_filename
            if not audio_path.exists():
                if len(missing_examples) < max_examples:
                    missing_examples.append(str(audio_path))
            else:
                found += 1

        summary[split] = {
            "expected": len(dataframe),
            "found": found,
            "missing": len(dataframe) - found,
            "missing_examples": missing_examples,
        }

    return summary


def is_complete_data_dir(data_dir):
    if not is_valid_data_dir(data_dir):
        return False

    summary = summarize_audio_files(data_dir)
    return all(split_summary["missing"] == 0 for split_summary in summary.values())


def format_audio_summary(summary):
    lines = []
    for split in ("train", "test"):
        split_summary = summary[split]
        lines.append(
            f"{split}: found {split_summary['found']}/{split_summary['expected']} "
            f"audio files, missing {split_summary['missing']}"
        )
        if split_summary["missing_examples"]:
            lines.append(f"    example: {split_summary['missing_examples'][0]}")
    return "; ".join(lines)


def resolve_data_dir(data_dir_arg):
    candidates = []
    if data_dir_arg:
        candidates.append(Path(data_dir_arg))
    candidates.extend(path for path in DEFAULT_DATA_DIR_CANDIDATES if path is not None)

    rejected = []
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if is_complete_data_dir(candidate):
            return candidate
        if is_valid_data_dir(candidate):
            summary = summarize_audio_files(candidate)
            rejected.append(
                f"  - {candidate} exists, but audio files are incomplete. "
                f"{format_audio_summary(summary)}"
            )
        else:
            rejected.append(f"  - {candidate}")

    raise FileNotFoundError(
        "Could not find a complete dataset directory. Expected train.csv, test.csv, "
        "sample_submission.csv, train/*.wav, and test/*.wav.\n"
        "Set it with --data-dir or P3_AUDIO_DATA_DIR.\n"
        "Checked:\n"
        + "\n".join(rejected)
    )


def run(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    data_dir = resolve_data_dir(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using data directory: {data_dir}")
    print(f"Using device: {device}")

    full_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    submit = pd.read_csv(data_dir / "sample_submission.csv")

    if args.limit_rows:
        full_df = full_df.sample(
            n=min(args.limit_rows, len(full_df)),
            random_state=args.seed,
        ).reset_index(drop=True)
        test_df = test_df.head(min(args.limit_rows, len(test_df))).reset_index(drop=True)
        submit = submit.head(len(test_df)).copy()

    train_df, val_df = train_test_split(
        full_df,
        test_size=args.val_size,
        random_state=args.seed,
        shuffle=True,
        stratify=full_df["Emotions"] if args.stratify else None,
    )

    train_loader = DataLoader(
        AudioEmotionDataset(train_df, data_dir, "train", LABEL_MAP),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        AudioEmotionDataset(val_df, data_dir, "train", LABEL_MAP),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        AudioEmotionDataset(test_df, data_dir, "test", LABEL_MAP),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    model = Model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(train_loader, model, criterion, optimizer, device)
        val_loss, val_acc = evaluate(val_loader, model, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(row)
        print(
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            },
            output_dir / f"attention_pooling_epoch_{epoch}.pth",
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "attention_pooling_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping.")
                break

    metrics = {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "history": history,
    }
    with open(output_dir / "attention_pooling_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    model.load_state_dict(torch.load(output_dir / "attention_pooling_best.pth", map_location=device))
    submit["Emotions"] = predict(test_loader, model, device)
    submission_path = output_dir / "attention_pooling_submission.csv"
    submit.to_csv(submission_path, index=False)

    print(f"\nBest validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Saved submission: {submission_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN with linear attention pooling.")
    parser.add_argument(
        "--data-dir",
        default=None,
        help=(
            "Directory containing train.csv, test.csv, sample_submission.csv, train/, "
            "and test/. If omitted, the script checks P3_AUDIO_DATA_DIR and common local paths."
        ),
    )
    parser.add_argument("--output-dir", default="results_attention_pooling")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--stratify", action="store_true")
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
