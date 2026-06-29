# Speech Emotion Classification with Attention Pooling

This project classifies speech audio into six emotion categories using handcrafted audio features and a 1D CNN with temporal attention pooling.

The original Kaggle notebook is kept in the repository, and the newer experiment is implemented as a standalone Python script:

```text
attention_pooling_experiment.py
```

## Task

The model predicts one of the following emotion labels:

```text
Angry, Disgust, Fear, Happy, Sad, Neutral
```

## Dataset

The dataset is from the Kaggle competition:

```text
2025 Basic P-3: Emotion Classification via Audio
```

For local training, place the dataset in this structure:

```text
2025-basic-p-3-emotion-classification-via-audio/
  train.csv
  test.csv
  sample_submission.csv
  train/
    Train_0.wav
    ...
  test/
    Test_0.wav
    ...
```

The dataset directory is intentionally ignored by Git because the audio files are large and should not be committed.

`attention_pooling_experiment.py` automatically checks these locations:

```text
P3_AUDIO_DATA_DIR
2025-basic-p-3-emotion-classification-via-audio/
p3dataset/
data/2025-basic-p-3-emotion-classification-via-audio/
data/
```

## Feature Extraction

Each audio file is loaded with `librosa` using:

```python
librosa.load(file_path, sr=22050, duration=3.0, offset=0.5)
```

The model uses 16 handcrafted features per time step:

| Feature | Dimension |
| --- | ---: |
| MFCC | 13 |
| Pitch | 1 |
| RMS Energy | 1 |
| Zero Crossing Rate | 1 |

The final input shape for the CNN is:

```text
batch_size x 16 x time_steps
```

## Model

The current experiment uses a 1D CNN feature extractor followed by linear attention pooling.

The attention pooling layer learns which time frames are most useful for emotion classification:

```python
class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        attention_scores = self.attention(x)
        attention_weights = torch.softmax(attention_scores, dim=1)
        return torch.sum(x * attention_weights, dim=1)
```

The classifier then maps the pooled 256-dimensional representation to the six emotion classes.

## Training Setup

Default training settings:

| Setting | Value |
| --- | ---: |
| Batch size | 32 |
| Epochs | 15 |
| Learning rate | 5e-4 |
| Validation split | 0.1 |
| Seed | 42 |
| Early stopping patience | 8 |
| Optimizer | Adam |
| Loss | CrossEntropyLoss |

CUDA is used automatically when a compatible GPU-enabled PyTorch installation is available. Use `--cpu` to force CPU execution.

## Results

The latest attention pooling experiment improved the best validation accuracy compared with the previous reported run.

| Epoch | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.5072 | 0.3866 | 1.4693 | 0.3955 |
| 5 | 1.3128 | 0.4798 | 1.2995 | 0.4896 |
| 10 | 1.1651 | 0.5542 | 1.2339 | 0.5299 |
| 11 | 1.1463 | 0.5601 | 1.1570 | 0.5672 |
| 15 | 1.0620 | 0.5971 | 1.1446 | 0.5493 |

Best validation accuracy:

```text
0.5672 at epoch 11
```

The previous README reported a best validation accuracy of approximately `0.5567`, so this attention pooling run is a small improvement.

## How to Run

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the full experiment:

```powershell
python attention_pooling_experiment.py
```

Optional examples:

```powershell
python attention_pooling_experiment.py --cpu
python attention_pooling_experiment.py --epochs 5
python attention_pooling_experiment.py --data-dir "C:\path\to\dataset"
python attention_pooling_experiment.py --limit-rows 100
```

## Outputs

Training outputs are saved to:

```text
results_attention_pooling/
  attention_pooling_best.pth
  attention_pooling_metrics.json
  attention_pooling_submission.csv
```

The results directory is ignored by Git because it contains generated checkpoints and submission files.

## Project Structure

```text
.
  2025-basic-p3-7045.ipynb
  attention_pooling_experiment.py
  requirements.txt
  README.md
```

Local-only files and folders:

```text
2025-basic-p-3-emotion-classification-via-audio/
results_attention_pooling/
```

## Notes

Feature extraction is currently performed on the fly with `librosa`. This makes the first full training run slow because MFCC, pitch, RMS, and ZCR are recomputed from audio files during loading.

A useful next improvement is to cache extracted features as `.npy` or `.pt` files so future runs can train much faster.

## Author

Kahyun Kim
