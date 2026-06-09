# Audio Emotion Classification with CNN and Attention

This project builds a deep learning model for audio emotion classification. The goal is to classify speech audio files into six emotion categories using handcrafted audio features and a 1D convolutional neural network.

The project was developed in Python using PyTorch, librosa, and scikit-learn in a Kaggle notebook environment.

## Project Overview

This project focuses on classifying audio samples into one of six emotion classes:

* Angry
* Disgust
* Fear
* Happy
* Sad
* Neutral

The overall workflow includes:

1. Loading audio metadata from CSV files
2. Mapping emotion labels into integer classes
3. Extracting handcrafted audio features from `.wav` files
4. Creating a custom PyTorch dataset
5. Padding variable-length audio feature sequences with a custom `collate_fn`
6. Building a 1D CNN-based neural network
7. Applying channel attention and temporal attention pooling
8. Training the model with cross-entropy loss
9. Validating model performance
10. Generating predictions for the test set
11. Saving the final submission file

## Dataset

The dataset used in this project is from the Kaggle competition:

`2025 Basic P-3: Emotion Classification via Audio`

The dataset structure is assumed to be:

```text
/kaggle/input/2025-basic-p-3-emotion-classification-via-audio
├── train.csv
├── test.csv
├── sample_submission.csv
├── train/
│   ├── audio files
└── test/
    ├── audio files
```

The training data contains audio file names and corresponding emotion labels. The test data contains audio file names without labels.

## Emotion Label Mapping

The emotion labels are converted into integer values so that they can be used for model training.

```python
label_map = {
    "Angry": 0,
    "Disgust": 1,
    "Fear": 2,
    "Happy": 3,
    "Sad": 4,
    "Neutral": 5
}
```

This mapping is used during training, and the predicted integer labels are converted back into emotion names before generating the submission file.

## Feature Extraction

Instead of feeding raw audio directly into the model, this project extracts handcrafted audio features using `librosa`.

The extracted features are:

* MFCC
* Pitch
* RMS Energy
* Zero Crossing Rate

Each audio file is loaded with a sampling rate of 22050 Hz. The model uses a 3-second audio segment with a 0.5-second offset.

```python
y, sr = librosa.load(file_path, sr=22050, duration=3.0, offset=0.5)
```

### Extracted Feature Dimensions

The final feature matrix consists of:

* 13 MFCC coefficients
* 1 pitch feature
* 1 RMS energy feature
* 1 zero crossing rate feature

Therefore, each time frame has 16 features.

```text
Feature shape: (time_step, 16)
```

During batching, the input is reshaped into the following format for the 1D CNN model:

```text
(batch_size, 16, time_step)
```

## Custom Dataset

A custom PyTorch dataset class is used to load audio files and extract features dynamically.

The dataset class handles both training and test data:

* For training data, it returns both features and labels.
* For test data, it returns only features.

```python
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, label_map, root_path=None, split="Train"):
        self.df = dataframe.reset_index(drop=True)
        self.label_map = label_map
        self.root_path = root_path
        self.split = split.upper()
```

## DataLoader and Padding

Since audio files may produce feature sequences of different lengths, a custom `collate_fn` is used.

The function applies padding so that all sequences in the same batch have the same length.

```python
padded_features = pad_sequence(features, batch_first=True)
padded_features = padded_features.permute(0, 2, 1)
```

This converts the batch into the input shape required by the 1D convolutional model.

Example batch shapes:

```text
Train Batch - Feature shape: torch.Size([32, 16, 130])
Test Batch - Feature shape: torch.Size([32, 16, 117])
```

## Model Architecture

The model is a 1D CNN-based neural network designed for audio sequence classification.

The architecture includes:

* 1D convolutional layers
* Batch normalization
* ReLU activation
* Dropout
* Max pooling
* Squeeze-and-Excitation channel attention
* Temporal attention pooling
* Fully connected classifier

### Attention Components

#### Squeeze-and-Excitation Block

The SE block learns channel-wise importance and reweights feature channels.

```python
class SE1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Conv1d(channels, channels // reduction, 1)
        self.fc2 = nn.Conv1d(channels // reduction, channels, 1)
```

#### Attention Pooling

The attention pooling layer learns which time frames are more important for emotion classification.

```python
class AttentionPooling1D(nn.Module):
    def __init__(self, in_channels, hidden=128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(hidden, 1, kernel_size=1)
        )
```

This helps the model focus on emotionally meaningful parts of the audio sequence.

## Training Setup

The model is trained using the following hyperparameters:

```python
args = {
    "batch_size": 32,
    "epochs": 15,
    "lr": 5e-4,
    "seed_val": 42,
    "patience": 8
}
```

The training set and validation set are split using `train_test_split`.

```python
train_df, val_df = train_test_split(
    full_df,
    test_size=0.1,
    random_state=42,
    shuffle=True
)
```

The model is trained using:

* Loss function: Cross-Entropy Loss
* Optimizer: Adam
* Learning rate: 0.0005
* Batch size: 32
* Epochs: 15

## Training Results

The model showed gradual improvement during training.

Selected training results:

| Epoch | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy |
| ----- | ---------: | -------------: | --------------: | ------------------: |
| 1     |     1.5203 |         0.3750 |          1.5100 |              0.3403 |
| 5     |     1.3036 |         0.4828 |          1.3410 |              0.4522 |
| 10    |     1.1694 |         0.5509 |          1.4662 |              0.4433 |
| 13    |     1.0976 |         0.5814 |          1.1530 |              0.5567 |
| 15    |     1.0547 |         0.5983 |          1.2082 |              0.5269 |

The best validation accuracy was approximately:

```text
0.5567
```

The final epoch achieved:

```text
Train Accuracy: 0.5983
Validation Accuracy: 0.5269
```

## Prediction and Submission

After training, the model predicts emotion classes for the test audio files.

```python
preds = test(test_dataloader, model, device)
preds = int_to_label(label_map, preds)
```

The predicted labels are inserted into the sample submission file.

```python
submit = pd.read_csv(args["submit_path"])
submit["Emotions"] = preds
submit.to_csv("//kaggle/working/results/submission_p3.csv", index=False)
```

The final output file is:

```text
submission_p3.csv
```

The test set contains 745 audio files.

## Example Prediction Output

Example rows from the generated submission file:

```text
Id            Emotions
Test_0.wav    Disgust
Test_1.wav    Neutral
Test_2.wav    Sad
Test_3.wav    Neutral
Test_4.wav    Neutral
```

## Technologies Used

* Python
* PyTorch
* librosa
* NumPy
* pandas
* scikit-learn
* matplotlib
* tqdm
* Kaggle Notebook
* Deep Learning
* Audio Feature Extraction
* 1D Convolutional Neural Network

## How to Run

1. Open the notebook in Kaggle.
2. Add the competition dataset to the notebook.
3. Install required packages if necessary.
4. Run all notebook cells in order.
5. Train the model.
6. Generate predictions for the test set.
7. Save the submission file.

## Project Structure

```text
.
├── 2025-basic-p3-7045.ipynb
├── README.md
└── submission_p3.csv
```

In the Kaggle environment, the dataset is loaded from:

```text
/kaggle/input/2025-basic-p-3-emotion-classification-via-audio
```

The submission file is saved to:

```text
/kaggle/working/results/submission_p3.csv
```

## Limitations

This project uses handcrafted features rather than raw waveform-based or spectrogram-based deep learning approaches. While MFCC, pitch, RMS, and ZCR are useful for audio classification, they may not fully capture all emotional patterns in speech.

The model also uses a relatively simple validation strategy with a single train-validation split. More reliable evaluation could be achieved through k-fold cross-validation.

In addition, the model achieved moderate validation accuracy, indicating that there is still room for improvement in feature extraction, model architecture, and training strategy.

## Future Improvements

Possible future improvements include:

* Using Mel-spectrograms as model input
* Applying data augmentation such as noise injection, time shifting, or pitch shifting
* Using k-fold cross-validation
* Trying pretrained audio models
* Comparing CNN, RNN, LSTM, and Transformer-based models
* Tuning the learning rate, dropout rate, and batch size
* Adding class balancing if the dataset is imbalanced
* Using early stopping based on validation accuracy
* Saving and loading the best model checkpoint

## Author

Kahyun Kim
