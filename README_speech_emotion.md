# Task 2: Emotion Recognition from Speech

## Objective
Recognize human emotions (neutral, calm, happy, sad, angry, fearful, disgust, surprised)
from speech audio.

## Dataset
RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) — audio-only speech
files. **Not included** — download it yourself:

1. Go to the [RAVDESS Zenodo page](https://zenodo.org/record/1188976).
2. Download `Audio_Speech_Actors_01-24.zip`.
3. Unzip it — this creates folders `Actor_01` through `Actor_24`, each containing `.wav`
   files.
4. Place that unzipped folder next to the script, or edit `DATA_DIR` at the top of the
   script to point to it.

The emotion label for each file is read directly from its filename (RAVDESS encodes it
as the 3rd number, e.g. `03-01-06-01-02-01-12.wav` = fearful).

## Approach
Deep learning + speech signal processing:
- **Feature extraction:** 40 MFCCs (Mel-Frequency Cepstral Coefficients) per audio frame,
  padded/truncated to a fixed length (130 frames, ~3 seconds) so every clip has the same
  shape.
- **Model:** a 2-layer LSTM (128 → 64 units, with Dropout) that reads the MFCCs as a time
  series, followed by a Dense classification head.

## Pipeline
1. Collect all `.wav` file paths under `DATA_DIR` and parse each one's emotion label from
   its filename.
2. Extract and pad/truncate MFCCs for every clip; normalize the resulting feature matrix.
3. Label-encode and one-hot encode the emotion labels.
4. Split into train/test sets (80/20, stratified).
5. Build and train the LSTM model (40 epochs, batch size 16, 10% validation split).
6. Evaluate on the test set: accuracy, a full classification report, and a confusion
   matrix.

## Requirements
```
numpy
librosa
scikit-learn
tensorflow
```
Install with:
```
pip install numpy librosa scikit-learn tensorflow
```

## How to Run
```
python speech_emotion_recognition.py
```

## Output
Model summary, training progress per epoch, then test accuracy, a per-emotion
classification report (precision/recall/F1), and a confusion matrix.

## Note
The script was verified end-to-end on synthetic dummy audio (correct file naming, no
runtime errors) before being finalized. Actual accuracy depends on running it against the
real RAVDESS dataset.
