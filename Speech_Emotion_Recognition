# TASK 2: Emotion Recognition from Speech
# Objective: Recognize human emotions (e.g., happy, angry, sad) from speech audio.
# Dataset: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
#
# Before running: download RAVDESS "Audio_Speech_Actors_01-24.zip" from
# https://zenodo.org/record/1188976, unzip it, and set DATA_DIR below to the
# folder that contains the Actor_01 ... Actor_24 subfolders.

import os
import glob
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 0. Settings
DATA_DIR = "ravdess_data"      # folder containing Actor_01 ... Actor_24
N_MFCC = 40                    # number of MFCC coefficients per frame
MAX_FRAMES = 130               # fixed number of time frames per clip (~3 sec)
SAMPLE_RATE = 22050

# RAVDESS filename convention: the 3rd number is the emotion code
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# 1. Collect file paths and labels
file_paths = glob.glob(os.path.join(DATA_DIR, "**", "*.wav"), recursive=True)
print(f"Found {len(file_paths)} audio files")

labels = []
for path in file_paths:
    filename = os.path.basename(path)
    emotion_code = filename.split("-")[2]
    labels.append(emotion_map[emotion_code])

# 2. Extract MFCC features
# Each clip becomes a (MAX_FRAMES, N_MFCC) matrix so the LSTM can read it
# as a sequence of frames over time, the same way it reads speech.
features = []
for path in file_paths:
    audio, sr = librosa.load(path, sr=SAMPLE_RATE, duration=3, offset=0.4)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    mfcc = mfcc.T  # shape becomes (time_frames, N_MFCC)

    # pad or truncate every clip to the same number of frames
    if mfcc.shape[0] < MAX_FRAMES:
        pad_width = MAX_FRAMES - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode="constant")
    else:
        mfcc = mfcc[:MAX_FRAMES, :]

    features.append(mfcc)

X = np.array(features)
print("Feature matrix shape:", X.shape)  # (num_samples, MAX_FRAMES, N_MFCC)

# Normalize features (helps the LSTM train faster and more stably)
mean = X.mean()
std = X.std()
X = (X - mean) / std

# 3. Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)
y_categorical = to_categorical(y_encoded)
num_classes = y_categorical.shape[1]
print("Classes:", list(label_encoder.classes_))

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
)

# 5. Build the LSTM model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(MAX_FRAMES, N_MFCC)),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(num_classes, activation="softmax"),
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# 6. Train
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=40,
    batch_size=16,
    verbose=1,
)

# 7. Evaluate
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nTest Accuracy:", accuracy_score(y_true, y_pred))
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
