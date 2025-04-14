
# * ==========================================
# * 6. Preprocess Data and Create Labels
# * ==========================================
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from config import (
    actions, no_sequences, sequence_length, DATA_PATH
)


# * Create a dictionary mapping each action to a label number like this: {'A': 0, 'B': 1, ..., 'space': 26, 'end': 27, 'delete': 28}
label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
# * Loop through each action (A-Z, space, end, delete), sequence (e.g., 30 sequences per letter), and frame (30 per sequence) to load data
# * which holds landmark keypoints,Builds a window (a list of 30 frames),And adds it to the final sequences list.

for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

# * Convert to numpy arrays: X = sequences, y = labels (one-hot encoded)
x = np.array(sequences)
y = to_categorical(labels).astype(int)

# * Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)


# * 7. Build and Train an LSTM Deep Learning Model