
# * ==========================================
# * 6. Preprocess Data and Create Labels
# * ==========================================
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from config import actions, no_sequences, sequence_length, DATA_PATH

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

# * ==================================================
# * 7. Build and Train an LSTM Deep Learning Model
# * ==================================================

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# * Adding Early Stopping (monitors validation loss)
early_stop_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# * Learning Rate Scheduler (Reduces learning rate when validation accuracy plateaus)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

# * Defining the model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
model.add(Dropout(0.2))  # Dropout layer to prevent overfitting
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(Dropout(0.2))  # Dropout layer to prevent overfitting
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# * Fitting the model with added callbacks
model.fit(x_train, y_train, epochs=2000, validation_data=(x_test, y_test),
          callbacks=[tb_callback, early_stop_callback, lr_scheduler])
