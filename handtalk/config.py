import os
import numpy as np

# * ==========================================
# * 4. Setup Folders for Data Collection
# * ==========================================



DATA_PATH = os.path.join('MP_Data') # path for exported data, numpy arrays
actions = np.array([chr(i) for i in range(65, 91)] + ['space', 'end' , 'delete']) #actions that we try to detect
print(actions)
no_sequences = 30 #thirty videos worth of data, Each action (gesture) will have 30 recorded videos
sequence_length = 30  #videos are going to be 30 frames in length, Each recorded video clip will contain 30 frames (images).

# * Create directories
for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass
