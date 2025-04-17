import os
import numpy as np

# * ==========================================
# * 4. Setup Folders for Data Collection
# * ==========================================



DATA_PATH = os.path.join('MP_Data') # path for exported data, numpy arrays
actions = np.array([
    'hello', 'thanks', 'my', 'name',
    'nice', 'meet', 'end', 'delete',
    'A', 'D', 'U'
]) #actions that we try to detect
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
