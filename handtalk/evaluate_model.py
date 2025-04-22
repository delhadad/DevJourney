from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
import numpy as np
from tensorflow.keras.models import load_model
from train_model import x_test, y_test, x_train, y_train  # No training happens here

# * ==========================================
# * 10. Evaluation using a Confusion Matrix
# * ==========================================

model = load_model('action.h5')

# For Training Data
yhat_train = model.predict(x_train)
ytrue_train = np.argmax(y_train, axis=1).tolist()
yhat_train = np.argmax(yhat_train, axis=1).tolist()

# Multilabel confusion matrix and accuracy score for training data
conf_matrix_train = multilabel_confusion_matrix(ytrue_train, yhat_train)
accuracy_train = accuracy_score(ytrue_train, yhat_train)

print("Training Data:")
print("Confusion Matrix:")
print(conf_matrix_train)
print(f'Accuracy Score: {accuracy_train}')

# For Test Data
yhat_test = model.predict(x_test)
ytrue_test = np.argmax(y_test, axis=1).tolist()
yhat_test = np.argmax(yhat_test, axis=1).tolist()

# Multilabel confusion matrix and accuracy score for test data
conf_matrix_test = multilabel_confusion_matrix(ytrue_test, yhat_test)
accuracy_test = accuracy_score(ytrue_test, yhat_test)

print("Test Data:")
print("Confusion Matrix:")
print(conf_matrix_test)
print(f'Accuracy Score: {accuracy_test}')
