import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, TimeDistributed, BatchNormalization
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Dataset path
dataset_path = "archive/real_and_fake_face"
img_size = (128, 128)
sequence_length = 10  # Number of frames per sequence

# Function to load dataset with sequences
def load_dataset(directory, sequence_length):
    sequences = []
    labels = []
    classes = {'training_real': 0, 'training_fake': 1}  # Label Mapping
    
    for label, class_index in classes.items():
        class_path = os.path.join(directory, label)
        files = sorted(os.listdir(class_path))  # Sorting ensures proper sequence
        
        for i in range(len(files) - sequence_length):
            seq = []
            for j in range(sequence_length):
                img_path = os.path.join(class_path, files[i + j])
                img = load_img(img_path, target_size=img_size)
                img = img_to_array(img) / 255.0
                seq.append(img)
            
            sequences.append(seq)
            labels.append(class_index)
    
    return np.array(sequences), np.array(labels)

# Load dataset
X, y = load_dataset(dataset_path, sequence_length)

# Train-Test Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert labels to categorical
y_train = to_categorical(y_train, 2)
y_val = to_categorical(y_val, 2)

# CNN Model for Feature Extraction
cnn_model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2,2),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    BatchNormalization(),
    Flatten(),
    Dense(128, activation='relu'),
    """_summary_
    """    Dropout(0.5)
])

# CNN + RNN Model
model = Sequential([
    TimeDistributed(cnn_model, input_shape=(sequence_length, 128, 128, 3)),  # Apply CNN to each frame
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(2, activation='softmax')  # Output 2 classes: Real & Fake
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=8)

# Save Model
model.save("deepfake_model.h5")
