import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, TimeDistributed, Flatten
import matplotlib.pyplot as plt

output_folder = r"C:\Users\pavan\OneDrive\Desktop\miniproject\output"
num_frames = 1 
height, width, channels = 64, 64, 3  # Dimensions of each frame
epochs = 10
batch_size = 32
accident_frames = np.load(os.path.join(output_folder, "accident_preprocessed_frames.npy"))
non_accident_frames = np.load(os.path.join(output_folder, "non_accident_preprocessed_frames.npy"))
accident_labels = np.load(os.path.join(output_folder, "accident_labels.npy"))
non_accident_labels = np.load(os.path.join(output_folder, "non_accident_labels.npy"))
all_frames = np.concatenate([accident_frames, non_accident_frames], axis=0)
all_labels = np.concatenate([np.ones(len(accident_frames)), np.zeros(len(non_accident_frames))], axis=0)
X_train, X_test, y_train, y_test = train_test_split(all_frames, all_labels, test_size=0.2, random_state=42)
X_train = X_train.reshape(-1, num_frames, height, width, channels)
X_test = X_test.reshape(-1, num_frames, height, width, channels)
model = Sequential([
    TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'), input_shape=(num_frames, height, width, channels)),
    TimeDistributed(MaxPooling2D((2, 2), padding='same')),
    TimeDistributed(Conv2D(8, (3, 3), activation='relu', padding='same')),
    TimeDistributed(MaxPooling2D((2, 2), padding='same')),
    TimeDistributed(Conv2D(8, (3, 3), activation='relu', padding='same')),
    TimeDistributed(MaxPooling2D((2, 2), padding='same')),
    TimeDistributed(Flatten()),
    LSTM(32, return_sequences=True),
    LSTM(32),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
sample_frames = np.concatenate([accident_frames[:100], non_accident_frames[:100]], axis=0)
mean_pixel_values = np.mean(sample_frames, axis=(1, 2, 3)) 
plt.figure(figsize=(8, 6))
plt.boxplot([mean_pixel_values[:100], mean_pixel_values[100:]], labels=['Accident', 'Non-Accident'])
plt.title('Distribution of Mean Pixel Values')
plt.xlabel('Class')
plt.ylabel('Mean Pixel Value')
plt.grid(True)
plt.show()
model.save(os.path.join(output_folder, "lstm-anomaly_detection_model.h5"))
