import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load Data
X = np.load("X.npy")   # (347, 63)
y = np.load("y.npy")   # (347,)

# One-hot encode labels
y = to_categorical(y)

# Reshape X for LSTM input: (samples, sequence_length, features)
# Since we only have single frames now, sequence_length = 1
X = X.reshape(X.shape[0], 1, X.shape[1])  # (347, 1, 63)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# Build LSTM Model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(1, 63)))
model.add(LSTM(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

# Save the model
model.save("sign_model.h5")

print("✨ Model Training Complete! Model saved as sign_model.h5 ✅")
