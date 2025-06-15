import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# === 1. Load Excel File ===
# Make sure this Excel file is in the same folder as your script
df = pd.read_excel("emotion_data.xlsx")

# Ensure consistent column names
df.columns = ["force", "loudness", "tone", "label"]

# === 2. Prepare Data ===
X = df[["force", "loudness", "tone"]].values.astype(np.float32)
y = df["label"].values.astype(int)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# One-hot encode labels for 17 emotion classes
y_encoded = tf.keras.utils.to_categorical(y, num_classes=17)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# === 3. Build Model ===
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(3,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(17, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === 4. Train Model ===
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=50, batch_size=8)

# === 5. Evaluate Model ===
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy:.4f}")

# === 6. Save Full Model (.h5) ===
model.save("emotion_model.h5")

# === 7. Export to TFLite ===
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("emotion_model.tflite", "wb") as f:
    f.write(tflite_model)

# === 8. Save Scaler Parameters ===
np.savetxt("scaler_mean.txt", scaler.mean_, fmt="%.6f")
np.savetxt("scaler_scale.txt", scaler.scale_, fmt="%.6f")

# === 9. (Optional) Plot Training Progress ===
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss")
plt.legend()
plt.tight_layout()
plt.show()
