import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Define dataset path
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

# Define class names
CLASSES = ['Bulging_Eyes', 'Cataracts', 'Glaucoma', 'Uveitis']
print(f"Available classes: {CLASSES}")

# Constants
IMG_SIZE = 224
BATCH_SIZE = 16  # Reduced batch size
EPOCHS = 50      # Increased epochs
LEARNING_RATE = 0.0001  # Reduced learning rate

# Enhanced Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest',
    validation_split=0.2,
    brightness_range=[0.8, 1.2],
    shear_range=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.2
)

# Training Data
train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    classes=CLASSES,
    subset="training",
    shuffle=True
)

# Validation Data
val_data = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    classes=CLASSES,
    subset="validation",
    shuffle=False
)

# Print class distribution in training data and calculate class weights
class_counts = {}
total_samples = len(train_data.labels)
n_classes = len(CLASSES)

print("\nClass distribution in training data:")
for class_name, class_index in train_data.class_indices.items():
    n_samples = len([1 for i in range(len(train_data.labels)) if train_data.labels[i] == class_index])
    print(f"{class_name}: {n_samples} images")
    class_counts[class_index] = n_samples if n_samples > 0 else 1  # Avoid division by zero

# Calculate balanced class weights
class_weights = {}
max_samples = max(class_counts.values())
for class_index, count in class_counts.items():
    class_weights[class_index] = max_samples / count

print("\nClass weights:")
for class_index, weight in class_weights.items():
    print(f"{CLASSES[class_index]}: {weight:.2f}")

# Load Pretrained EfficientNet Model
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Fine-tune the model
base_model.trainable = True
for layer in base_model.layers[:-30]:  # Freeze more layers
    layer.trainable = False

# Enhanced Model Architecture
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
# Final layer without activation
logits = Dense(len(CLASSES))(x)
# Apply softmax separately
outputs = tf.keras.layers.Activation('softmax')(logits)

# Create Model
model = Model(inputs=base_model.input, outputs=outputs)

# Compile with a lower learning rate
optimizer = Adam(learning_rate=LEARNING_RATE)
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=["accuracy", tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Print model summary
model.summary()

# Enhanced callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Train Model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights  # Use calculated class weights
)

# Save final model
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_model.h5")
model.save(model_path)
print(f"\nModel saved as '{model_path}'")

# Print final metrics
print("\nFinal Training Metrics:")
for metric in history.history:
    if not metric.startswith('val_'):
        print(f"{metric}: {history.history[metric][-1]:.4f}")
        if f'val_{metric}' in history.history:
            print(f"val_{metric}: {history.history[f'val_{metric}'][-1]:.4f}")

# Plot training history
metrics_to_plot = ['accuracy', 'loss', 'auc', 'precision', 'recall']
n_metrics = len(metrics_to_plot)
plt.figure(figsize=(15, 4 * ((n_metrics + 1) // 2)))

for i, metric in enumerate(metrics_to_plot, 1):
    plt.subplot((n_metrics + 1) // 2, 2, i)
    plt.plot(history.history[metric], label=f'Training {metric}')
    plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
    plt.title(f'Model {metric}')
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.close()
