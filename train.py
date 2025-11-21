import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_and_preprocess_data(csv_file):
    """Load and preprocess the engine data"""
    print("Loading dataset...")
    df = pd.read_csv(csv_file)

    # Define feature columns (exclude time and target)
    feature_columns = [
        'Average speed (km/h)',
        'Barometric pressure (kPa)',
        'Calculated boost (bar)',
        'Calculated engine load value (%)',
        'Distance travelled (km)',
        'Engine coolant temperature (℃)',
        'Engine RPM (rpm)',
        'Engine RPM x1000 (rpm)',
        'Intake manifold absolute pressure (kPa)',
        'Long term fuel % trim - Bank 1 (%)',
        'OBD Module Voltage (V)',
        'Short term fuel % trim - Bank 1 (%)',
        'Vehicle acceleration (g)',
        'Vehicle speed (km/h)'
    ]

    # Extract features and labels
    X = df[feature_columns].values
    y = df['anomaly_type'].values

    # Handle any NaN or infinite values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Class distribution: {np.bincount(y)}")

    return X, y, feature_columns

def create_sequences(X, y, sequence_length=10):
    """Create sequences for temporal pattern detection"""
    X_seq = []
    y_seq = []

    for i in range(len(X) - sequence_length + 1):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y[i+sequence_length-1])

    return np.array(X_seq), np.array(y_seq)

def build_model(input_shape, num_classes):
    """Build deep learning model for anomaly detection"""
    model = models.Sequential([
        # LSTM layers for temporal pattern recognition
        layers.LSTM(128, return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.3),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(32),
        layers.Dropout(0.2),

        # Dense layers
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

def convert_to_tflite(model, model_name='engine_anomaly_detector'):
    """Convert trained model to TFLite format"""
    print("\nConverting model to TFLite format...")

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Optimization for mobile/edge devices
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    # Fix for 'tf.TensorListReserve' op requiring static element_shape
    # This allows for TensorFlow ops that do not have a TFLite equivalent
    # and disables an experimental lowering of tensor list ops.
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter._experimental_lower_tensor_list_ops = False

    tflite_model = converter.convert()

    # Save the TFLite model
    tflite_filename = f'{model_name}.tflite'
    with open(tflite_filename, 'wb') as f:
        f.write(tflite_model)

    print(f"TFLite model saved as: {tflite_filename}")
    print(f"Model size: {len(tflite_model) / 1024:.2f} KB")

    return tflite_filename

def train_model(csv_file='peugeot_208_engine_data.csv',
                sequence_length=10,
                batch_size=64,
                epochs=50):
    """Complete training pipeline"""

    # Load data
    X, y, feature_columns = load_and_preprocess_data(csv_file)

    # Split data
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Normalize features
    print("Normalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler for inference
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler saved as: scaler.pkl")

    # Create sequences
    print(f"Creating sequences (length={sequence_length})...")
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, sequence_length)

    print(f"Training sequences shape: {X_train_seq.shape}")
    print(f"Testing sequences shape: {X_test_seq.shape}")

    # Convert labels to categorical
    num_classes = len(np.unique(y))
    y_train_cat = keras.utils.to_categorical(y_train_seq, num_classes)
    y_test_cat = keras.utils.to_categorical(y_test_seq, num_classes)

    # Build model
    print("\nBuilding model...")
    input_shape = (sequence_length, X_train_seq.shape[2])
    model = build_model(input_shape, num_classes)

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )

    print("\nModel Summary:")
    model.summary()

    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    )

    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train_seq, y_train_cat,
        validation_split=0.2,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # Evaluate model
    print("\nEvaluating model...")
    test_results = model.evaluate(X_test_seq, y_test_cat, verbose=0)
    print(f"Test Loss: {test_results[0]:.4f}")
    print(f"Test Accuracy: {test_results[1]:.4f}")
    print(f"Test Precision: {test_results[2]:.4f}")
    print(f"Test Recall: {test_results[3]:.4f}")

    # Get predictions for detailed metrics
    y_pred = model.predict(X_test_seq, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Per-class accuracy
    print("\nPer-class accuracy:")
    for i in range(num_classes):
        mask = y_test_seq == i
        if mask.sum() > 0:
            class_acc = (y_pred_classes[mask] == i).mean()
            print(f"Class {i}: {class_acc:.4f}")

    # Confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_test_seq, y_pred_classes)
    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    class_names = ['Normal', 'Weak Injectors', 'Fuel Leak', 'Oil Leak', 'Vacuum Leak']
    print(classification_report(y_test_seq, y_pred_classes, target_names=class_names))

    # Save metadata
    metadata = {
        'feature_columns': feature_columns,
        'sequence_length': sequence_length,
        'num_classes': num_classes,
        'class_names': class_names,
        'input_shape': input_shape
    }

    with open('model_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    print("Metadata saved as: model_metadata.pkl")

    # Convert to TFLite
    tflite_filename = convert_to_tflite(model)

    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)
    print(f"TFLite model: {tflite_filename}")
    print(f"Scaler: scaler.pkl")
    print(f"Metadata: model_metadata.pkl")
    print("\nAnomaly Detection Classes:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")

    return model, history, tflite_filename

if __name__ == "__main__":
    print("="*60)
    print("Peugeot 208 Engine Anomaly Detection Training")
    print("="*60)

    # Check if data file exists
    import sys
    if not os.path.exists('peugeot_208_engine_data.csv'):
        print("\nError: 'peugeot_208_engine_data.csv' not found!")
        print("Please run the data generation script first.")
        sys.exit(1)

    # Train model
    model, history, tflite_file = train_model(
        csv_file='peugeot_208_engine_data.csv',
        sequence_length=10,
        batch_size=64,
        epochs=50
    )

    print("\n✓ Training pipeline completed successfully!")
    print(f"✓ TFLite model ready for deployment: {tflite_file}")

