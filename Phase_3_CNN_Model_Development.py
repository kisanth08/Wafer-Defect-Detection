import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
import warnings
import os
import random
warnings.filterwarnings('ignore')

def load_prepared_data():
    X_train = np.load('X_train.npy')
    X_val = np.load('X_val.npy')
    X_test = np.load('X_test.npy')
    y_train = np.load('y_train.npy')
    y_val = np.load('y_val.npy')
    y_test = np.load('y_test.npy')
    return (X_train, X_val, X_test), (y_train, y_val, y_test)

def normalize_data(X_train, X_val, X_test):
    X_train = X_train.astype('float32') / 255.0 if X_train.max() > 1 else X_train.astype('float32')
    X_val = X_val.astype('float32') / 255.0 if X_val.max() > 1 else X_val.astype('float32')
    X_test = X_test.astype('float32') / 255.0 if X_test.max() > 1 else X_test.astype('float32')
    return X_train, X_val, X_test

def build_improved_cnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0005),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def setup_callbacks():
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    model_checkpoint = callbacks.ModelCheckpoint('best_wafer_model.h5', monitor='val_loss', 
                                                 mode='min', save_best_only=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    return [early_stop, model_checkpoint, reduce_lr]

def augmented_balanced_generator(X, y, batch_size=64, total_classes=9):
    class_indices = {c: np.where(y == c)[0] for c in range(total_classes)}
    valid_classes = [c for c in range(total_classes) if len(class_indices[c]) > 0]
    samples_per_class = max(1, batch_size // len(valid_classes))
    
    while True:
        batch_X, batch_y = [], []
        for c in valid_classes:
            indices = np.random.choice(class_indices[c], size=samples_per_class, replace=True)
            images = X[indices]
            
            augmented_images = []
            for img in images:
                rot_k = random.choice([0, 1, 2, 3])
                img_aug = np.rot90(img, k=rot_k)
                if random.random() > 0.5:
                    img_aug = np.flipud(img_aug)
                augmented_images.append(img_aug)
                
            batch_X.append(np.array(augmented_images))
            batch_y.append(y[indices])
            
        X_out = np.vstack(batch_X)
        y_out = np.concatenate(batch_y)
        shuffle_idx = np.random.permutation(len(y_out))
        yield X_out[shuffle_idx], y_out[shuffle_idx]

if __name__ == "__main__":
    if os.path.exists('best_wafer_model.h5'):
        os.remove('best_wafer_model.h5')

    (X_train, X_val, X_test), (y_train, y_val, y_test) = load_prepared_data()
    X_train, X_val, X_test = normalize_data(X_train, X_val, X_test)
    
    num_classes = 9 
    model = build_improved_cnn(input_shape=X_train.shape[1:], num_classes=num_classes)

    print("\n================================================================================")
    print("RUNNING HIGH-STABILITY GRADIENT OVERSAMPLING WITH PROPORTIONAL VALIDATION")
    print("================================================================================")
    
    
    X_train_sub, _, y_train_sub, _ = train_test_split(
        X_train, y_train, train_size=min(30000, len(X_train))/len(X_train), random_state=42, stratify=y_train
    )
    
    X_val_sub, _, y_val_sub, _ = train_test_split(
        X_val, y_val, train_size=min(6000, len(X_val))/len(X_val), random_state=42, stratify=y_val
    )
    
    train_gen = augmented_balanced_generator(X_train_sub, y_train_sub, batch_size=128, total_classes=num_classes)
    
    print("🚀 Training model tracking real error drop patterns...")
    history = model.fit(
        train_gen,
        steps_per_epoch=len(X_train_sub) // 128,
        validation_data=(X_val_sub, y_val_sub), 
        epochs=30,
        callbacks=setup_callbacks(),
        verbose=2
    )

    print("✅ Phase 3 Model Training Complete.")