
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath):
    print("="*80)
    print("STEP 1: LOADING DATASET")
    print("="*80)
    
    print(f"Loading dataset from {filepath}...")
    df = pd.read_pickle(filepath)
    print("✓ Dataset loaded successfully!")
    print(f"Total initial records: {len(df):,}")
    return df

def clean_and_explore(df):
    print("\n" + "="*80)
    print("STEP 2 & 3: DATA CLEANING AND EDA")
    print("="*80)
    
    df['failureType'] = df['failureType'].apply(lambda x: x[0][0] if len(x) > 0 else 'none')
    valid_labels = ['none', 'Loc', 'Edge-Loc', 'Center', 'Edge-Ring', 
                    'Scratch', 'Random', 'Near-full', 'Donut']
    
    df_clean = df[df['failureType'].isin(valid_labels)].copy()
    print(f"Records after cleaning: {len(df_clean):,}")
    
    le = LabelEncoder()
    y = le.fit_transform(df_clean['failureType'])
    return df_clean, y, le

def process_wafer_maps(wafer_series, target_size=(64, 64)):
    print("\n" + "="*80)
    print("STEP 4: WAFER MAP IMAGE PROCESSING")
    print("="*80)
    
    resized_images = []
    for w_map in wafer_series:
        if w_map.size == 0:
            resized = np.zeros(target_size)
        else:
            resized = cv2.resize(w_map, target_size, interpolation=cv2.INTER_NEAREST)
        resized_images.append(resized)
            
    X = np.expand_dims(np.array(resized_images), axis=-1)
    return X

def get_class_weights(y):
    print("\n" + "="*80)
    print("STEP 5: HANDLING CLASS IMBALANCE")
    print("="*80)
    
    classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    class_weights = dict(zip(classes, weights))
    return class_weights


if __name__ == "__main__":
    FILE_PATH = r'E:\Dataset\LSWMD.pkl\LSWMD.pkl' 
    
    try:
        
        df = load_data(FILE_PATH)
        df_clean, y, le = clean_and_explore(df)
        X = process_wafer_maps(df_clean['waferMap'])
        class_weights = get_class_weights(y)
        
        
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, random_state=42,
                                                           stratify=y_temp)
        
       
        print("\nSaving processed data and class weights...")
        np.save('X_train.npy', X_train)
        np.save('X_val.npy', X_val)
        np.save('X_test.npy', X_test)
        np.save('y_train.npy', y_train)
        np.save('y_val.npy', y_val)
        np.save('y_test.npy', y_test)
        np.save('class_weights.npy', class_weights)
        
        print("\n✅ Phase 2 Complete! Data and weights saved.")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
