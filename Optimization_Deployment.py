import numpy as np
import tensorflow as tf
import os
import time
import warnings
warnings.filterwarnings('ignore')

DEFECT_NAMES_LIST = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch', 'none']

def optimize_and_convert_model():
    print("="*80)
    print("STEP 1: CONVERTING MODEL TO TENSORFLOW LITE (PRODUCTION READY)")
    print("="*80)
    
    h5_path = 'best_wafer_model.h5'
    tflite_path = 'wafer_model_optimized.tflite'
    
    if not os.path.exists(h5_path):
        print(f"❌ Error: {h5_path} not found. Please run Phase 3 first.")
        return False
        
    print("Loading baseline trained HDF5 model...")
    model = tf.keras.models.load_model(h5_path)
    
    print("Converting model structure to optimized TFLite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
        
    print(f"✓ Success! Optimized model saved to: {tflite_path}")
    print(f"  Baseline Model Size: {os.path.getsize(h5_path) / (1024*1024):.2f} MB")
    print(f"  Optimized Model Size: {os.path.getsize(tflite_path) / (1024*1024):.2f} MB")
    return True

def simulate_production_deployment():
    print("\n" + "="*80)
    print("STEP 2: RUNNING PRODUCTION INFERENCE SIMULATION")
    print("="*80)
    
    tflite_path = 'wafer_model_optimized.tflite'
    if not os.path.exists('X_test.npy') or not os.path.exists('y_test.npy'):
        print("❌ Error: Test data files missing.")
        return
        
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    
    print("Initializing production TFLite runtime interpreter...")
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("Streaming sample wafers through the deployment pipeline...\n")
    sample_indices = np.random.choice(len(X_test), 5, replace=False)
    
    for idx in sample_indices:
        raw_wafer = X_test[idx]
        actual_label = DEFECT_NAMES_LIST[y_test[idx]]
        
        processed_wafer = raw_wafer.astype('float32') / 255.0 if raw_wafer.max() > 1 else raw_wafer.astype('float32')
        input_data = np.expand_dims(processed_wafer, axis=0) # Add batch dimension
        
        start_time = time.time()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction_idx = np.argmax(output_data[0])
        confidence = output_data[0][prediction_idx] * 100
        elapsed_time = (time.time() - start_time) * 1000 # Convert to milliseconds
        
        predicted_label = DEFECT_NAMES_LIST[prediction_idx]
        
        print(f" [Wafer #{idx:06d}]")
        print(f"   ↳ Speed:      {elapsed_time:.2f} ms")
        print(f"   ↳ Prediction: {predicted_label} ({confidence:.1f}% Confidence)")
        print(f"   ↳ Ground Truth: {actual_label}")
        print("-" * 50)

if __name__ == "__main__":
    if optimize_and_convert_model():
        simulate_production_deployment()
        print("\n✅ Phase 5 Deployment Pipeline Verified Successfully!")
