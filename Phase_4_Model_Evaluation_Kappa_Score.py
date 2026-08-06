import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    cohen_kappa_score,
    accuracy_score,
    precision_recall_fscore_support
)
import seaborn as sns
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

DEFECT_NAMES_LIST = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch', 'none']
DEFECT_NAMES_DICT = {i: name for i, name in enumerate(DEFECT_NAMES_LIST)}

def load_model_and_data():
    print("="*80)
    print("STEP 1: LOADING MODEL AND DATA")
    print("="*80)
    try:
        model = keras.models.load_model('best_wafer_model.h5')
        print("\n✓ Model loaded: best_wafer_model.h5")
        
        X_test = np.load('X_test.npy')
        y_test = np.load('y_test.npy')
        
        print(f"Detected full test array dimensions: {X_test.shape}")
        X_test = X_test.astype('float32') / 255.0 if X_test.max() > 1 else X_test.astype('float32')
        
        return model, X_test, y_test
    except FileNotFoundError as e:
        print(f"\n⚠️  Error: {e}")
        return None, None, None

def make_predictions_batched(model, X_test, batch_size=256):
    print("\n" + "="*80)
    print("STEP 2: GENERATING PREDICTIONS (MEMORY-SAFE BATCHED)")
    print("="*80)
    y_pred_proba = model.predict(X_test, batch_size=batch_size, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    return y_pred, y_pred_proba

def calculate_metrics(y_true, y_pred):
    print("\n" + "="*80)
    print("STEP 3 & 4: COMPREHENSIVE EVALUATION METRICS")
    print("="*80)
    
    accuracy = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    print(f"\n📊 OVERALL METRICS FOR THE FULL TEST DATASET:")
    print(f"  Accuracy:     {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Kappa Score:  {kappa:.4f}")
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(9)), average=None, zero_division=0
    )
    
    print(f"\n🎯 PER-CLASS METRICS:")
    print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 60)
    
    for i in range(9):
        class_name = DEFECT_NAMES_DICT[i]
        print(f"{class_name:<12} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {int(support[i]):<10}")
        
    return kappa

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(9)))
    plt.figure(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=DEFECT_NAMES_LIST, yticklabels=DEFECT_NAMES_LIST)
    plt.title('Confusion Matrix - Full Test Dataset Evaluation')
    plt.ylabel('True Class Labels')
    plt.xlabel('Predicted Class Labels')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    print("\n✓ Confusion matrix plot exported to file: confusion_matrix.png")

if __name__ == "__main__":
    model, X_test, y_test = load_model_and_data()
    if model is not None:
        y_pred, y_pred_proba = make_predictions_batched(model, X_test)
        calculate_metrics(y_test, y_pred)
        plot_confusion_matrix(y_test, y_pred)
        print("\n✅ Evaluation on Full Test Array Complete!")