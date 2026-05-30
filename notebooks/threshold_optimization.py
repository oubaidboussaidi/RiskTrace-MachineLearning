import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score

def optimize_threshold(y_true, normalized_scores):
    """
    Sweeps through anomaly score thresholds from 0.01 to 0.99 
    to find the mathematical argmax of the F1-Score.
    """
    thresholds = np.arange(0.01, 1.00, 0.01)
    
    f1_scores = []
    precisions = []
    recalls = []

    for t in thresholds:
        y_pred = (normalized_scores >= t).astype(int)
        
        f1 = f1_score(y_true, y_pred, zero_division=0)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        
        f1_scores.append(f1)
        precisions.append(p)
        recalls.append(r)

    # Find the peak of the F1-Score
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    print(f"--- THRESHOLD OPTIMIZATION RESULTS ---")
    print(f"Optimal Threshold : {best_threshold:.2f}")
    print(f"Max F1-Score      : {best_f1:.4f}")
    print(f"Precision at {best_threshold:.2f}  : {precisions[best_idx]:.4f}")
    print(f"Recall at {best_threshold:.2f}     : {recalls[best_idx]:.4f}")

    # Plot the optimization curve
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, label='F1-Score', color='purple', linewidth=2.5)
    plt.plot(thresholds, precisions, label='Precision', color='blue', linestyle='--')
    plt.plot(thresholds, recalls, label='Recall', color='green', linestyle='--')
    
    # Mark the exact elbow point / peak
    plt.axvline(x=best_threshold, color='red', linestyle=':', 
                label=f'Optimal Threshold = {best_threshold:.2f}')
    
    plt.title('Isolation Forest Threshold Optimization (Argmax of F1-Score)')
    plt.xlabel('Anomaly Score Threshold')
    plt.ylabel('Score Metric')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.savefig('optimization_curve.png')
    # plt.show()

# Example mock execution (how it would run in your pipeline):
if __name__ == "__main__":
    print("Running threshold sweeping simulation...\n")
    
    # Simulating 50,000 traffic logs (1% anomaly rate)
    np.random.seed(42)
    n_normal = 49500
    n_anomaly = 500
    
    y_true = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)])
    
    # Simulating Isolation Forest sigmoid-transformed scores
    # Normal traffic clusters around 0.25, anomalies cluster around 0.80
    scores_normal = np.random.normal(0.25, 0.15, n_normal)
    scores_anomaly = np.random.normal(0.80, 0.08, n_anomaly)
    
    normalized_scores = np.clip(np.concatenate([scores_normal, scores_anomaly]), 0, 1)
    
    optimize_threshold(y_true, normalized_scores)
