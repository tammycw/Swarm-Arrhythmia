# Particle Swarm Optimization (PSO)
# PSO Training of a Neural Network on MIT-BIH Arrhythmia Dataset
# The MIT-BIH Arrhythmia Database is a standard benchmark in biomedical engineering for ECG heartbeat classification. 
# It contains ~110,000 annotated beats from 47 subjects, 
# with classes like Normal (N), Ventricular Ectopic (V), Supraventricular (S), Fusion (F), etc.

# Use wfdb to load a few records.
# Extract fixed-length heartbeat segments (common practice).
# Use a small MLP (Multi-Layer Perceptron) on flattened segments or basic features.
# Train the network weights using Particle Swarm Optimization (PSO).

import builtins
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wfdb

log_path = "training_output.txt"
log_file = open(log_path, "w", encoding="utf-8")
_original_print = builtins.print


def tee_print(*args, **kwargs):
    kwargs_copy = dict(kwargs)
    kwargs_copy.pop("file", None)
    _original_print(*args, **kwargs_copy)
    _original_print(*args, file=log_file, **kwargs_copy)


builtins.print = tee_print
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Choose data source:
# - "excel" : read cached data from mitbih_data.xlsx
# - "database" : reload data from the MIT-BIH source (PhysioNet)
data_source = "excel"  # change to "database" to reload from MIT-BIH
excel_path = "mitbih_data.xlsx"

def compute_classification_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true.astype(int).ravel(), y_pred.astype(int).ravel())
    if cm.shape[0] != 2 or cm.shape[1] != 2:
        raise ValueError("Confusion matrix must be 2x2 for binary classification")

    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    balanced_accuracy = 0.5 * (recall + specificity)

    total = tp + tn + fp + fn
    expected_true = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / total if total else 0.0
    kappa = (accuracy - expected_true) / (1 - expected_true) if (1 - expected_true) else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "kappa": kappa,
        "confusion_matrix": cm,
    }


def print_metrics(label, y_true, y_pred):
    metrics = compute_classification_metrics(y_true, y_pred)
    print(f"{label}:")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall (Sensitivity): {metrics['recall']:.3f}")
    print(f"  Specificity: {metrics['specificity']:.3f}")
    print(f"  F1-score: {metrics['f1']:.3f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")
    print(f"  Kappa: {metrics['kappa']:.3f}")
    print("  Confusion Matrix:")
    print(metrics['confusion_matrix'])
    print("  Labels: [0=Normal, 1=Abnormal]")

# ====================== 1. Load MIT-BIH Data ======================
def load_mit_bih_data(record_numbers=[100, 101, 102, 103, 105, 106, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 200, 201, 202, 203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 217, 219, 220, 221, 222, 223, 228, 230, 231, 232, 233, 234], samples_per_beat=180, max_samples=12000):
    X = []
    y = []
    record_ids = []
    beat_classes = []
    
    for rec in record_numbers:
        try:
            # Download from PhysioNet if not present
            record = wfdb.rdrecord(str(rec), pn_dir='mitdb')
            annotation = wfdb.rdann(str(rec), 'atr', pn_dir='mitdb')
            
            signal = record.p_signal[:, 0]  # Use MLII lead (common)
            
            for i, symbol in enumerate(annotation.symbol):
                if symbol in ['N', 'V', 'S', 'F']:  # Focus on main classes: Normal (N), Ventricular Ectopic (V), Supraventricular (S), Fusion (F)
                    center = annotation.sample[i]
                    half = samples_per_beat // 2
                    
                    if center - half >= 0 and center + half < len(signal):
                        segment = signal[center - half:center + half]
                        X.append(segment)
                        # Map to classes: 0=Normal, 1=Abnormal
                        label = 0 if symbol == 'N' else 1
                        y.append(label)
                        record_ids.append(rec)
                        beat_classes.append(symbol)

                        if max_samples is not None and len(X) >= max_samples:
                            break
            if max_samples is not None and len(X) >= max_samples:
                break
        except Exception as e:
            print(f"Error loading record {rec}: {e}")
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    record_ids = np.array(record_ids, dtype=int)
    beat_classes = np.array(beat_classes, dtype=object)
    return X, y, record_ids, beat_classes



def load_from_excel(path):
    features = pd.read_excel(path, sheet_name="features")
    labels = pd.read_excel(path, sheet_name="labels")
    metadata = pd.read_excel(path, sheet_name="metadata")

    X = features.to_numpy(dtype=np.float32)
    y = labels.iloc[:, 0].to_numpy(dtype=np.float32).reshape(-1, 1)
    record_ids = metadata["record_id"].to_numpy(dtype=int)
    beat_classes = metadata["beat_class"].to_numpy(dtype=object)
    return X, y, record_ids, beat_classes


print("Loading MIT-BIH data...")
if data_source == "excel" and os.path.exists(excel_path):
    X, y, record_ids, beat_classes = load_from_excel(excel_path)
    print(f"Loaded data from {excel_path}")
else:
    X, y, record_ids, beat_classes = load_mit_bih_data(max_samples=12000)  # Larger subset for better training coverage
    print(f"Loaded data from MIT-BIH source")

    # Save the loaded data for reuse as an Excel file
    feature_df = pd.DataFrame(X, dtype=float)
    label_df = pd.DataFrame(y.reshape(-1, 1), columns=["label"])
    metadata_df = pd.DataFrame({
        "record_id": record_ids,
        "beat_class": beat_classes,
        "label": y.reshape(-1),
    })
    with pd.ExcelWriter(excel_path) as writer:
        feature_df.to_excel(writer, sheet_name="features", index=False)
        label_df.to_excel(writer, sheet_name="labels", index=False)
        metadata_df.to_excel(writer, sheet_name="metadata", index=False)
    print(f"Saved MIT-BIH data to {excel_path}")

# Preprocess
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
class_counts = np.bincount(y.astype(int).ravel())
if len(class_counts) < 2 or np.any(class_counts < 2):
    print("Not enough samples in each class for stratified splitting; using a non-stratified split instead.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train_t = torch.from_numpy(X_train)
y_train_t = torch.from_numpy(y_train)
X_test_t = torch.from_numpy(X_test)
y_test_t = torch.from_numpy(y_test)

print(f"Dataset shape: {X.shape} | Normal: {int((y == 0).sum())} | Abnormal: {int((y == 1).sum())}")


# ====================== 2. Neural Network ======================
class ECG_MLP(nn.Module):
    def __init__(self, input_size=180):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


# ====================== 3. PSO Utilities ======================
def get_flattened_weights(model):
    return np.concatenate([p.data.cpu().numpy().flatten() for p in model.parameters()])

def set_flattened_weights(model, weights):
    idx = 0
    for p in model.parameters():
        size = p.data.numel()
        p.data.copy_(torch.from_numpy(weights[idx:idx + size].reshape(p.shape)))
        idx += size

# ====================== 4. PSO Training ======================
n_particles = 40
max_iter = 80
w, c1, c2 = 0.65, 1.8, 1.9

model = ECG_MLP(input_size=X.shape[1])
dim = len(get_flattened_weights(model))

positions = np.random.uniform(-0.5, 0.5, (n_particles, dim))
velocities = np.zeros((n_particles, dim))
pbest_pos = positions.copy()
pbest_scores = np.full(n_particles, np.inf)
gbest_pos = None
gbest_score = np.inf

def objective(weights):
    set_flattened_weights(model, weights)
    with torch.no_grad():
        outputs = model(X_train_t)
        loss = nn.BCELoss()(outputs, y_train_t)
    return loss.item()

# Baseline evaluation before PSO
with torch.no_grad():
    baseline_train_pred = (model(X_train_t) > 0.5).float()
    baseline_test_pred = (model(X_test_t) > 0.5).float()
    baseline_train_acc = (baseline_train_pred == y_train_t).float().mean().item()
    baseline_test_acc = (baseline_test_pred == y_test_t).float().mean().item()

print("Baseline performance before PSO:")
print(f"  Train Accuracy: {baseline_train_acc:.3f}")
print(f"  Test Accuracy:  {baseline_test_acc:.3f}")
print_metrics("Baseline Test Metrics", y_test_t.numpy().astype(int).ravel(), baseline_test_pred.numpy().astype(int).ravel())

print("\nStarting PSO optimization on MIT-BIH ECG data...")

for it in range(max_iter):
    for i in range(n_particles):
        fitness = objective(positions[i])
        
        if fitness < pbest_scores[i]:
            pbest_scores[i] = fitness
            pbest_pos[i] = positions[i].copy()
        
        if fitness < gbest_score:
            gbest_score = fitness
            gbest_pos = positions[i].copy()
    
    # Velocity & position update
    for i in range(n_particles):
        r1, r2 = np.random.rand(dim), np.random.rand(dim)
        velocities[i] = (w * velocities[i] +
                        c1 * r1 * (pbest_pos[i] - positions[i]) +
                        c2 * r2 * (gbest_pos - positions[i]))
        positions[i] += velocities[i]
    
    if it % 15 == 0:
        print(f"Iter {it:3d} | Best Loss: {gbest_score:.4f}")

# ====================== 5. Final Evaluation ======================
set_flattened_weights(model, gbest_pos)
print(f"\nFinal Training Loss: {gbest_score:.4f}")

with torch.no_grad():
    train_pred = (model(X_train_t) > 0.5).float()
    test_pred = (model(X_test_t) > 0.5).float()
    
    train_acc = (train_pred == y_train_t).float().mean().item()
    test_acc = (test_pred == y_test_t).float().mean().item()

print(f"Train Accuracy: {train_acc:.3f}")
print(f"Test Accuracy:  {test_acc:.3f}")
print_metrics("Final Test Metrics", y_test_t.numpy().astype(int).ravel(), test_pred.numpy().astype(int).ravel())

results_path = "model_results.xlsx"
train_results_df = pd.DataFrame({
    "observation": y_train_t.numpy().astype(int).ravel(),
    "prediction": train_pred.numpy().astype(int).ravel(),
    "set": "train",
})
test_results_df = pd.DataFrame({
    "observation": y_test_t.numpy().astype(int).ravel(),
    "prediction": test_pred.numpy().astype(int).ravel(),
    "set": "test",
})
with pd.ExcelWriter(results_path) as writer:
    train_results_df.to_excel(writer, sheet_name="train_results", index=False)
    test_results_df.to_excel(writer, sheet_name="test_results", index=False)
print(f"Saved training/testing results to {results_path}")

log_file.close()
builtins.print = _original_print