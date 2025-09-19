import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef, confusion_matrix, roc_curve, precision_recall_curve, auc
import joblib
from model import ProteinClassifier
import matplotlib.pyplot as plt

print("Loading test data...")
test_protbert_data = np.load('/root/autodl-tmp/project_hyf/data/test_ProtBERT_with_labels.npy')
X_test_protbert = test_protbert_data[:, :-1]  # (B, 1024)
y_test_protbert = test_protbert_data[:, -1]

test_pssm_data = np.load('/root/autodl-tmp/project_hyf/data/data/test_PSSM_with_labels.npy')
X_test_pssm = test_pssm_data[:, :-1]  # (B, 20)
y_test_pssm = test_pssm_data[:, -1]

test_csv_data = pd.read_csv('/root/autodl-tmp/project_hyf/data/data/test_CSV_with_labels.csv').values
X_test_csv = test_csv_data[:, :-1]  # (B, 108)
y_test_csv = test_csv_data[:, -1]
X_test_protbert = np.expand_dims(X_test_protbert, axis=1)  # (B, 1, 1024)
X_test_pssm = np.expand_dims(X_test_pssm, axis=1)  # (B, 1, 20)
X_test_csv = np.expand_dims(X_test_csv, axis=1)  # (B, 1, 108)

y_test = y_test_protbert

print("Applying PCA to ProtBERT...")
pca = joblib.load('/root/autodl-tmp/project_hyf/mlp_feartures/Innovation/innovation_results/pca_model.pkl')
X_test_protbert = pca.transform(X_test_protbert.reshape(-1, X_test_protbert.shape[-1]))  # (B*1, 128)
X_test_protbert = X_test_protbert.reshape(-1, 1, 128)  # (B, 1, 128)

print("Normalizing test features...")
scaler_pssm = joblib.load('/root/autodl-tmp/project_hyf/mlp_feartures/Innovation/innovation_results/scaler_pssm.pkl')
scaler_csv = joblib.load('/root/autodl-tmp/project_hyf/mlp_feartures/Innovation/innovation_results/scaler_csv.pkl')

X_test_pssm = scaler_pssm.transform(X_test_pssm.reshape(-1, X_test_pssm.shape[-1])).reshape(X_test_pssm.shape)  # (B, 1, 20)
X_test_csv = scaler_csv.transform(X_test_csv.reshape(-1, X_test_csv.shape[-1])).reshape(X_test_csv.shape)  # (B, 1, 108)

X_test_protbert = torch.tensor(X_test_protbert, dtype=torch.float32)  # (B, 1, 128)
X_test_pssm = torch.tensor(X_test_pssm, dtype=torch.float32)  # (B, 1, 20)
X_test_csv = torch.tensor(X_test_csv, dtype=torch.float32)  # (B, 1, 108)
y_test = torch.tensor(y_test, dtype=torch.long)  # (B,)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ProteinClassifier(protbert_dim=128, pssm_dim=20, phy_dim=108).to(device)
model.load_state_dict(torch.load("/root/autodl-tmp/project_hyf/mlp_feartures/Innovation/innovation_results/protein_classifier1.pth"))
model.eval()

with torch.no_grad():
    protbert = X_test_protbert.to(device)
    pssm = X_test_pssm.to(device)
    phy = X_test_csv.to(device)
    outputs = model(protbert, pssm, phy)
    probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
    preds = (probs > 0.3).astype(int).flatten()

accuracy = accuracy_score(y_test.numpy(), preds)
f1 = f1_score(y_test.numpy(), preds)
precision = precision_score(y_test.numpy(), preds)
recall = recall_score(y_test.numpy(), preds)
roc_auc = roc_auc_score(y_test.numpy(), probs)
tnr = recall_score(y_test.numpy(), preds, pos_label=0)
cm = confusion_matrix(y_test.numpy(), preds)
TN, FP, FN, TP = cm.ravel()
fpr = FP / (FP + TN)

fpr_roc, tpr_roc, _ = roc_curve(y_test.numpy(), probs)
plt.figure()
plt.plot(fpr_roc, tpr_roc, label=f'ROC curve (area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
roc_image_path = '/root/autodl-tmp/project_hyf/mlp_feartures/Innovation/innovation_results/roc_curve.png'
plt.savefig(roc_image_path)
plt.show()

precision_pr, recall_pr, _ = precision_recall_curve(y_test.numpy(), probs)
pr_auc = auc(recall_pr, precision_pr)

plt.figure()
plt.plot(recall_pr, precision_pr, label=f'PR curve (area = {pr_auc:.4f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
pr_image_path = '/root/autodl-tmp/project_hyf/mlp_feartures/Innovation/innovation_results/pr_curve.png'
plt.savefig(pr_image_path)
plt.show()
print(f"PR-AUC: {pr_auc:.4f}")