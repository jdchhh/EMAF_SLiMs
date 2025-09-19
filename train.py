import torch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from torch.utils.data import DataLoader, TensorDataset
from model import ProteinClassifier
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F

print("Loading training data...")
protbert_data = np.load('/root/autodl-tmp/project_hyf/data/train_ProtBERT_with_labels.npy')
X_train_protbert = protbert_data[:, :-1]
y_train_protbert = protbert_data[:, -1]

pssm_data = np.load('/root/autodl-tmp/project_hyf/data/train_PSSM_with_labels.npy')
X_train_pssm = pssm_data[:, :-1]
y_train_pssm = pssm_data[:, -1]

csv_data = pd.read_csv('/root/autodl-tmp/project_hyf/train_csv_with_lable.csv').values
X_train_csv = csv_data[:, :-1]
y_train_csv = csv_data[:, -1]

X_train_protbert = np.expand_dims(X_train_protbert, axis=1)  # (B, 1, 1024)
X_train_pssm = np.expand_dims(X_train_pssm, axis=1)  # (B, 1, 20)
X_train_csv = np.expand_dims(X_train_csv, axis=1)  # (B, 1, 108)

print("Applying PCA to ProtBERT features...")
protbert_pca = PCA(n_components=128)
X_train_protbert = protbert_pca.fit_transform(X_train_protbert.reshape(-1, X_train_protbert.shape[-1]))
X_train_protbert = X_train_protbert.reshape(-1, 1, 128)
joblib.dump(protbert_pca, '/root/autodl-tmp/project_hyf/mlp_feartures/Innovation/innovation_results/pca_model.pkl')

print("Normalizing PSSM & Phy features...")
scaler_pssm = StandardScaler()
scaler_csv = StandardScaler()
X_train_pssm = scaler_pssm.fit_transform(X_train_pssm.reshape(-1, X_train_pssm.shape[-1])).reshape(-1, 1, 20)
X_train_csv = scaler_csv.fit_transform(X_train_csv.reshape(-1, X_train_csv.shape[-1])).reshape(-1, 1, 108)

joblib.dump(scaler_pssm, '/root/autodl-tmp/project_hyf/mlp_feartures/Innovation/innovation_results/scaler_pssm.pkl')
joblib.dump(scaler_csv, '/root/autodl-tmp/project_hyf/mlp_feartures/Innovation/innovation_results/scaler_csv.pkl')

X_train = np.concatenate([X_train_protbert, X_train_pssm, X_train_csv], axis=-1)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

batch_size = 64
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ProteinClassifier(protbert_dim=128, pssm_dim=20, phy_dim=108).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        features = batch[0].to(device)
        # 拆分特征
        protbert = features[:, :, :128]
        pssm = features[:, :, 128:128+20]
        phy = features[:, :, 128+20:]
        labels = batch[1].to(device)

        optimizer.zero_grad()
        outputs = model(protbert, pssm, phy)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

torch.save(model.state_dict(), "/root/autodl-tmp/project_hyf/mlp_feartures/Innovation/innovation_results/protein_classifier1.pth")
print("Model saved!")