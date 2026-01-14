import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

class MaternalRiskModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.3):
        super(MaternalRiskModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.layer3 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.layer3(out)
        out = self.sigmoid(out)
        return out

def train_model(model, dataloader, criterion, optimizer, device, privacy_engine=None):
    """
    Train the model for one epoch
    """
    model.train()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    
    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * features.size(0)
        predicted = (outputs.data > 0.5).float()
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    
    # Calculate metrics
    all_predictions = np.array(all_predictions).flatten()
    all_labels = np.array(all_labels).flatten()
    
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    
    return epoch_loss, accuracy, precision, recall, f1

def evaluate_model(model, dataloader, device):
    """
    Evaluate the model
    """
    model.eval()
    all_predictions = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            probs = outputs.cpu().numpy()
            predicted = (outputs.data > 0.5).float().cpu().numpy()
            
            all_probs.extend(probs)
            all_predictions.extend(predicted)
            all_labels.extend(labels.cpu().numpy())
    
    all_labels = np.array(all_labels).flatten()
    all_predictions = np.array(all_predictions).flatten()
    all_probs = np.array(all_probs).flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    
    # Calculate AUC if there are positive examples
    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs)
    else:
        auc = 0.0
    
    return accuracy, precision, recall, f1, auc

def setup_differential_privacy(model, optimizer, dataloader, noise_multiplier, max_grad_norm, delta):
    """
    Set up differential privacy using Opacus
    """
    privacy_engine = PrivacyEngine()
    
    model, optimizer, dataloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    
    return model, optimizer, dataloader, privacy_engine
