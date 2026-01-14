import torch
import torch.nn as nn
import copy
from app.models.model_utils import train_model, evaluate_model

class HospitalNode:
    def __init__(self, node_id, dataloader, device, config):
        self.node_id = node_id
        self.dataloader = dataloader
        self.device = device
        self.config = config
        self.model = None
        self.optimizer = None
        self.criterion = nn.BCELoss()
        
    def initialize_model(self, model):
        """Initialize with the global model"""
        self.model = copy.deepcopy(model).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE
        )
        
    def local_train(self, privacy_engine=None):
        """Train on local data for one epoch"""
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_model first.")
            
        epoch_loss, accuracy, precision, recall, f1 = train_model(
            self.model,
            self.dataloader,
            self.criterion,
            self.optimizer,
            self.device,
            privacy_engine
        )
        
        # Get model parameters to send back to coordinator
        model_params = [param.data.cpu().numpy() for param in self.model.parameters()]
        
        metrics = {
            'loss': epoch_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'samples': len(self.dataloader.dataset)
        }
        
        return model_params, metrics
        
    def evaluate(self):
        """Evaluate the model on local data"""
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_model first.")
            
        accuracy, precision, recall, f1, auc = evaluate_model(
            self.model,
            self.dataloader,
            self.device
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'samples': len(self.dataloader.dataset)
        }
