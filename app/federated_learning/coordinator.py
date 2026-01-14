import numpy as np
import torch
import torch.nn as nn
from app.data.storage import record_training_round
from app.models.model_utils import MaternalRiskModel, evaluate_model

class FederatedLearningCoordinator:
    def __init__(self, hospital_nodes, test_dataloader, config):
        self.hospital_nodes = hospital_nodes
        self.test_dataloader = test_dataloader
        self.config = config
        self.global_model = MaternalRiskModel(
            config.INPUT_SIZE,
            config.HIDDEN_SIZE,
            config.OUTPUT_SIZE,
            config.DROPOUT_RATE
        ).to(config.DEVICE)
        self.global_round = 0
        self.history = {
            'train_metrics': [],
            'test_metrics': []
        }
        
    def aggregate_parameters(self, all_params, sample_sizes):
        """
        Aggregate model parameters using Federated Averaging
        """
        # Calculate total samples
        total_samples = sum(sample_sizes)
        
        # Initialize averaged parameters
        averaged_params = []
        for i in range(len(all_params[0])):
            # Weighted average based on sample size
            weighted_sum = np.zeros_like(all_params[0][i])
            for j in range(len(all_params)):
                weighted_sum += all_params[j][i] * sample_sizes[j]
            averaged_params.append(weighted_sum / total_samples)
            
        return averaged_params
    
    def update_global_model(self, averaged_params):
        """Update the global model with averaged parameters"""
        with torch.no_grad():
            for param, avg_param in zip(self.global_model.parameters(), averaged_params):
                param.data = torch.tensor(avg_param).to(self.config.DEVICE)
    
    def run_federated_round(self):
        """Run one round of federated learning"""
        print(f"Starting federated round {self.global_round + 1}")
        
        # Initialize all hospital models with the global model
        for hospital in self.hospital_nodes:
            hospital.initialize_model(self.global_model)
        
        # Train on each hospital's data
        all_params = []
        sample_sizes = []
        round_metrics = []
        
        for i, hospital in enumerate(self.hospital_nodes):
            print(f"  Training on hospital {i+1}...")
            params, metrics = hospital.local_train()
            all_params.append(params)
            sample_sizes.append(metrics['samples'])
            round_metrics.append(metrics)
        
        # Aggregate parameters
        averaged_params = self.aggregate_parameters(all_params, sample_sizes)
        
        # Update global model
        self.update_global_model(averaged_params)
        
        # Evaluate global model on test set
        test_metrics = self.evaluate_global_model()
        
        # Record metrics
        avg_round_metrics = self.average_metrics(round_metrics, sample_sizes)
        self.history['train_metrics'].append(avg_round_metrics)
        self.history['test_metrics'].append(test_metrics)
        
        record_training_round(self.global_round + 1, avg_round_metrics, test_metrics)

        print(f"Round {self.global_round + 1} completed:")
        print(f"  Train Loss: {avg_round_metrics['loss']:.4f}, Accuracy: {avg_round_metrics['accuracy']:.4f}")
        print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}, AUC: {test_metrics['auc']:.4f}")
        
        self.global_round += 1
        
        return avg_round_metrics, test_metrics
    
    def evaluate_global_model(self):
        """Evaluate the global model on the test set"""
        accuracy, precision, recall, f1, auc = evaluate_model(
            self.global_model,
            self.test_dataloader,
            self.config.DEVICE
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    def average_metrics(self, metrics_list, sample_sizes):
        """Calculate weighted average of metrics based on sample sizes"""
        total_samples = sum(sample_sizes)
        averaged_metrics = {}
        
        for key in metrics_list[0].keys():
            if key == 'samples':
                continue
            weighted_sum = 0
            for i, metrics in enumerate(metrics_list):
                weighted_sum += metrics[key] * sample_sizes[i]
            averaged_metrics[key] = weighted_sum / total_samples
            
        return averaged_metrics
    
    def run_federated_training(self, rounds):
        """Run multiple rounds of federated learning"""
        for _ in range(rounds):
            self.run_federated_round()
        
        return self.history
