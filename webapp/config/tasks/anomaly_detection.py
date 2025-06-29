"""
Anomaly Detection Task Configuration
"""

# Task metadata
TASK_METADATA = {
    'name': 'Anomaly Detection',
    'description': 'Detect anomalous nodes or edges in graphs',
    'category': 'anomaly_detection',
    'icon': 'fas fa-exclamation-triangle',
    'color': 'danger',
    'paper': 'Various GNN papers for anomaly detection',
    'applications': [
        'Fraud detection in financial networks',
        'Intrusion detection in computer networks',
        'Disease outbreak detection',
        'Quality control in manufacturing'
    ],
    'metrics': ['precision', 'recall', 'f1_score', 'auc', 'auprc'],
    'datasets': ['cora', 'citeseer', 'pubmed', 'enron', 'facebook']
}

# Task configuration
ANOMALY_DETECTION_TASK = {
    'name': 'anomaly_detection',
    'description': 'Detect anomalous nodes or edges in graphs',
    'supported_models': ['gcn_ae', 'gat_ae', 'dominant', 'anomalydae', 'ganomaly', 'graphsage_ae'],
    'parameters': {
        'hidden_dim': {
            'name': 'Hidden Dimension',
            'type': 'number',
            'default': 64,
            'min': 16,
            'max': 512,
            'step': 16,
            'description': 'Dimension of hidden layers'
        },
        'num_layers': {
            'name': 'Number of Layers',
            'type': 'number',
            'default': 3,
            'min': 1,
            'max': 10,
            'step': 1,
            'description': 'Number of GNN layers'
        },
        'dropout': {
            'name': 'Dropout Rate',
            'type': 'number',
            'default': 0.1,
            'min': 0.0,
            'max': 0.5,
            'step': 0.1,
            'description': 'Dropout rate for regularization'
        },
        'learning_rate': {
            'name': 'Learning Rate',
            'type': 'number',
            'default': 0.001,
            'min': 0.0001,
            'max': 0.1,
            'step': 0.0001,
            'description': 'Learning rate for optimization'
        },
        'epochs': {
            'name': 'Epochs',
            'type': 'number',
            'default': 100,
            'min': 10,
            'max': 1000,
            'step': 10,
            'description': 'Number of training epochs'
        },
        'batch_size': {
            'name': 'Batch Size',
            'type': 'number',
            'default': 32,
            'min': 1,
            'max': 256,
            'step': 1,
            'description': 'Batch size for training'
        },
        'anomaly_threshold': {
            'name': 'Anomaly Threshold',
            'type': 'number',
            'default': 0.5,
            'min': 0.1,
            'max': 0.9,
            'step': 0.1,
            'description': 'Threshold for anomaly detection'
        },
        'latent_dim': {
            'name': 'Latent Dimension',
            'type': 'number',
            'default': 32,
            'min': 8,
            'max': 256,
            'step': 8,
            'description': 'Dimension of latent space for autoencoders'
        }
    },
    'default_params': {
        'hidden_dim': 64,
        'num_layers': 3,
        'dropout': 0.1,
        'learning_rate': 0.001,
        'epochs': 100,
        'batch_size': 32,
        'anomaly_threshold': 0.5,
        'latent_dim': 32
    },
    'parameter_categories': {
        'architecture': ['hidden_dim', 'num_layers', 'dropout', 'latent_dim'],
        'training': ['learning_rate', 'epochs', 'batch_size'],
        'detection': ['anomaly_threshold']
    }
}
