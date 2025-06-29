"""
Node Regression Task Configuration
"""

# Task metadata
TASK_METADATA = {
    'name': 'Node Regression',
    'description': 'Predict continuous values for nodes in a graph',
    'category': 'node_tasks',
    'icon': 'fas fa-chart-line',
    'color': 'info',
    'paper': 'Various GNN papers for node regression',
    'applications': [
        'Molecular property prediction',
        'Social network analysis',
        'Recommendation systems',
        'Financial risk assessment'
    ],
    'metrics': ['mse', 'mae', 'r2_score'],
    'datasets': ['cora', 'citeseer', 'pubmed', 'ogbn-mag', 'ogbn-products']
}

# Task configuration
NODE_REGRESSION_TASK = {
    'name': 'node_regression',
    'description': 'Predict continuous values for nodes in a graph',
    'supported_models': ['gcn', 'gat', 'graphsage', 'gin', 'chebnet', 'appnp'],
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
        }
    },
    'default_params': {
        'hidden_dim': 64,
        'num_layers': 3,
        'dropout': 0.1,
        'learning_rate': 0.001,
        'epochs': 100,
        'batch_size': 32
    },
    'parameter_categories': {
        'architecture': ['hidden_dim', 'num_layers', 'dropout'],
        'training': ['learning_rate', 'epochs', 'batch_size']
    }
}
