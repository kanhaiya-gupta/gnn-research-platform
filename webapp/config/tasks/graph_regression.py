"""
Graph Regression Task Configuration
"""

# Task metadata
TASK_METADATA = {
    'name': 'Graph Regression',
    'description': 'Predict continuous values for entire graphs',
    'category': 'graph_tasks',
    'icon': 'fas fa-chart-line',
    'color': 'success',
    'paper': 'Various GNN papers for graph regression',
    'applications': [
        'Molecular property prediction',
        'Drug discovery',
        'Material science',
        'Chemical property prediction'
    ],
    'metrics': ['mse', 'mae', 'r2_score'],
    'datasets': ['ogbg-molhiv', 'ogbg-molpcba', 'ogbg-moltox21', 'ogbg-molbace']
}

# Task configuration
GRAPH_REGRESSION_TASK = {
    'name': 'graph_regression',
    'description': 'Predict continuous values for entire graphs',
    'supported_models': ['gcn', 'gat', 'graphsage', 'gin', 'diffpool', 'sortpool'],
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
        'pooling_method': {
            'name': 'Pooling Method',
            'type': 'select',
            'default': 'mean',
            'options': [
                {'value': 'mean', 'label': 'Mean Pooling'},
                {'value': 'sum', 'label': 'Sum Pooling'},
                {'value': 'max', 'label': 'Max Pooling'},
                {'value': 'attention', 'label': 'Attention Pooling'}
            ],
            'description': 'Graph-level pooling method'
        }
    },
    'default_params': {
        'hidden_dim': 64,
        'num_layers': 3,
        'dropout': 0.1,
        'learning_rate': 0.001,
        'epochs': 100,
        'batch_size': 32,
        'pooling_method': 'mean'
    },
    'parameter_categories': {
        'architecture': ['hidden_dim', 'num_layers', 'dropout', 'pooling_method'],
        'training': ['learning_rate', 'epochs', 'batch_size']
    }
}
