"""
Dynamic Graph Learning Task Configuration
"""

# Task metadata
TASK_METADATA = {
    'name': 'Dynamic Graph Learning',
    'description': 'Learn from graphs that evolve over time',
    'category': 'dynamic_graph',
    'icon': 'fas fa-clock',
    'color': 'info',
    'paper': 'Various GNN papers for dynamic graph learning',
    'applications': [
        'Social network evolution analysis',
        'Temporal link prediction',
        'Dynamic community detection',
        'Time-series graph forecasting'
    ],
    'metrics': ['mse', 'mae', 'r2_score', 'auc', 'auprc'],
    'datasets': ['enron', 'uc_irvine', 'facebook', 'wikipedia', 'reddit']
}

# Task configuration
DYNAMIC_GRAPH_LEARNING_TASK = {
    'name': 'dynamic_graph_learning',
    'description': 'Learn from graphs that evolve over time',
    'supported_models': ['dysat', 'tgat', 'evolvegcn', 'dyrep', 'jodie', 'tgn'],
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
        'num_time_steps': {
            'name': 'Number of Time Steps',
            'type': 'number',
            'default': 10,
            'min': 5,
            'max': 50,
            'step': 1,
            'description': 'Number of temporal snapshots to consider'
        },
        'attention_heads': {
            'name': 'Attention Heads',
            'type': 'number',
            'default': 8,
            'min': 1,
            'max': 16,
            'step': 1,
            'description': 'Number of attention heads for temporal attention'
        }
    },
    'default_params': {
        'hidden_dim': 64,
        'num_layers': 3,
        'dropout': 0.1,
        'learning_rate': 0.001,
        'epochs': 100,
        'batch_size': 32,
        'num_time_steps': 10,
        'attention_heads': 8
    },
    'parameter_categories': {
        'architecture': ['hidden_dim', 'num_layers', 'dropout', 'attention_heads'],
        'training': ['learning_rate', 'epochs', 'batch_size'],
        'temporal': ['num_time_steps']
    }
}
