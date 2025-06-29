"""
Dynamic Graph Learning Parameters
This module defines parameters specific to dynamic graph learning tasks.
"""

DYNAMIC_GRAPH_LEARNING_PARAMETERS = {
    # Architecture Parameters
    'hidden_dim': {
        'name': 'Hidden Dimension',
        'description': 'Dimension of hidden layers in the GNN',
        'type': 'int',
        'default': 64,
        'min': 16,
        'max': 512,
        'step': 16,
        'category': 'architecture',
        'required': True
    },
    'num_layers': {
        'name': 'Number of Layers',
        'description': 'Number of GNN layers',
        'type': 'int',
        'default': 3,
        'min': 1,
        'max': 10,
        'step': 1,
        'category': 'architecture',
        'required': True
    },
    'dropout': {
        'name': 'Dropout Rate',
        'description': 'Dropout rate for regularization',
        'type': 'float',
        'default': 0.1,
        'min': 0.0,
        'max': 0.5,
        'step': 0.1,
        'category': 'regularization',
        'required': True
    },
    'activation': {
        'name': 'Activation Function',
        'description': 'Activation function for hidden layers',
        'type': 'select',
        'default': 'relu',
        'options': ['relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu'],
        'category': 'architecture',
        'required': True
    },
    'batch_norm': {
        'name': 'Batch Normalization',
        'description': 'Whether to use batch normalization',
        'type': 'bool',
        'default': True,
        'category': 'regularization',
        'required': False
    },
    'residual': {
        'name': 'Residual Connections',
        'description': 'Whether to use residual connections',
        'type': 'bool',
        'default': False,
        'category': 'architecture',
        'required': False
    },
    
    # Dynamic Graph Specific Parameters
    'temporal_model': {
        'name': 'Temporal Model',
        'description': 'Type of temporal model to use',
        'type': 'select',
        'default': 'rnn',
        'options': ['rnn', 'lstm', 'gru', 'transformer', 'tcn'],
        'category': 'dynamic_graph',
        'required': True
    },
    'time_steps': {
        'name': 'Time Steps',
        'description': 'Number of time steps to consider',
        'type': 'int',
        'default': 10,
        'min': 1,
        'max': 100,
        'step': 1,
        'category': 'dynamic_graph',
        'required': True
    },
    'prediction_horizon': {
        'name': 'Prediction Horizon',
        'description': 'Number of time steps to predict into the future',
        'type': 'int',
        'default': 5,
        'min': 1,
        'max': 50,
        'step': 1,
        'category': 'dynamic_graph',
        'required': True
    },
    'temporal_aggregation': {
        'name': 'Temporal Aggregation',
        'description': 'Method to aggregate temporal information',
        'type': 'select',
        'default': 'attention',
        'options': ['attention', 'mean', 'max', 'last', 'concat'],
        'category': 'dynamic_graph',
        'required': True
    },
    'use_temporal_features': {
        'name': 'Use Temporal Features',
        'description': 'Whether to use temporal features',
        'type': 'bool',
        'default': True,
        'category': 'dynamic_graph',
        'required': False
    },
    'temporal_feature_dim': {
        'name': 'Temporal Feature Dimension',
        'description': 'Dimension of temporal features',
        'type': 'int',
        'default': 32,
        'min': 8,
        'max': 256,
        'step': 8,
        'category': 'dynamic_graph',
        'required': False
    },
    'memory_size': {
        'name': 'Memory Size',
        'description': 'Size of memory for temporal models',
        'type': 'int',
        'default': 100,
        'min': 10,
        'max': 1000,
        'step': 10,
        'category': 'dynamic_graph',
        'required': False
    },
    
    # Training Parameters
    'learning_rate': {
        'name': 'Learning Rate',
        'description': 'Learning rate for optimization',
        'type': 'float',
        'default': 0.001,
        'min': 0.00001,
        'max': 0.1,
        'step': 0.0001,
        'category': 'training',
        'required': True
    },
    'epochs': {
        'name': 'Number of Epochs',
        'description': 'Number of training epochs',
        'type': 'int',
        'default': 100,
        'min': 10,
        'max': 1000,
        'step': 10,
        'category': 'training',
        'required': True
    },
    'batch_size': {
        'name': 'Batch Size',
        'description': 'Batch size for training',
        'type': 'int',
        'default': 32,
        'min': 1,
        'max': 256,
        'step': 1,
        'category': 'training',
        'required': True
    },
    'optimizer': {
        'name': 'Optimizer',
        'description': 'Optimization algorithm',
        'type': 'select',
        'default': 'adam',
        'options': ['adam', 'sgd', 'adamw', 'rmsprop', 'adagrad'],
        'category': 'training',
        'required': True
    },
    'loss_function': {
        'name': 'Loss Function',
        'description': 'Loss function for dynamic graph learning',
        'type': 'select',
        'default': 'mse',
        'options': ['mse', 'mae', 'huber', 'smooth_l1', 'temporal_loss'],
        'category': 'training',
        'required': True
    },
    'weight_decay': {
        'name': 'Weight Decay',
        'description': 'L2 regularization coefficient',
        'type': 'float',
        'default': 0.0001,
        'min': 0.0,
        'max': 0.1,
        'step': 0.001,
        'category': 'regularization',
        'required': False
    },
    'scheduler': {
        'name': 'Learning Rate Scheduler',
        'description': 'Learning rate scheduling strategy',
        'type': 'select',
        'default': 'none',
        'options': ['none', 'step', 'cosine', 'plateau', 'exponential'],
        'category': 'training',
        'required': False
    },
    'early_stopping_patience': {
        'name': 'Early Stopping Patience',
        'description': 'Number of epochs to wait before early stopping',
        'type': 'int',
        'default': 10,
        'min': 1,
        'max': 50,
        'step': 1,
        'category': 'training',
        'required': False
    },
    'gradient_clip': {
        'name': 'Gradient Clipping',
        'description': 'Gradient clipping value',
        'type': 'float',
        'default': 1.0,
        'min': 0.1,
        'max': 10.0,
        'step': 0.1,
        'category': 'training',
        'required': False
    },
    
    # Dataset Parameters
    'dataset': {
        'name': 'Dataset',
        'description': 'Select a dynamic graph dataset',
        'type': 'select',
        'default': 'enron',
        'options': [
            'enron', 'uc_irvine', 'facebook', 'twitter', 'reddit',
            'wikipedia', 'stackoverflow', 'github', 'bitcoin', 'ethereum'
        ],
        'category': 'dataset',
        'required': True
    },
    'temporal_split': {
        'name': 'Temporal Split',
        'description': 'Fraction of time for validation',
        'type': 'float',
        'default': 0.2,
        'min': 0.1,
        'max': 0.5,
        'step': 0.1,
        'category': 'dataset',
        'required': True
    },
    'test_split': {
        'name': 'Test Split',
        'description': 'Fraction of time for testing',
        'type': 'float',
        'default': 0.2,
        'min': 0.1,
        'max': 0.5,
        'step': 0.1,
        'category': 'dataset',
        'required': True
    },
    'use_node_features': {
        'name': 'Use Node Features',
        'description': 'Whether to use node features',
        'type': 'bool',
        'default': True,
        'category': 'dataset',
        'required': False
    },
    'use_edge_features': {
        'name': 'Use Edge Features',
        'description': 'Whether to use edge features',
        'type': 'bool',
        'default': False,
        'category': 'dataset',
        'required': False
    },
    'normalize_features': {
        'name': 'Normalize Features',
        'description': 'Whether to normalize node and edge features',
        'type': 'bool',
        'default': True,
        'category': 'dataset',
        'required': False
    },
    'time_window': {
        'name': 'Time Window',
        'description': 'Time window for temporal aggregation',
        'type': 'int',
        'default': 30,
        'min': 1,
        'max': 365,
        'step': 1,
        'category': 'dataset',
        'required': False
    },
    
    # Model-Specific Parameters
    'attention_heads': {
        'name': 'Attention Heads',
        'description': 'Number of attention heads (for GAT/Transformer)',
        'type': 'int',
        'default': 8,
        'min': 1,
        'max': 16,
        'step': 1,
        'category': 'model_specific',
        'required': False,
        'models': ['gat', 'transformer']
    },
    'attention_dropout': {
        'name': 'Attention Dropout',
        'description': 'Dropout rate for attention weights',
        'type': 'float',
        'default': 0.1,
        'min': 0.0,
        'max': 0.5,
        'step': 0.1,
        'category': 'model_specific',
        'required': False,
        'models': ['gat', 'transformer']
    },
    'neighbor_sampling': {
        'name': 'Neighbor Sampling',
        'description': 'Number of neighbors to sample (for GraphSAGE)',
        'type': 'int',
        'default': 25,
        'min': 5,
        'max': 100,
        'step': 5,
        'category': 'model_specific',
        'required': False,
        'models': ['graphsage']
    },
    'aggregator_type': {
        'name': 'Aggregator Type',
        'description': 'Type of neighbor aggregation (for GraphSAGE)',
        'type': 'select',
        'default': 'mean',
        'options': ['mean', 'max', 'sum', 'lstm'],
        'category': 'model_specific',
        'required': False,
        'models': ['graphsage']
    },
    'epsilon': {
        'name': 'Epsilon',
        'description': 'Epsilon parameter for GIN',
        'type': 'float',
        'default': 0.0,
        'min': 0.0,
        'max': 1.0,
        'step': 0.1,
        'category': 'model_specific',
        'required': False,
        'models': ['gin']
    },
    'k_hop': {
        'name': 'K-Hop',
        'description': 'Number of hops for Chebyshev polynomials',
        'type': 'int',
        'default': 3,
        'min': 1,
        'max': 10,
        'step': 1,
        'category': 'model_specific',
        'required': False,
        'models': ['chebnet']
    },
    'hidden_size': {
        'name': 'Hidden Size',
        'description': 'Hidden size for RNN/LSTM/GRU',
        'type': 'int',
        'default': 64,
        'min': 16,
        'max': 512,
        'step': 16,
        'category': 'model_specific',
        'required': False,
        'models': ['rnn', 'lstm', 'gru']
    },
    'num_layers_rnn': {
        'name': 'RNN Layers',
        'description': 'Number of layers for RNN/LSTM/GRU',
        'type': 'int',
        'default': 2,
        'min': 1,
        'max': 5,
        'step': 1,
        'category': 'model_specific',
        'required': False,
        'models': ['rnn', 'lstm', 'gru']
    },
    'bidirectional': {
        'name': 'Bidirectional',
        'description': 'Whether to use bidirectional RNN/LSTM/GRU',
        'type': 'bool',
        'default': False,
        'category': 'model_specific',
        'required': False,
        'models': ['rnn', 'lstm', 'gru']
    },
    'kernel_size': {
        'name': 'Kernel Size',
        'description': 'Kernel size for TCN',
        'type': 'int',
        'default': 3,
        'min': 1,
        'max': 10,
        'step': 1,
        'category': 'model_specific',
        'required': False,
        'models': ['tcn']
    },
    'dilation': {
        'name': 'Dilation',
        'description': 'Dilation factor for TCN',
        'type': 'int',
        'default': 2,
        'min': 1,
        'max': 10,
        'step': 1,
        'category': 'model_specific',
        'required': False,
        'models': ['tcn']
    }
}

# Default parameters for quick start
DEFAULT_DYNAMIC_GRAPH_LEARNING_PARAMS = {
    'hidden_dim': 64,
    'num_layers': 3,
    'dropout': 0.1,
    'activation': 'relu',
    'batch_norm': True,
    'residual': False,
    'temporal_model': 'rnn',
    'time_steps': 10,
    'prediction_horizon': 5,
    'temporal_aggregation': 'attention',
    'use_temporal_features': True,
    'temporal_feature_dim': 32,
    'memory_size': 100,
    'learning_rate': 0.001,
    'epochs': 100,
    'batch_size': 32,
    'optimizer': 'adam',
    'loss_function': 'mse',
    'weight_decay': 0.0001,
    'scheduler': 'none',
    'early_stopping_patience': 10,
    'gradient_clip': 1.0,
    'dataset': 'enron',
    'temporal_split': 0.2,
    'test_split': 0.2,
    'use_node_features': True,
    'use_edge_features': False,
    'normalize_features': True,
    'time_window': 30
}

# Parameter categories for UI organization
PARAMETER_CATEGORIES = {
    'architecture': {
        'name': 'Architecture Parameters',
        'description': 'Model architecture configuration',
        'icon': 'fas fa-cogs',
        'color': 'primary'
    },
    'dynamic_graph': {
        'name': 'Dynamic Graph Parameters',
        'description': 'Parameters specific to dynamic graph learning',
        'icon': 'fas fa-clock',
        'color': 'secondary'
    },
    'training': {
        'name': 'Training Parameters',
        'description': 'Training configuration and optimization',
        'icon': 'fas fa-graduation-cap',
        'color': 'success'
    },
    'regularization': {
        'name': 'Regularization Parameters',
        'description': 'Regularization techniques to prevent overfitting',
        'icon': 'fas fa-shield-alt',
        'color': 'warning'
    },
    'dataset': {
        'name': 'Dataset Parameters',
        'description': 'Dataset configuration and preprocessing',
        'icon': 'fas fa-database',
        'color': 'info'
    },
    'model_specific': {
        'name': 'Model-Specific Parameters',
        'description': 'Parameters specific to certain model architectures',
        'icon': 'fas fa-tools',
        'color': 'dark'
    }
}
