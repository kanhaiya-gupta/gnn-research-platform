"""
Graph Regression Parameters
This module defines parameters specific to graph regression tasks.
"""

GRAPH_REGRESSION_PARAMETERS = {
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
    'output_dim': {
        'name': 'Output Dimension',
        'description': 'Number of output values per graph',
        'type': 'int',
        'default': 1,
        'min': 1,
        'max': 10,
        'step': 1,
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
    
    # Graph-Level Parameters
    'readout_method': {
        'name': 'Readout Method',
        'description': 'Method to aggregate node features to graph-level',
        'type': 'select',
        'default': 'mean',
        'options': ['mean', 'sum', 'max', 'attention', 'sort'],
        'category': 'graph_level',
        'required': True
    },
    'pooling_method': {
        'name': 'Pooling Method',
        'description': 'Graph pooling method for hierarchical models',
        'type': 'select',
        'default': 'diffpool',
        'options': ['diffpool', 'sortpool', 'sagpool', 'edgepool', 'none'],
        'category': 'graph_level',
        'required': False
    },
    'graph_feature_dim': {
        'name': 'Graph Feature Dimension',
        'description': 'Dimension of graph-level features',
        'type': 'int',
        'default': 128,
        'min': 32,
        'max': 512,
        'step': 32,
        'category': 'graph_level',
        'required': False
    },
    'mlp_layers': {
        'name': 'MLP Layers',
        'description': 'Number of MLP layers for graph regression',
        'type': 'int',
        'default': 2,
        'min': 1,
        'max': 5,
        'step': 1,
        'category': 'graph_level',
        'required': False
    },
    'use_graph_features': {
        'name': 'Use Graph Features',
        'description': 'Whether to use additional graph-level features',
        'type': 'bool',
        'default': True,
        'category': 'graph_level',
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
        'description': 'Loss function for regression',
        'type': 'select',
        'default': 'mse',
        'options': ['mse', 'mae', 'huber', 'smooth_l1', 'log_cosh'],
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
    
    # Dataset Parameters
    'dataset': {
        'name': 'Dataset',
        'description': 'Select a graph dataset for graph regression',
        'type': 'select',
        'default': 'zinc',
        'options': [
            'zinc', 'qm9', 'qm7', 'esol', 'freesolv', 'lipophilicity',
            'bace', 'bbbp', 'hiv', 'tox21', 'sider', 'clintox'
        ],
        'category': 'dataset',
        'required': True
    },
    'target_property': {
        'name': 'Target Property',
        'description': 'Target property to predict',
        'type': 'select',
        'default': 'logp',
        'options': ['logp', 'qed', 'sas', 'mol_weight', 'tpsa', 'lipophilicity'],
        'category': 'dataset',
        'required': True
    },
    'validation_split': {
        'name': 'Validation Split',
        'description': 'Fraction of graphs for validation',
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
        'description': 'Fraction of graphs for testing',
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
        'default': True,
        'category': 'dataset',
        'required': False
    },
    'max_nodes': {
        'name': 'Maximum Nodes',
        'description': 'Maximum number of nodes per graph',
        'type': 'int',
        'default': 1000,
        'min': 10,
        'max': 10000,
        'step': 10,
        'category': 'dataset',
        'required': False
    },
    'max_edges': {
        'name': 'Maximum Edges',
        'description': 'Maximum number of edges per graph',
        'type': 'int',
        'default': 5000,
        'min': 10,
        'max': 50000,
        'step': 10,
        'category': 'dataset',
        'required': False
    },
    'normalize_target': {
        'name': 'Normalize Target',
        'description': 'Whether to normalize the target values',
        'type': 'bool',
        'default': True,
        'category': 'dataset',
        'required': False
    },
    
    # Model-Specific Parameters
    'attention_heads': {
        'name': 'Attention Heads',
        'description': 'Number of attention heads (for GAT)',
        'type': 'int',
        'default': 8,
        'min': 1,
        'max': 16,
        'step': 1,
        'category': 'model_specific',
        'required': False,
        'models': ['gat']
    },
    'attention_dropout': {
        'name': 'Attention Dropout',
        'description': 'Dropout rate for attention weights (for GAT)',
        'type': 'float',
        'default': 0.1,
        'min': 0.0,
        'max': 0.5,
        'step': 0.1,
        'category': 'model_specific',
        'required': False,
        'models': ['gat']
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
    'pooling_ratio': {
        'name': 'Pooling Ratio',
        'description': 'Ratio of nodes to keep after pooling',
        'type': 'float',
        'default': 0.5,
        'min': 0.1,
        'max': 0.9,
        'step': 0.1,
        'category': 'model_specific',
        'required': False,
        'models': ['diffpool', 'sagpool']
    },
    'sort_k': {
        'name': 'Sort K',
        'description': 'Number of nodes to keep after sorting (for SortPool)',
        'type': 'int',
        'default': 30,
        'min': 5,
        'max': 100,
        'step': 5,
        'category': 'model_specific',
        'required': False,
        'models': ['sortpool']
    }
}

# Default parameters for quick start
DEFAULT_GRAPH_REGRESSION_PARAMS = {
    'hidden_dim': 64,
    'num_layers': 3,
    'dropout': 0.1,
    'activation': 'relu',
    'output_dim': 1,
    'batch_norm': True,
    'residual': False,
    'readout_method': 'mean',
    'pooling_method': 'diffpool',
    'graph_feature_dim': 128,
    'mlp_layers': 2,
    'use_graph_features': True,
    'learning_rate': 0.001,
    'epochs': 100,
    'batch_size': 32,
    'optimizer': 'adam',
    'loss_function': 'mse',
    'weight_decay': 0.0001,
    'scheduler': 'none',
    'early_stopping_patience': 10,
    'dataset': 'zinc',
    'target_property': 'logp',
    'validation_split': 0.2,
    'test_split': 0.2,
    'use_node_features': True,
    'use_edge_features': True,
    'max_nodes': 1000,
    'max_edges': 5000,
    'normalize_target': True
}

# Parameter categories for UI organization
PARAMETER_CATEGORIES = {
    'architecture': {
        'name': 'Architecture Parameters',
        'description': 'Model architecture configuration',
        'icon': 'fas fa-cogs',
        'color': 'primary'
    },
    'graph_level': {
        'name': 'Graph-Level Parameters',
        'description': 'Parameters for graph-level operations',
        'icon': 'fas fa-project-diagram',
        'color': 'info'
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
        'color': 'secondary'
    },
    'model_specific': {
        'name': 'Model-Specific Parameters',
        'description': 'Parameters specific to certain model architectures',
        'icon': 'fas fa-tools',
        'color': 'dark'
    }
}
