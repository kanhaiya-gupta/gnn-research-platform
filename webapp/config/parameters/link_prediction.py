"""
Link Prediction Parameters
This module defines parameters specific to link prediction tasks.
"""

LINK_PREDICTION_PARAMETERS = {
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
    'embedding_dim': {
        'name': 'Embedding Dimension',
        'description': 'Dimension of node embeddings',
        'type': 'int',
        'default': 64,
        'min': 16,
        'max': 512,
        'step': 16,
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
    
    # Link Prediction Specific Parameters
    'link_prediction_method': {
        'name': 'Link Prediction Method',
        'description': 'Method for link prediction',
        'type': 'select',
        'default': 'dot_product',
        'options': ['dot_product', 'cosine', 'euclidean', 'mlp', 'bilinear'],
        'category': 'link_prediction',
        'required': True
    },
    'negative_sampling_ratio': {
        'name': 'Negative Sampling Ratio',
        'description': 'Ratio of negative samples to positive samples',
        'type': 'float',
        'default': 1.0,
        'min': 0.1,
        'max': 10.0,
        'step': 0.1,
        'category': 'link_prediction',
        'required': True
    },
    'use_edge_features': {
        'name': 'Use Edge Features',
        'description': 'Whether to use edge features in prediction',
        'type': 'bool',
        'default': False,
        'category': 'link_prediction',
        'required': False
    },
    'edge_feature_dim': {
        'name': 'Edge Feature Dimension',
        'description': 'Dimension of edge features',
        'type': 'int',
        'default': 32,
        'min': 8,
        'max': 256,
        'step': 8,
        'category': 'link_prediction',
        'required': False
    },
    'prediction_threshold': {
        'name': 'Prediction Threshold',
        'description': 'Threshold for link prediction',
        'type': 'float',
        'default': 0.5,
        'min': 0.0,
        'max': 1.0,
        'step': 0.1,
        'category': 'link_prediction',
        'required': False
    },
    'use_structural_features': {
        'name': 'Use Structural Features',
        'description': 'Whether to use structural features (degree, clustering, etc.)',
        'type': 'bool',
        'default': True,
        'category': 'link_prediction',
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
        'description': 'Loss function for link prediction',
        'type': 'select',
        'default': 'bce',
        'options': ['bce', 'focal_loss', 'margin_loss', 'contrastive_loss'],
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
        'description': 'Select a graph dataset for link prediction',
        'type': 'select',
        'default': 'cora',
        'options': [
            'cora', 'citeseer', 'pubmed', 'ogbn_arxiv', 'ogbn_products',
            'ogbn_mag', 'reddit', 'flickr', 'amazon_photo', 'amazon_computers'
        ],
        'category': 'dataset',
        'required': True
    },
    'validation_split': {
        'name': 'Validation Split',
        'description': 'Fraction of edges for validation',
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
        'description': 'Fraction of edges for testing',
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
    'temporal_split': {
        'name': 'Temporal Split',
        'description': 'Whether to use temporal split for link prediction',
        'type': 'bool',
        'default': False,
        'category': 'dataset',
        'required': False
    },
    'time_window': {
        'name': 'Time Window',
        'description': 'Time window for temporal link prediction',
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
    'latent_dim': {
        'name': 'Latent Dimension',
        'description': 'Latent dimension for VGAE',
        'type': 'int',
        'default': 32,
        'min': 8,
        'max': 256,
        'step': 8,
        'category': 'model_specific',
        'required': False,
        'models': ['vgae']
    },
    'beta': {
        'name': 'Beta',
        'description': 'Beta parameter for VGAE (KL divergence weight)',
        'type': 'float',
        'default': 1.0,
        'min': 0.0,
        'max': 10.0,
        'step': 0.1,
        'category': 'model_specific',
        'required': False,
        'models': ['vgae']
    },
    'subgraph_size': {
        'name': 'Subgraph Size',
        'description': 'Size of subgraphs for SEAL',
        'type': 'int',
        'default': 20,
        'min': 5,
        'max': 100,
        'step': 5,
        'category': 'model_specific',
        'required': False,
        'models': ['seal']
    }
}

# Default parameters for quick start
DEFAULT_LINK_PREDICTION_PARAMS = {
    'hidden_dim': 64,
    'num_layers': 3,
    'dropout': 0.1,
    'activation': 'relu',
    'embedding_dim': 64,
    'batch_norm': True,
    'residual': False,
    'link_prediction_method': 'dot_product',
    'negative_sampling_ratio': 1.0,
    'use_edge_features': False,
    'edge_feature_dim': 32,
    'prediction_threshold': 0.5,
    'use_structural_features': True,
    'learning_rate': 0.001,
    'epochs': 100,
    'batch_size': 32,
    'optimizer': 'adam',
    'loss_function': 'bce',
    'weight_decay': 0.0001,
    'scheduler': 'none',
    'early_stopping_patience': 10,
    'dataset': 'cora',
    'validation_split': 0.2,
    'test_split': 0.2,
    'use_node_features': True,
    'temporal_split': False,
    'time_window': 30
}

# Export for compatibility with __init__.py
DEFAULT_PARAMETERS = DEFAULT_LINK_PREDICTION_PARAMS

# Parameter categories for UI organization
PARAMETER_CATEGORIES = {
    'architecture': {
        'name': 'Architecture Parameters',
        'description': 'Model architecture configuration',
        'icon': 'fas fa-cogs',
        'color': 'primary'
    },
    'link_prediction': {
        'name': 'Link Prediction Parameters',
        'description': 'Parameters specific to link prediction',
        'icon': 'fas fa-link',
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
