"""
Anomaly Detection Parameters
This module defines parameters specific to anomaly detection tasks.
"""

ANOMALY_DETECTION_PARAMETERS = {
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
    
    # Anomaly Detection Specific Parameters
    'detection_type': {
        'name': 'Detection Type',
        'description': 'Type of anomaly to detect',
        'type': 'select',
        'default': 'node',
        'options': ['node', 'edge', 'subgraph', 'graph'],
        'category': 'anomaly_detection',
        'required': True
    },
    'detection_method': {
        'name': 'Detection Method',
        'description': 'Method for anomaly detection',
        'type': 'select',
        'default': 'autoencoder',
        'options': ['autoencoder', 'one_class_svm', 'isolation_forest', 'local_outlier_factor', 'reconstruction_error'],
        'category': 'anomaly_detection',
        'required': True
    },
    'anomaly_threshold': {
        'name': 'Anomaly Threshold',
        'description': 'Threshold for anomaly detection',
        'type': 'float',
        'default': 0.95,
        'min': 0.5,
        'max': 0.99,
        'step': 0.01,
        'category': 'anomaly_detection',
        'required': True
    },
    'contamination': {
        'name': 'Contamination',
        'description': 'Expected proportion of anomalies in the dataset',
        'type': 'float',
        'default': 0.1,
        'min': 0.01,
        'max': 0.5,
        'step': 0.01,
        'category': 'anomaly_detection',
        'required': False
    },
    'use_reconstruction_error': {
        'name': 'Use Reconstruction Error',
        'description': 'Whether to use reconstruction error for anomaly detection',
        'type': 'bool',
        'default': True,
        'category': 'anomaly_detection',
        'required': False
    },
    'use_structural_features': {
        'name': 'Use Structural Features',
        'description': 'Whether to use structural features for anomaly detection',
        'type': 'bool',
        'default': True,
        'category': 'anomaly_detection',
        'required': False
    },
    'embedding_dim': {
        'name': 'Embedding Dimension',
        'description': 'Dimension of node embeddings for anomaly detection',
        'type': 'int',
        'default': 32,
        'min': 8,
        'max': 256,
        'step': 8,
        'category': 'anomaly_detection',
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
        'description': 'Loss function for anomaly detection',
        'type': 'select',
        'default': 'reconstruction_loss',
        'options': ['reconstruction_loss', 'contrastive_loss', 'margin_loss', 'one_class_loss'],
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
        'description': 'Select a graph dataset for anomaly detection',
        'type': 'select',
        'default': 'cora',
        'options': [
            'cora', 'citeseer', 'pubmed', 'reddit', 'flickr',
            'amazon_photo', 'amazon_computers', 'ogbn_arxiv', 'ogbn_products'
        ],
        'category': 'dataset',
        'required': True
    },
    'anomaly_ratio': {
        'name': 'Anomaly Ratio',
        'description': 'Ratio of anomalies to inject into the dataset',
        'type': 'float',
        'default': 0.1,
        'min': 0.01,
        'max': 0.5,
        'step': 0.01,
        'category': 'dataset',
        'required': True
    },
    'anomaly_type': {
        'name': 'Anomaly Type',
        'description': 'Type of anomalies to inject',
        'type': 'select',
        'default': 'structural',
        'options': ['structural', 'feature', 'mixed', 'random'],
        'category': 'dataset',
        'required': True
    },
    'validation_split': {
        'name': 'Validation Split',
        'description': 'Fraction of data for validation',
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
        'description': 'Fraction of data for testing',
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
    'latent_dim': {
        'name': 'Latent Dimension',
        'description': 'Latent dimension for autoencoder',
        'type': 'int',
        'default': 32,
        'min': 8,
        'max': 256,
        'step': 8,
        'category': 'model_specific',
        'required': False,
        'models': ['autoencoder']
    },
    'reconstruction_weight': {
        'name': 'Reconstruction Weight',
        'description': 'Weight for reconstruction loss in autoencoder',
        'type': 'float',
        'default': 1.0,
        'min': 0.1,
        'max': 10.0,
        'step': 0.1,
        'category': 'model_specific',
        'required': False,
        'models': ['autoencoder']
    }
}

# Default parameters for quick start
DEFAULT_ANOMALY_DETECTION_PARAMS = {
    'hidden_dim': 64,
    'num_layers': 3,
    'dropout': 0.1,
    'activation': 'relu',
    'batch_norm': True,
    'residual': False,
    'detection_type': 'node',
    'detection_method': 'autoencoder',
    'anomaly_threshold': 0.95,
    'contamination': 0.1,
    'use_reconstruction_error': True,
    'use_structural_features': True,
    'embedding_dim': 32,
    'learning_rate': 0.001,
    'epochs': 100,
    'batch_size': 32,
    'optimizer': 'adam',
    'loss_function': 'reconstruction_loss',
    'weight_decay': 0.0001,
    'scheduler': 'none',
    'early_stopping_patience': 10,
    'dataset': 'cora',
    'anomaly_ratio': 0.1,
    'anomaly_type': 'structural',
    'validation_split': 0.2,
    'test_split': 0.2,
    'use_node_features': True,
    'use_edge_features': False,
    'normalize_features': True
}

# Parameter categories for UI organization
PARAMETER_CATEGORIES = {
    'architecture': {
        'name': 'Architecture Parameters',
        'description': 'Model architecture configuration',
        'icon': 'fas fa-cogs',
        'color': 'primary'
    },
    'anomaly_detection': {
        'name': 'Anomaly Detection Parameters',
        'description': 'Parameters specific to anomaly detection',
        'icon': 'fas fa-exclamation-triangle',
        'color': 'danger'
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
