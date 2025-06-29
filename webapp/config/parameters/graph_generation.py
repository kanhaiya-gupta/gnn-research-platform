"""
Graph Generation Parameters
This module defines parameters specific to graph generation tasks.
"""

GRAPH_GENERATION_PARAMETERS = {
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
    
    # Graph Generation Specific Parameters
    'generation_method': {
        'name': 'Generation Method',
        'description': 'Method for graph generation',
        'type': 'select',
        'default': 'vae',
        'options': ['vae', 'gan', 'flow', 'diffusion', 'autoregressive'],
        'category': 'graph_generation',
        'required': True
    },
    'max_nodes': {
        'name': 'Maximum Nodes',
        'description': 'Maximum number of nodes in generated graphs',
        'type': 'int',
        'default': 50,
        'min': 5,
        'max': 200,
        'step': 5,
        'category': 'graph_generation',
        'required': True
    },
    'max_edges': {
        'name': 'Maximum Edges',
        'description': 'Maximum number of edges in generated graphs',
        'type': 'int',
        'default': 200,
        'min': 10,
        'max': 1000,
        'step': 10,
        'category': 'graph_generation',
        'required': True
    },
    'node_feature_dim': {
        'name': 'Node Feature Dimension',
        'description': 'Dimension of node features in generated graphs',
        'type': 'int',
        'default': 32,
        'min': 8,
        'max': 256,
        'step': 8,
        'category': 'graph_generation',
        'required': False
    },
    'edge_feature_dim': {
        'name': 'Edge Feature Dimension',
        'description': 'Dimension of edge features in generated graphs',
        'type': 'int',
        'default': 16,
        'min': 4,
        'max': 128,
        'step': 4,
        'category': 'graph_generation',
        'required': False
    },
    'use_node_features': {
        'name': 'Generate Node Features',
        'description': 'Whether to generate node features',
        'type': 'bool',
        'default': True,
        'category': 'graph_generation',
        'required': False
    },
    'use_edge_features': {
        'name': 'Generate Edge Features',
        'description': 'Whether to generate edge features',
        'type': 'bool',
        'default': False,
        'category': 'graph_generation',
        'required': False
    },
    'latent_dim': {
        'name': 'Latent Dimension',
        'description': 'Dimension of latent space',
        'type': 'int',
        'default': 32,
        'min': 8,
        'max': 256,
        'step': 8,
        'category': 'graph_generation',
        'required': True
    },
    'temperature': {
        'name': 'Temperature',
        'description': 'Temperature for sampling from distributions',
        'type': 'float',
        'default': 1.0,
        'min': 0.1,
        'max': 5.0,
        'step': 0.1,
        'category': 'graph_generation',
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
        'description': 'Loss function for graph generation',
        'type': 'select',
        'default': 'reconstruction_loss',
        'options': ['reconstruction_loss', 'adversarial_loss', 'wasserstein_loss', 'flow_loss'],
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
        'description': 'Select a graph dataset for training the generator',
        'type': 'select',
        'default': 'zinc',
        'options': [
            'zinc', 'qm9', 'qm7', 'mutag', 'ptc_mr', 'enzymes',
            'proteins', 'nci1', 'nci109', 'reddit_binary', 'reddit_multi_5k'
        ],
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
    'beta': {
        'name': 'Beta',
        'description': 'Beta parameter for VAE (KL divergence weight)',
        'type': 'float',
        'default': 1.0,
        'min': 0.0,
        'max': 10.0,
        'step': 0.1,
        'category': 'model_specific',
        'required': False,
        'models': ['vae']
    },
    'discriminator_layers': {
        'name': 'Discriminator Layers',
        'description': 'Number of layers in discriminator (for GAN)',
        'type': 'int',
        'default': 3,
        'min': 1,
        'max': 10,
        'step': 1,
        'category': 'model_specific',
        'required': False,
        'models': ['gan']
    },
    'critic_iterations': {
        'name': 'Critic Iterations',
        'description': 'Number of critic iterations per generator iteration (for WGAN)',
        'type': 'int',
        'default': 5,
        'min': 1,
        'max': 20,
        'step': 1,
        'category': 'model_specific',
        'required': False,
        'models': ['gan']
    },
    'flow_steps': {
        'name': 'Flow Steps',
        'description': 'Number of flow steps (for Flow-based models)',
        'type': 'int',
        'default': 4,
        'min': 1,
        'max': 20,
        'step': 1,
        'category': 'model_specific',
        'required': False,
        'models': ['flow']
    },
    'diffusion_steps': {
        'name': 'Diffusion Steps',
        'description': 'Number of diffusion steps (for Diffusion models)',
        'type': 'int',
        'default': 1000,
        'min': 100,
        'max': 5000,
        'step': 100,
        'category': 'model_specific',
        'required': False,
        'models': ['diffusion']
    }
}

# Default parameters for quick start
DEFAULT_GRAPH_GENERATION_PARAMS = {
    'hidden_dim': 64,
    'num_layers': 3,
    'dropout': 0.1,
    'activation': 'relu',
    'batch_norm': True,
    'residual': False,
    'generation_method': 'vae',
    'max_nodes': 50,
    'max_edges': 200,
    'node_feature_dim': 32,
    'edge_feature_dim': 16,
    'use_node_features': True,
    'use_edge_features': False,
    'latent_dim': 32,
    'temperature': 1.0,
    'learning_rate': 0.001,
    'epochs': 100,
    'batch_size': 32,
    'optimizer': 'adam',
    'loss_function': 'reconstruction_loss',
    'weight_decay': 0.0001,
    'scheduler': 'none',
    'early_stopping_patience': 10,
    'dataset': 'zinc',
    'validation_split': 0.2,
    'test_split': 0.2,
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
    'graph_generation': {
        'name': 'Graph Generation Parameters',
        'description': 'Parameters specific to graph generation',
        'icon': 'fas fa-magic',
        'color': 'purple'
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
