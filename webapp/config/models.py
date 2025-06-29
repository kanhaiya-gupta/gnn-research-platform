"""
Model Configuration
This module defines the available GNN models and their configurations.
"""

# Import task registry to get supported models for each task
from .tasks import get_task_config

# Model definitions with their architectures and parameters
GNN_MODELS = {
    'gcn': {
        'name': 'Graph Convolutional Network',
        'description': 'Spectral-based graph convolution using Chebyshev polynomials',
        'paper': 'Kipf & Welling, ICLR 2017',
        'category': 'convolutional',
        'architecture': 'spectral',
        'inductive': False,
        'supported_tasks': ['node_classification', 'link_prediction', 'graph_classification', 'graph_regression'],
        
        # Model-specific parameters
        'parameters': {
            'hidden_dim': {'default': 64, 'range': [16, 512]},
            'num_layers': {'default': 3, 'range': [1, 10]},
            'dropout': {'default': 0.1, 'range': [0.0, 0.5]},
            'activation': {'default': 'relu', 'options': ['relu', 'tanh', 'sigmoid']},
            'bias': {'default': True, 'type': 'bool'},
            'normalize': {'default': True, 'type': 'bool'},
            'cached': {'default': True, 'type': 'bool'}
        },
        
        # Implementation details
        'implementation': {
            'framework': 'pytorch_geometric',
            'class': 'GCNConv',
            'optimization': 'efficient',
            'memory_efficient': True
        }
    },
    
    'gat': {
        'name': 'Graph Attention Network',
        'description': 'Attention-based graph convolution with multi-head attention',
        'paper': 'Veličković et al., ICLR 2018',
        'category': 'attention',
        'architecture': 'attention',
        'inductive': False,
        'supported_tasks': ['node_classification', 'link_prediction', 'graph_classification', 'graph_regression'],
        
        # Model-specific parameters
        'parameters': {
            'hidden_dim': {'default': 64, 'range': [16, 512]},
            'num_layers': {'default': 3, 'range': [1, 10]},
            'dropout': {'default': 0.1, 'range': [0.0, 0.5]},
            'activation': {'default': 'relu', 'options': ['relu', 'tanh', 'sigmoid']},
            'attention_heads': {'default': 8, 'range': [1, 16]},
            'attention_dropout': {'default': 0.1, 'range': [0.0, 0.5]},
            'concat': {'default': True, 'type': 'bool'},
            'negative_slope': {'default': 0.2, 'range': [0.0, 1.0]}
        },
        
        # Implementation details
        'implementation': {
            'framework': 'pytorch_geometric',
            'class': 'GATConv',
            'optimization': 'attention',
            'memory_efficient': False
        }
    },
    
    'graphsage': {
        'name': 'GraphSAGE',
        'description': 'Inductive graph representation learning with neighbor sampling',
        'paper': 'Hamilton et al., NeurIPS 2017',
        'category': 'inductive',
        'architecture': 'spatial',
        'inductive': True,
        'supported_tasks': ['node_classification', 'link_prediction', 'graph_classification', 'graph_regression'],
        
        # Model-specific parameters
        'parameters': {
            'hidden_dim': {'default': 64, 'range': [16, 512]},
            'num_layers': {'default': 3, 'range': [1, 10]},
            'dropout': {'default': 0.1, 'range': [0.0, 0.5]},
            'activation': {'default': 'relu', 'options': ['relu', 'tanh', 'sigmoid']},
            'neighbor_sampling': {'default': 25, 'range': [5, 100]},
            'aggregator_type': {'default': 'mean', 'options': ['mean', 'max', 'sum', 'lstm']},
            'bias': {'default': True, 'type': 'bool'}
        },
        
        # Implementation details
        'implementation': {
            'framework': 'pytorch_geometric',
            'class': 'SAGEConv',
            'optimization': 'sampling',
            'memory_efficient': True
        }
    },
    
    'gin': {
        'name': 'Graph Isomorphism Network',
        'description': 'Graph isomorphism network with injective aggregation',
        'paper': 'Xu et al., ICLR 2019',
        'category': 'convolutional',
        'architecture': 'spatial',
        'inductive': True,
        'supported_tasks': ['node_classification', 'graph_classification', 'graph_regression'],
        
        # Model-specific parameters
        'parameters': {
            'hidden_dim': {'default': 64, 'range': [16, 512]},
            'num_layers': {'default': 3, 'range': [1, 10]},
            'dropout': {'default': 0.1, 'range': [0.0, 0.5]},
            'activation': {'default': 'relu', 'options': ['relu', 'tanh', 'sigmoid']},
            'epsilon': {'default': 0.0, 'range': [0.0, 1.0]},
            'train_eps': {'default': True, 'type': 'bool'},
            'bias': {'default': True, 'type': 'bool'}
        },
        
        # Implementation details
        'implementation': {
            'framework': 'pytorch_geometric',
            'class': 'GINConv',
            'optimization': 'efficient',
            'memory_efficient': True
        }
    },
    
    'chebnet': {
        'name': 'Chebyshev Graph Convolution',
        'description': 'Spectral graph convolution using Chebyshev polynomials',
        'paper': 'Defferrard et al., NeurIPS 2016',
        'category': 'convolutional',
        'architecture': 'spectral',
        'inductive': False,
        'supported_tasks': ['node_classification', 'link_prediction', 'graph_classification'],
        
        # Model-specific parameters
        'parameters': {
            'hidden_dim': {'default': 64, 'range': [16, 512]},
            'num_layers': {'default': 3, 'range': [1, 10]},
            'dropout': {'default': 0.1, 'range': [0.0, 0.5]},
            'activation': {'default': 'relu', 'options': ['relu', 'tanh', 'sigmoid']},
            'k_hop': {'default': 3, 'range': [1, 10]},
            'bias': {'default': True, 'type': 'bool'}
        },
        
        # Implementation details
        'implementation': {
            'framework': 'pytorch_geometric',
            'class': 'ChebConv',
            'optimization': 'polynomial',
            'memory_efficient': True
        }
    },
    
    'vgae': {
        'name': 'Variational Graph Autoencoder',
        'description': 'Variational autoencoder for graph generation and link prediction',
        'paper': 'Kipf & Welling, ICLR 2016',
        'category': 'generative',
        'architecture': 'autoencoder',
        'inductive': False,
        'supported_tasks': ['link_prediction', 'graph_generation'],
        
        # Model-specific parameters
        'parameters': {
            'hidden_dim': {'default': 64, 'range': [16, 512]},
            'num_layers': {'default': 3, 'range': [1, 10]},
            'dropout': {'default': 0.1, 'range': [0.0, 0.5]},
            'activation': {'default': 'relu', 'options': ['relu', 'tanh', 'sigmoid']},
            'latent_dim': {'default': 32, 'range': [8, 256]},
            'beta': {'default': 1.0, 'range': [0.0, 10.0]},
            'bias': {'default': True, 'type': 'bool'}
        },
        
        # Implementation details
        'implementation': {
            'framework': 'pytorch_geometric',
            'class': 'VGAE',
            'optimization': 'variational',
            'memory_efficient': True
        }
    },
    
    'seal': {
        'name': 'SEAL',
        'description': 'Subgraphs, Embeddings and Attributes for Link prediction',
        'paper': 'Zhang & Chen, NeurIPS 2018',
        'category': 'link_prediction',
        'architecture': 'subgraph',
        'inductive': True,
        'supported_tasks': ['link_prediction'],
        
        # Model-specific parameters
        'parameters': {
            'hidden_dim': {'default': 64, 'range': [16, 512]},
            'num_layers': {'default': 3, 'range': [1, 10]},
            'dropout': {'default': 0.1, 'range': [0.0, 0.5]},
            'activation': {'default': 'relu', 'options': ['relu', 'tanh', 'sigmoid']},
            'subgraph_size': {'default': 20, 'range': [5, 100]},
            'use_structural_features': {'default': True, 'type': 'bool'},
            'use_node_features': {'default': True, 'type': 'bool'}
        },
        
        # Implementation details
        'implementation': {
            'framework': 'custom',
            'class': 'SEAL',
            'optimization': 'subgraph',
            'memory_efficient': False
        }
    },
    
    'sgc': {
        'name': 'Simple Graph Convolution',
        'description': 'Simple graph convolution with pre-computed features',
        'paper': 'Wu et al., ICML 2019',
        'category': 'convolutional',
        'architecture': 'spectral',
        'inductive': False,
        'supported_tasks': ['node_classification', 'graph_classification'],
        
        # Model-specific parameters
        'parameters': {
            'hidden_dim': {'default': 64, 'range': [16, 512]},
            'num_layers': {'default': 3, 'range': [1, 10]},
            'dropout': {'default': 0.1, 'range': [0.0, 0.5]},
            'activation': {'default': 'relu', 'options': ['relu', 'tanh', 'sigmoid']},
            'k_hop': {'default': 2, 'range': [1, 5]},
            'cached': {'default': True, 'type': 'bool'}
        },
        
        # Implementation details
        'implementation': {
            'framework': 'pytorch_geometric',
            'class': 'SGConv',
            'optimization': 'precomputed',
            'memory_efficient': True
        }
    },
    
    'appnp': {
        'name': 'APPNP',
        'description': 'Approximate Personalized Propagation of Neural Predictions',
        'paper': 'Klicpera et al., ICLR 2019',
        'category': 'propagation',
        'architecture': 'message_passing',
        'inductive': False,
        'supported_tasks': ['node_classification'],
        
        # Model-specific parameters
        'parameters': {
            'hidden_dim': {'default': 64, 'range': [16, 512]},
            'num_layers': {'default': 3, 'range': [1, 10]},
            'dropout': {'default': 0.1, 'range': [0.0, 0.5]},
            'activation': {'default': 'relu', 'options': ['relu', 'tanh', 'sigmoid']},
            'alpha': {'default': 0.1, 'range': [0.0, 1.0]},
            'k_hop': {'default': 10, 'range': [1, 20]},
            'cached': {'default': True, 'type': 'bool'}
        },
        
        # Implementation details
        'implementation': {
            'framework': 'pytorch_geometric',
            'class': 'APPNP',
            'optimization': 'propagation',
            'memory_efficient': True
        }
    }
}

# Model categories for organization
MODEL_CATEGORIES = {
    'convolutional': {
        'name': 'Convolutional Models',
        'description': 'Models based on graph convolution operations',
        'icon': 'fas fa-filter',
        'color': 'primary',
        'models': ['gcn', 'gin', 'chebnet', 'sgc']
    },
    'attention': {
        'name': 'Attention Models',
        'description': 'Models using attention mechanisms',
        'icon': 'fas fa-eye',
        'color': 'info',
        'models': ['gat']
    },
    'inductive': {
        'name': 'Inductive Models',
        'description': 'Models that can generalize to unseen nodes',
        'icon': 'fas fa-arrows-alt',
        'color': 'success',
        'models': ['graphsage', 'gin']
    },
    'generative': {
        'name': 'Generative Models',
        'description': 'Models for graph generation',
        'icon': 'fas fa-magic',
        'color': 'purple',
        'models': ['vgae']
    },
    'link_prediction': {
        'name': 'Link Prediction Models',
        'description': 'Specialized models for link prediction',
        'icon': 'fas fa-link',
        'color': 'warning',
        'models': ['seal']
    },
    'propagation': {
        'name': 'Propagation Models',
        'description': 'Models based on message propagation',
        'icon': 'fas fa-share-alt',
        'color': 'secondary',
        'models': ['appnp']
    }
}

def get_model_config(model_name):
    """Get model configuration by name."""
    return GNN_MODELS.get(model_name)

def get_models_by_category(category):
    """Get models by category."""
    return MODEL_CATEGORIES.get(category, {}).get('models', [])

def get_models_for_task(task_name):
    """Get models that support a specific task."""
    task_config = get_task_config(task_name)
    if task_config:
        return task_config.get('supported_models', [])
    return []

def get_model_parameters(model_name):
    """Get parameters for a specific model."""
    model_config = get_model_config(model_name)
    return model_config.get('parameters', {}) if model_config else {}

def is_model_available_for_task(model_name, task_name):
    """Check if a model supports a specific task."""
    model_config = get_model_config(model_name)
    if model_config:
        return task_name in model_config.get('supported_tasks', [])
    return False

def get_all_models():
    """Get all available models."""
    return GNN_MODELS

def get_all_categories():
    """Get all model categories."""
    return MODEL_CATEGORIES

def get_model_metadata(model_name):
    """Get metadata for a specific model."""
    model_config = get_model_config(model_name)
    if model_config:
        return {
            'name': model_config['name'],
            'description': model_config['description'],
            'paper': model_config['paper'],
            'category': model_config['category'],
            'architecture': model_config['architecture'],
            'inductive': model_config['inductive']
        }
    return None 