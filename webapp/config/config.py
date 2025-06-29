"""
Main Configuration File
This module contains the main configuration for the GNN platform.
"""

import os
from pathlib import Path
from typing import Dict, Any

# Base directory configuration
BASE_DIR = Path(__file__).parent.parent.parent
WEBAPP_DIR = BASE_DIR / 'webapp'
STATIC_DIR = WEBAPP_DIR / 'static'
TEMPLATES_DIR = WEBAPP_DIR / 'templates'
CONFIG_DIR = WEBAPP_DIR / 'config'

# Platform configuration
PLATFORM_CONFIG = {
    'name': 'Graph Neural Network Platform',
    'version': '1.0.0',
    'description': 'A comprehensive platform for Graph Neural Network experiments',
    'author': 'GNN Platform Team',
    'contact': 'support@gnnplatform.com',
    
    # UI Configuration
    'ui': {
        'theme': 'default',
        'color_scheme': 'light',
        'sidebar_collapsed': False,
        'show_tutorials': True,
        'show_examples': True,
        'enable_dark_mode': True,
        'enable_animations': True
    },
    
    # Feature flags
    'features': {
        'enable_real_time_training': True,
        'enable_model_comparison': True,
        'enable_parameter_tuning': True,
        'enable_visualization': True,
        'enable_export': True,
        'enable_collaboration': True,
        'enable_versioning': True
    },
    
    # Security configuration
    'security': {
        'enable_authentication': False,
        'enable_authorization': False,
        'session_timeout': 3600,
        'max_file_size': 100 * 1024 * 1024,  # 100MB
        'allowed_file_types': ['.csv', '.json', '.pkl', '.pt', '.pth']
    },
    
    # Performance configuration
    'performance': {
        'max_concurrent_experiments': 5,
        'max_training_time': 7200,  # 2 hours
        'max_memory_usage': 8 * 1024 * 1024 * 1024,  # 8GB
        'enable_caching': True,
        'cache_size': 1000,
        'enable_compression': True
    }
}

# Dataset configuration
DATASET_CONFIG = {
    'default_datasets': {
        'node_classification': ['cora', 'citeseer', 'pubmed'],
        'link_prediction': ['cora', 'citeseer', 'pubmed'],
        'graph_classification': ['mutag', 'ptc_mr', 'enzymes'],
        'graph_regression': ['zinc', 'qm9', 'qm7'],
        'community_detection': ['karate', 'football', 'polbooks'],
        'anomaly_detection': ['cora', 'citeseer', 'pubmed'],
        'graph_generation': ['zinc', 'qm9', 'mutag'],
        'graph_embedding': ['cora', 'citeseer', 'pubmed'],
        'dynamic_graph_learning': ['enron', 'uc_irvine', 'facebook']
    },
    
    'dataset_metadata': {
        'cora': {
            'name': 'Cora',
            'description': 'Citation network dataset',
            'type': 'citation',
            'nodes': 2708,
            'edges': 5429,
            'features': 1433,
            'classes': 7
        },
        'citeseer': {
            'name': 'CiteSeer',
            'description': 'Citation network dataset',
            'type': 'citation',
            'nodes': 3327,
            'edges': 4732,
            'features': 3703,
            'classes': 6
        },
        'pubmed': {
            'name': 'PubMed',
            'description': 'Biomedical citation network',
            'type': 'citation',
            'nodes': 19717,
            'edges': 44338,
            'features': 500,
            'classes': 3
        },
        'zinc': {
            'name': 'ZINC',
            'description': 'Molecular dataset',
            'type': 'molecular',
            'graphs': 12000,
            'avg_nodes': 23,
            'avg_edges': 49,
            'task': 'regression'
        }
    }
}

# Model configuration
MODEL_CONFIG = {
    'supported_models': {
        'gcn': {
            'name': 'Graph Convolutional Network',
            'description': 'Spectral-based graph convolution',
            'category': 'convolutional',
            'supported_tasks': ['node_classification', 'link_prediction', 'graph_classification']
        },
        'gat': {
            'name': 'Graph Attention Network',
            'description': 'Attention-based graph convolution',
            'category': 'attention',
            'supported_tasks': ['node_classification', 'link_prediction', 'graph_classification']
        },
        'graphsage': {
            'name': 'GraphSAGE',
            'description': 'Inductive graph representation learning',
            'category': 'inductive',
            'supported_tasks': ['node_classification', 'link_prediction', 'graph_classification']
        },
        'gin': {
            'name': 'Graph Isomorphism Network',
            'description': 'Graph isomorphism network',
            'category': 'convolutional',
            'supported_tasks': ['node_classification', 'graph_classification', 'graph_regression']
        },
        'vgae': {
            'name': 'Variational Graph Autoencoder',
            'description': 'Variational autoencoder for graphs',
            'category': 'generative',
            'supported_tasks': ['link_prediction', 'graph_generation']
        }
    },
    
    'default_hyperparameters': {
        'hidden_dim': 64,
        'num_layers': 3,
        'dropout': 0.1,
        'learning_rate': 0.001,
        'epochs': 100,
        'batch_size': 32,
        'optimizer': 'adam',
        'activation': 'relu'
    }
}

# Training configuration
TRAINING_CONFIG = {
    'default_settings': {
        'validation_split': 0.2,
        'test_split': 0.2,
        'random_state': 42,
        'early_stopping_patience': 10,
        'checkpoint_frequency': 10,
        'save_best_model': True,
        'save_predictions': True
    },
    
    'optimizers': {
        'adam': {
            'name': 'Adam',
            'description': 'Adaptive moment estimation',
            'default_lr': 0.001,
            'default_beta1': 0.9,
            'default_beta2': 0.999
        },
        'sgd': {
            'name': 'SGD',
            'description': 'Stochastic gradient descent',
            'default_lr': 0.01,
            'default_momentum': 0.9
        },
        'adamw': {
            'name': 'AdamW',
            'description': 'Adam with weight decay',
            'default_lr': 0.001,
            'default_weight_decay': 0.01
        }
    },
    
    'schedulers': {
        'none': {
            'name': 'No Scheduler',
            'description': 'Constant learning rate'
        },
        'step': {
            'name': 'Step LR',
            'description': 'Step learning rate scheduler',
            'default_step_size': 30,
            'default_gamma': 0.1
        },
        'cosine': {
            'name': 'Cosine Annealing',
            'description': 'Cosine annealing scheduler',
            'default_t_max': 100
        }
    }
}

# Visualization configuration
VISUALIZATION_CONFIG = {
    'enabled_visualizations': [
        'training_curves',
        'confusion_matrix',
        'roc_curve',
        'precision_recall_curve',
        'embedding_visualization',
        'attention_weights',
        'graph_structure',
        'feature_importance'
    ],
    
    'plot_settings': {
        'figure_size': (10, 6),
        'dpi': 100,
        'style': 'seaborn',
        'color_palette': 'viridis',
        'save_format': 'png'
    },
    
    'interactive_plots': {
        'enable_zoom': True,
        'enable_pan': True,
        'enable_hover': True,
        'enable_selection': True
    }
}

# Export configuration
EXPORT_CONFIG = {
    'supported_formats': {
        'json': {
            'name': 'JSON',
            'description': 'JavaScript Object Notation',
            'extension': '.json',
            'compression': False
        },
        'csv': {
            'name': 'CSV',
            'description': 'Comma-separated values',
            'extension': '.csv',
            'compression': False
        },
        'pkl': {
            'name': 'Pickle',
            'description': 'Python pickle format',
            'extension': '.pkl',
            'compression': True
        },
        'pt': {
            'name': 'PyTorch',
            'description': 'PyTorch model format',
            'extension': '.pt',
            'compression': True
        }
    },
    
    'export_settings': {
        'include_metadata': True,
        'include_predictions': True,
        'include_embeddings': True,
        'include_visualizations': True,
        'compress_output': True
    }
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'gnn_platform.log',
    'max_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5,
    'console_output': True,
    'file_output': True
}

# Development configuration
DEV_CONFIG = {
    'debug_mode': False,
    'enable_profiling': False,
    'enable_tracing': False,
    'log_level': 'INFO',
    'auto_reload': True,
    'show_debug_toolbar': False
}

# Environment-specific configurations
def get_config(environment='development'):
    """Get configuration for a specific environment."""
    configs = {
        'development': {
            'debug': True,
            'host': 'localhost',
            'port': 5000,
            'database': 'sqlite:///dev.db',
            'cache': 'memory'
        },
        'production': {
            'debug': False,
            'host': '0.0.0.0',
            'port': 80,
            'database': 'postgresql://user:pass@localhost/gnn_platform',
            'cache': 'redis://localhost:6379'
        },
        'testing': {
            'debug': True,
            'host': 'localhost',
            'port': 5001,
            'database': 'sqlite:///test.db',
            'cache': 'memory'
        }
    }
    return configs.get(environment, configs['development'])

# Export all configurations
__all__ = [
    'PLATFORM_CONFIG',
    'DATASET_CONFIG',
    'MODEL_CONFIG',
    'TRAINING_CONFIG',
    'VISUALIZATION_CONFIG',
    'EXPORT_CONFIG',
    'LOGGING_CONFIG',
    'DEV_CONFIG',
    'get_config'
]

class Config:
    """Configuration class for GNN web application."""
    
    def __init__(self):
        # Base directory
        self.BASE_DIR = Path(__file__).parent.parent.parent
        self.WEBAPP_DIR = self.BASE_DIR / 'webapp'
        
        # API configuration
        self.API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
        
        # GNN purposes with metadata
        self.GNN_PURPOSES = {
            'node_classification': {
                'name': 'Node Classification',
                'description': 'Classify nodes into categories based on features and graph structure',
                'icon': 'fas fa-project-diagram',
                'color': 'primary'
            },
            'node_regression': {
                'name': 'Node Regression',
                'description': 'Predict continuous node attributes using graph structure',
                'icon': 'fas fa-chart-line',
                'color': 'success'
            },
            'edge_classification': {
                'name': 'Edge Classification',
                'description': 'Classify edges or relationships between nodes',
                'icon': 'fas fa-link',
                'color': 'info'
            },
            'link_prediction': {
                'name': 'Link Prediction',
                'description': 'Predict missing or future edges in a graph',
                'icon': 'fas fa-link',
                'color': 'warning'
            },
            'graph_classification': {
                'name': 'Graph Classification',
                'description': 'Classify entire graphs based on structural patterns',
                'icon': 'fas fa-cubes',
                'color': 'secondary'
            },
            'graph_regression': {
                'name': 'Graph Regression',
                'description': 'Predict continuous graph-level attributes',
                'icon': 'fas fa-wave-square',
                'color': 'dark'
            },
            'community_detection': {
                'name': 'Community Detection',
                'description': 'Identify clusters or communities within graphs',
                'icon': 'fas fa-users',
                'color': 'danger'
            },
            'anomaly_detection': {
                'name': 'Anomaly Detection',
                'description': 'Detect anomalous nodes or edges in graphs',
                'icon': 'fas fa-exclamation-triangle',
                'color': 'primary'
            },
            'dynamic_graph_learning': {
                'name': 'Dynamic Graph Learning',
                'description': 'Learn from graphs that evolve over time',
                'icon': 'fas fa-sync-alt',
                'color': 'success'
            },
            'graph_generation': {
                'name': 'Graph Generation',
                'description': 'Generate new graphs with desired properties',
                'icon': 'fas fa-draw-polygon',
                'color': 'info'
            },
            'graph_embedding_visualization': {
                'name': 'Graph Embedding & Visualization',
                'description': 'Embed graph elements into vector spaces and visualize',
                'icon': 'fas fa-vector-square',
                'color': 'warning'
            }
        }
        
        # Models for each purpose
        self.MODELS_BY_PURPOSE = {
            'node_classification': {
                'gcn': {
                    'name': 'Graph Convolutional Network',
                    'description': 'Spectral-based graph convolution',
                    'category': 'convolutional'
                },
                'gat': {
                    'name': 'Graph Attention Network',
                    'description': 'Attention-based graph convolution',
                    'category': 'attention'
                },
                'graphsage': {
                    'name': 'GraphSAGE',
                    'description': 'Inductive graph representation learning',
                    'category': 'inductive'
                }
            },
            'node_regression': {
                'gcn': {
                    'name': 'Graph Convolutional Network',
                    'description': 'Spectral-based graph convolution',
                    'category': 'convolutional'
                },
                'gin': {
                    'name': 'Graph Isomorphism Network',
                    'description': 'Graph isomorphism network',
                    'category': 'convolutional'
                }
            },
            'link_prediction': {
                'gcn': {
                    'name': 'Graph Convolutional Network',
                    'description': 'Spectral-based graph convolution',
                    'category': 'convolutional'
                },
                'vgae': {
                    'name': 'Variational Graph Autoencoder',
                    'description': 'Variational autoencoder for graphs',
                    'category': 'generative'
                }
            },
            'graph_classification': {
                'gcn': {
                    'name': 'Graph Convolutional Network',
                    'description': 'Spectral-based graph convolution',
                    'category': 'convolutional'
                },
                'gin': {
                    'name': 'Graph Isomorphism Network',
                    'description': 'Graph isomorphism network',
                    'category': 'convolutional'
                }
            },
            'graph_regression': {
                'gin': {
                    'name': 'Graph Isomorphism Network',
                    'description': 'Graph isomorphism network',
                    'category': 'convolutional'
                }
            },
            'community_detection': {
                'gcn': {
                    'name': 'Graph Convolutional Network',
                    'description': 'Spectral-based graph convolution',
                    'category': 'convolutional'
                }
            },
            'anomaly_detection': {
                'gcn': {
                    'name': 'Graph Convolutional Network',
                    'description': 'Spectral-based graph convolution',
                    'category': 'convolutional'
                }
            },
            'dynamic_graph_learning': {
                'gcn': {
                    'name': 'Graph Convolutional Network',
                    'description': 'Spectral-based graph convolution',
                    'category': 'convolutional'
                }
            },
            'graph_generation': {
                'vgae': {
                    'name': 'Variational Graph Autoencoder',
                    'description': 'Variational autoencoder for graphs',
                    'category': 'generative'
                }
            },
            'graph_embedding_visualization': {
                'gcn': {
                    'name': 'Graph Convolutional Network',
                    'description': 'Spectral-based graph convolution',
                    'category': 'convolutional'
                }
            }
        }
        
        # Parameters for each purpose
        self.PARAMETERS_BY_PURPOSE = {
            'node_classification': {
                'hidden_dim': {
                    'name': 'Hidden Dimension',
                    'type': 'int',
                    'default': 64,
                    'min': 16,
                    'max': 512,
                    'description': 'Number of hidden units in each layer'
                },
                'num_layers': {
                    'name': 'Number of Layers',
                    'type': 'int',
                    'default': 3,
                    'min': 1,
                    'max': 10,
                    'description': 'Number of graph convolution layers'
                },
                'learning_rate': {
                    'name': 'Learning Rate',
                    'type': 'float',
                    'default': 0.001,
                    'min': 0.0001,
                    'max': 0.1,
                    'description': 'Learning rate for optimization'
                },
                'epochs': {
                    'name': 'Epochs',
                    'type': 'int',
                    'default': 100,
                    'min': 10,
                    'max': 1000,
                    'description': 'Number of training epochs'
                },
                'dropout': {
                    'name': 'Dropout Rate',
                    'type': 'float',
                    'default': 0.1,
                    'min': 0.0,
                    'max': 0.9,
                    'description': 'Dropout rate for regularization'
                }
            },
            'node_regression': {
                'hidden_dim': {
                    'name': 'Hidden Dimension',
                    'type': 'int',
                    'default': 64,
                    'min': 16,
                    'max': 512,
                    'description': 'Number of hidden units in each layer'
                },
                'num_layers': {
                    'name': 'Number of Layers',
                    'type': 'int',
                    'default': 3,
                    'min': 1,
                    'max': 10,
                    'description': 'Number of graph convolution layers'
                },
                'learning_rate': {
                    'name': 'Learning Rate',
                    'type': 'float',
                    'default': 0.001,
                    'min': 0.0001,
                    'max': 0.1,
                    'description': 'Learning rate for optimization'
                },
                'epochs': {
                    'name': 'Epochs',
                    'type': 'int',
                    'default': 100,
                    'min': 10,
                    'max': 1000,
                    'description': 'Number of training epochs'
                }
            },
            'link_prediction': {
                'hidden_dim': {
                    'name': 'Hidden Dimension',
                    'type': 'int',
                    'default': 64,
                    'min': 16,
                    'max': 512,
                    'description': 'Number of hidden units in each layer'
                },
                'num_layers': {
                    'name': 'Number of Layers',
                    'type': 'int',
                    'default': 3,
                    'min': 1,
                    'max': 10,
                    'description': 'Number of graph convolution layers'
                },
                'learning_rate': {
                    'name': 'Learning Rate',
                    'type': 'float',
                    'default': 0.001,
                    'min': 0.0001,
                    'max': 0.1,
                    'description': 'Learning rate for optimization'
                },
                'epochs': {
                    'name': 'Epochs',
                    'type': 'int',
                    'default': 100,
                    'min': 10,
                    'max': 1000,
                    'description': 'Number of training epochs'
                }
            }
        }
    
    def get_all_purposes(self) -> Dict[str, Any]:
        """Get all GNN purposes."""
        return self.GNN_PURPOSES
    
    def get_purpose_info(self, purpose_name: str) -> Dict[str, Any]:
        """Get information about a specific purpose."""
        return self.GNN_PURPOSES.get(purpose_name, {})
    
    def get_models_by_purpose(self, purpose_name: str) -> Dict[str, Any]:
        """Get models for a specific purpose."""
        return self.MODELS_BY_PURPOSE.get(purpose_name, {})
    
    def get_parameters_by_purpose(self, purpose_name: str) -> Dict[str, Any]:
        """Get parameters for a specific purpose."""
        return self.PARAMETERS_BY_PURPOSE.get(purpose_name, {})
    
    def get_model_specific_parameters(self, purpose_name: str, model_id: str) -> Dict[str, Any]:
        """Get model-specific parameters for a purpose and model."""
        # For now, return the general purpose parameters
        # This could be extended to have model-specific parameters
        return self.get_parameters_by_purpose(purpose_name)
    
    def map_parameters_to_backend(self, purpose_name: str, model_id: str, frontend_params: Dict[str, Any]) -> Dict[str, Any]:
        """Map frontend parameters to backend format."""
        # For now, just return the frontend parameters as-is
        # This could be extended to transform parameters for the backend
        return frontend_params 