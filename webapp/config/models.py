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
        'supported_tasks': ['node_classification', 'node_regression', 'link_prediction', 'edge_classification', 'graph_classification', 'graph_regression', 'community_detection'],
        
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
        'supported_tasks': ['node_classification', 'node_regression', 'link_prediction', 'edge_classification', 'graph_classification', 'graph_regression', 'community_detection'],
        
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
        'supported_tasks': ['node_classification', 'node_regression', 'link_prediction', 'edge_classification', 'graph_classification', 'graph_regression', 'community_detection'],
        
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
        'supported_tasks': ['node_classification', 'node_regression', 'graph_classification', 'graph_regression', 'community_detection'],
        
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
        'supported_tasks': ['node_classification', 'node_regression', 'link_prediction', 'edge_classification', 'graph_classification', 'community_detection'],
        
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
        'supported_tasks': ['link_prediction', 'edge_classification', 'graph_generation'],
        
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
        'supported_tasks': ['link_prediction', 'edge_classification'],
        
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
        'supported_tasks': ['node_classification', 'node_regression', 'graph_classification', 'community_detection'],
        
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
        'supported_tasks': ['node_classification', 'node_regression', 'community_detection'],
        
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
    },
    
    'diffpool': {
        'name': 'DiffPool',
        'description': 'Differentiable graph pooling with hierarchical clustering',
        'paper': 'Ying et al., NeurIPS 2018',
        'category': 'pooling',
        'architecture': 'hierarchical',
        'inductive': False,
        'supported_tasks': ['graph_classification', 'graph_regression'],
        
        # Model-specific parameters
        'parameters': {
            'hidden_dim': {'default': 64, 'range': [16, 512]},
            'num_layers': {'default': 3, 'range': [1, 10]},
            'dropout': {'default': 0.1, 'range': [0.0, 0.5]},
            'activation': {'default': 'relu', 'options': ['relu', 'tanh', 'sigmoid']},
            'pooling_ratio': {'default': 0.5, 'range': [0.1, 0.9]},
            'link_pred': {'default': True, 'type': 'bool'},
            'bias': {'default': True, 'type': 'bool'}
        },
        
        # Implementation details
        'implementation': {
            'framework': 'pytorch_geometric',
            'class': 'DiffPool',
            'optimization': 'hierarchical',
            'memory_efficient': False
        }
    },
    
    'sortpool': {
        'name': 'SortPool',
        'description': 'SortPooling for graph neural networks',
        'paper': 'Zhang et al., AAAI 2018',
        'category': 'pooling',
        'architecture': 'sorting',
        'inductive': False,
        'supported_tasks': ['graph_classification', 'graph_regression'],
        
        # Model-specific parameters
        'parameters': {
            'hidden_dim': {'default': 64, 'range': [16, 512]},
            'num_layers': {'default': 3, 'range': [1, 10]},
            'dropout': {'default': 0.1, 'range': [0.0, 0.5]},
            'activation': {'default': 'relu', 'options': ['relu', 'tanh', 'sigmoid']},
            'k': {'default': 30, 'range': [10, 100]},
            'bias': {'default': True, 'type': 'bool'}
        },
        
        # Implementation details
        'implementation': {
            'framework': 'pytorch_geometric',
            'class': 'SortPool',
            'optimization': 'sorting',
            'memory_efficient': True
        }
    },
    
    'gcn_ae': {
        'name': 'GCN Autoencoder',
        'description': 'Graph Convolutional Network with Autoencoder for anomaly detection',
        'paper': 'Various papers on GCN autoencoders',
        'category': 'anomaly_detection',
        'architecture': 'autoencoder',
        'inductive': False,
        'supported_tasks': ['anomaly_detection'],
        
        # Model-specific parameters
        'parameters': {
            'hidden_dim': {'default': 64, 'range': [16, 512]},
            'num_layers': {'default': 3, 'range': [1, 10]},
            'dropout': {'default': 0.1, 'range': [0.0, 0.5]},
            'activation': {'default': 'relu', 'options': ['relu', 'tanh', 'sigmoid']},
            'latent_dim': {'default': 32, 'range': [8, 256]},
            'anomaly_threshold': {'default': 0.5, 'range': [0.1, 0.9]},
            'bias': {'default': True, 'type': 'bool'}
        },
        
        # Implementation details
        'implementation': {
            'framework': 'pytorch_geometric',
            'class': 'GCNAutoencoder',
            'optimization': 'reconstruction',
            'memory_efficient': True
        }
    },
    
    'gat_ae': {
        'name': 'GAT Autoencoder',
        'description': 'Graph Attention Network with Autoencoder for anomaly detection',
        'paper': 'Various papers on GAT autoencoders',
        'category': 'anomaly_detection',
        'architecture': 'attention_autoencoder',
        'inductive': False,
        'supported_tasks': ['anomaly_detection'],
        
        # Model-specific parameters
        'parameters': {
            'hidden_dim': {'default': 64, 'range': [16, 512]},
            'num_layers': {'default': 3, 'range': [1, 10]},
            'dropout': {'default': 0.1, 'range': [0.0, 0.5]},
            'activation': {'default': 'relu', 'options': ['relu', 'tanh', 'sigmoid']},
            'attention_heads': {'default': 8, 'range': [1, 16]},
            'latent_dim': {'default': 32, 'range': [8, 256]},
            'anomaly_threshold': {'default': 0.5, 'range': [0.1, 0.9]},
            'bias': {'default': True, 'type': 'bool'}
        },
        
        # Implementation details
        'implementation': {
            'framework': 'pytorch_geometric',
            'class': 'GATAutoencoder',
            'optimization': 'attention_reconstruction',
            'memory_efficient': False
        }
    },
    
    'dominant': {
        'name': 'DOMINANT',
        'description': 'Deep Anomaly Detection on Attributed Networks',
        'paper': 'Ding et al., WSDM 2019',
        'category': 'anomaly_detection',
        'architecture': 'reconstruction',
        'inductive': False,
        'supported_tasks': ['anomaly_detection'],
        
        # Model-specific parameters
        'parameters': {
            'hidden_dim': {'default': 64, 'range': [16, 512]},
            'num_layers': {'default': 3, 'range': [1, 10]},
            'dropout': {'default': 0.1, 'range': [0.0, 0.5]},
            'activation': {'default': 'relu', 'options': ['relu', 'tanh', 'sigmoid']},
            'alpha': {'default': 0.5, 'range': [0.1, 0.9]},
            'beta': {'default': 0.5, 'range': [0.1, 0.9]},
            'anomaly_threshold': {'default': 0.5, 'range': [0.1, 0.9]},
            'bias': {'default': True, 'type': 'bool'}
        },
        
        # Implementation details
        'implementation': {
            'framework': 'custom',
            'class': 'DOMINANT',
            'optimization': 'reconstruction',
            'memory_efficient': True
        }
    },
    
    'anomalydae': {
        'name': 'AnomalyDAE',
        'description': 'Anomaly Detection on Attributed Networks via Variational Graph Autoencoder',
        'paper': 'Fan et al., KDD 2020',
        'category': 'anomaly_detection',
        'architecture': 'variational_autoencoder',
        'inductive': False,
        'supported_tasks': ['anomaly_detection'],
        
        # Model-specific parameters
        'parameters': {
            'hidden_dim': {'default': 64, 'range': [16, 512]},
            'num_layers': {'default': 3, 'range': [1, 10]},
            'dropout': {'default': 0.1, 'range': [0.0, 0.5]},
            'activation': {'default': 'relu', 'options': ['relu', 'tanh', 'sigmoid']},
            'latent_dim': {'default': 32, 'range': [8, 256]},
            'beta': {'default': 1.0, 'range': [0.0, 10.0]},
            'anomaly_threshold': {'default': 0.5, 'range': [0.1, 0.9]},
            'bias': {'default': True, 'type': 'bool'}
        },
        
        # Implementation details
        'implementation': {
            'framework': 'custom',
            'class': 'AnomalyDAE',
            'optimization': 'variational',
            'memory_efficient': True
        }
    },
    
    'ganomaly': {
        'name': 'GAnomaly',
        'description': 'Generative Adversarial Network for Anomaly Detection',
        'paper': 'Various papers on GAN-based anomaly detection',
        'category': 'anomaly_detection',
        'architecture': 'generative_adversarial',
        'inductive': False,
        'supported_tasks': ['anomaly_detection'],
        
        # Model-specific parameters
        'parameters': {
            'hidden_dim': {'default': 64, 'range': [16, 512]},
            'num_layers': {'default': 3, 'range': [1, 10]},
            'dropout': {'default': 0.1, 'range': [0.0, 0.5]},
            'activation': {'default': 'relu', 'options': ['relu', 'tanh', 'sigmoid']},
            'latent_dim': {'default': 32, 'range': [8, 256]},
            'noise_dim': {'default': 16, 'range': [4, 128]},
            'anomaly_threshold': {'default': 0.5, 'range': [0.1, 0.9]},
            'bias': {'default': True, 'type': 'bool'}
        },
        
        # Implementation details
        'implementation': {
            'framework': 'custom',
            'class': 'GAnomaly',
            'optimization': 'adversarial',
            'memory_efficient': False
        }
    },
    
    'graphsage_ae': {
        'name': 'GraphSAGE Autoencoder',
        'description': 'GraphSAGE with Autoencoder for anomaly detection',
        'paper': 'Various papers on GraphSAGE autoencoders',
        'category': 'anomaly_detection',
        'architecture': 'inductive_autoencoder',
        'inductive': True,
        'supported_tasks': ['anomaly_detection'],
        
        # Model-specific parameters
        'parameters': {
            'hidden_dim': {'default': 64, 'range': [16, 512]},
            'num_layers': {'default': 3, 'range': [1, 10]},
            'dropout': {'default': 0.1, 'range': [0.0, 0.5]},
            'activation': {'default': 'relu', 'options': ['relu', 'tanh', 'sigmoid']},
            'neighbor_sampling': {'default': 25, 'range': [5, 100]},
            'latent_dim': {'default': 32, 'range': [8, 256]},
            'anomaly_threshold': {'default': 0.5, 'range': [0.1, 0.9]},
            'bias': {'default': True, 'type': 'bool'}
        },
        
        # Implementation details
        'implementation': {
            'framework': 'pytorch_geometric',
            'class': 'GraphSAGEAutoencoder',
            'optimization': 'sampling_reconstruction',
            'memory_efficient': True
        }
    },
    
    'dysat': {
        'name': 'DySAT',
        'description': 'Dynamic Self-Attention Network for Dynamic Graph Embedding',
        'paper': 'Sankar et al., WSDM 2020',
        'category': 'dynamic_graph',
        'architecture': 'temporal_attention',
        'inductive': False,
        'supported_tasks': ['dynamic_graph_learning'],
        
        # Model-specific parameters
        'parameters': {
            'hidden_dim': {'default': 64, 'range': [16, 512]},
            'num_layers': {'default': 3, 'range': [1, 10]},
            'dropout': {'default': 0.1, 'range': [0.0, 0.5]},
            'activation': {'default': 'relu', 'options': ['relu', 'tanh', 'sigmoid']},
            'attention_heads': {'default': 8, 'range': [1, 16]},
            'num_time_steps': {'default': 10, 'range': [5, 50]},
            'bias': {'default': True, 'type': 'bool'}
        },
        
        # Implementation details
        'implementation': {
            'framework': 'custom',
            'class': 'DySAT',
            'optimization': 'temporal_attention',
            'memory_efficient': False
        }
    },
    
    'tgat': {
        'name': 'Temporal Graph Attention Network',
        'description': 'Temporal Graph Attention Network for Dynamic Graph Learning',
        'paper': 'Xu et al., ICLR 2020',
        'category': 'dynamic_graph',
        'architecture': 'temporal_attention',
        'inductive': False,
        'supported_tasks': ['dynamic_graph_learning'],
        
        # Model-specific parameters
        'parameters': {
            'hidden_dim': {'default': 64, 'range': [16, 512]},
            'num_layers': {'default': 3, 'range': [1, 10]},
            'dropout': {'default': 0.1, 'range': [0.0, 0.5]},
            'activation': {'default': 'relu', 'options': ['relu', 'tanh', 'sigmoid']},
            'attention_heads': {'default': 8, 'range': [1, 16]},
            'num_time_steps': {'default': 10, 'range': [5, 50]},
            'bias': {'default': True, 'type': 'bool'}
        },
        
        # Implementation details
        'implementation': {
            'framework': 'custom',
            'class': 'TGAT',
            'optimization': 'temporal_attention',
            'memory_efficient': False
        }
    },
    
    'evolvegcn': {
        'name': 'EvolveGCN',
        'description': 'Evolving Graph Convolutional Networks for Dynamic Graph Learning',
        'paper': 'Pareja et al., NeurIPS 2020',
        'category': 'dynamic_graph',
        'architecture': 'evolving_convolution',
        'inductive': False,
        'supported_tasks': ['dynamic_graph_learning'],
        
        # Model-specific parameters
        'parameters': {
            'hidden_dim': {'default': 64, 'range': [16, 512]},
            'num_layers': {'default': 3, 'range': [1, 10]},
            'dropout': {'default': 0.1, 'range': [0.0, 0.5]},
            'activation': {'default': 'relu', 'options': ['relu', 'tanh', 'sigmoid']},
            'evolve_type': {'default': 'GRU', 'options': ['GRU', 'HGRU']},
            'num_time_steps': {'default': 10, 'range': [5, 50]},
            'bias': {'default': True, 'type': 'bool'}
        },
        
        # Implementation details
        'implementation': {
            'framework': 'custom',
            'class': 'EvolveGCN',
            'optimization': 'evolving',
            'memory_efficient': True
        }
    },
    
    'dyrep': {
        'name': 'DyRep',
        'description': 'Learning Representations over Dynamic Graphs',
        'paper': 'Trivedi et al., ICLR 2019',
        'category': 'dynamic_graph',
        'architecture': 'temporal_point_process',
        'inductive': False,
        'supported_tasks': ['dynamic_graph_learning'],
        
        # Model-specific parameters
        'parameters': {
            'hidden_dim': {'default': 64, 'range': [16, 512]},
            'num_layers': {'default': 3, 'range': [1, 10]},
            'dropout': {'default': 0.1, 'range': [0.0, 0.5]},
            'activation': {'default': 'relu', 'options': ['relu', 'tanh', 'sigmoid']},
            'num_time_steps': {'default': 10, 'range': [5, 50]},
            'bias': {'default': True, 'type': 'bool'}
        },
        
        # Implementation details
        'implementation': {
            'framework': 'custom',
            'class': 'DyRep',
            'optimization': 'temporal_point_process',
            'memory_efficient': True
        }
    },
    
    'jodie': {
        'name': 'JODIE',
        'description': 'Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks',
        'paper': 'Kumar et al., KDD 2019',
        'category': 'dynamic_graph',
        'architecture': 'temporal_interaction',
        'inductive': False,
        'supported_tasks': ['dynamic_graph_learning'],
        
        # Model-specific parameters
        'parameters': {
            'hidden_dim': {'default': 64, 'range': [16, 512]},
            'num_layers': {'default': 3, 'range': [1, 10]},
            'dropout': {'default': 0.1, 'range': [0.0, 0.5]},
            'activation': {'default': 'relu', 'options': ['relu', 'tanh', 'sigmoid']},
            'num_time_steps': {'default': 10, 'range': [5, 50]},
            'bias': {'default': True, 'type': 'bool'}
        },
        
        # Implementation details
        'implementation': {
            'framework': 'custom',
            'class': 'JODIE',
            'optimization': 'temporal_interaction',
            'memory_efficient': True
        }
    },
    
    'tgn': {
        'name': 'Temporal Graph Network',
        'description': 'Temporal Graph Networks for Deep Learning on Dynamic Graphs',
        'paper': 'Rossi et al., NeurIPS 2020',
        'category': 'dynamic_graph',
        'architecture': 'temporal_network',
        'inductive': False,
        'supported_tasks': ['dynamic_graph_learning'],
        
        # Model-specific parameters
        'parameters': {
            'hidden_dim': {'default': 64, 'range': [16, 512]},
            'num_layers': {'default': 3, 'range': [1, 10]},
            'dropout': {'default': 0.1, 'range': [0.0, 0.5]},
            'activation': {'default': 'relu', 'options': ['relu', 'tanh', 'sigmoid']},
            'num_time_steps': {'default': 10, 'range': [5, 50]},
            'bias': {'default': True, 'type': 'bool'}
        },
        
        # Implementation details
        'implementation': {
            'framework': 'custom',
            'class': 'TGN',
            'optimization': 'temporal_network',
            'memory_efficient': True
        }
    },
    
    'graphrnn': {
        'name': 'GraphRNN',
        'description': 'Generating Realistic Graphs with Recurrent Neural Networks',
        'paper': 'You et al., ICML 2018',
        'category': 'graph_generation',
        'architecture': 'recurrent',
        'inductive': False,
        'supported_tasks': ['graph_generation'],
        
        # Model-specific parameters
        'parameters': {
            'hidden_dim': {'default': 64, 'range': [16, 512]},
            'num_layers': {'default': 3, 'range': [1, 10]},
            'dropout': {'default': 0.1, 'range': [0.0, 0.5]},
            'activation': {'default': 'relu', 'options': ['relu', 'tanh', 'sigmoid']},
            'max_nodes': {'default': 20, 'range': [5, 100]},
            'bias': {'default': True, 'type': 'bool'}
        },
        
        # Implementation details
        'implementation': {
            'framework': 'custom',
            'class': 'GraphRNN',
            'optimization': 'recurrent',
            'memory_efficient': True
        }
    },
    
    'graphvae': {
        'name': 'GraphVAE',
        'description': 'Variational Autoencoder for Graph Generation',
        'paper': 'Simonovsky & Komodakis, NeurIPS 2018',
        'category': 'graph_generation',
        'architecture': 'variational_autoencoder',
        'inductive': False,
        'supported_tasks': ['graph_generation'],
        
        # Model-specific parameters
        'parameters': {
            'hidden_dim': {'default': 64, 'range': [16, 512]},
            'num_layers': {'default': 3, 'range': [1, 10]},
            'dropout': {'default': 0.1, 'range': [0.0, 0.5]},
            'activation': {'default': 'relu', 'options': ['relu', 'tanh', 'sigmoid']},
            'latent_dim': {'default': 32, 'range': [8, 256]},
            'max_nodes': {'default': 20, 'range': [5, 100]},
            'bias': {'default': True, 'type': 'bool'}
        },
        
        # Implementation details
        'implementation': {
            'framework': 'custom',
            'class': 'GraphVAE',
            'optimization': 'variational',
            'memory_efficient': True
        }
    },
    
    'graphgan': {
        'name': 'GraphGAN',
        'description': 'Generative Adversarial Network for Graph Generation',
        'paper': 'Wang et al., AAAI 2018',
        'category': 'graph_generation',
        'architecture': 'generative_adversarial',
        'inductive': False,
        'supported_tasks': ['graph_generation'],
        
        # Model-specific parameters
        'parameters': {
            'hidden_dim': {'default': 64, 'range': [16, 512]},
            'num_layers': {'default': 3, 'range': [1, 10]},
            'dropout': {'default': 0.1, 'range': [0.0, 0.5]},
            'activation': {'default': 'relu', 'options': ['relu', 'tanh', 'sigmoid']},
            'noise_dim': {'default': 16, 'range': [4, 128]},
            'max_nodes': {'default': 20, 'range': [5, 100]},
            'bias': {'default': True, 'type': 'bool'}
        },
        
        # Implementation details
        'implementation': {
            'framework': 'custom',
            'class': 'GraphGAN',
            'optimization': 'adversarial',
            'memory_efficient': False
        }
    },
    
    'molgan': {
        'name': 'MolGAN',
        'description': 'An implicit generative model for small molecular graphs',
        'paper': 'De Cao & Kipf, ICML 2018',
        'category': 'graph_generation',
        'architecture': 'generative_adversarial',
        'inductive': False,
        'supported_tasks': ['graph_generation'],
        
        # Model-specific parameters
        'parameters': {
            'hidden_dim': {'default': 64, 'range': [16, 512]},
            'num_layers': {'default': 3, 'range': [1, 10]},
            'dropout': {'default': 0.1, 'range': [0.0, 0.5]},
            'activation': {'default': 'relu', 'options': ['relu', 'tanh', 'sigmoid']},
            'noise_dim': {'default': 16, 'range': [4, 128]},
            'max_nodes': {'default': 20, 'range': [5, 100]},
            'bias': {'default': True, 'type': 'bool'}
        },
        
        # Implementation details
        'implementation': {
            'framework': 'custom',
            'class': 'MolGAN',
            'optimization': 'adversarial',
            'memory_efficient': False
        }
    },
    
    'graphaf': {
        'name': 'GraphAF',
        'description': 'Flow-based Autoregressive Model for Graph Generation',
        'paper': 'Shi et al., ICML 2020',
        'category': 'graph_generation',
        'architecture': 'flow_autoregressive',
        'inductive': False,
        'supported_tasks': ['graph_generation'],
        
        # Model-specific parameters
        'parameters': {
            'hidden_dim': {'default': 64, 'range': [16, 512]},
            'num_layers': {'default': 3, 'range': [1, 10]},
            'dropout': {'default': 0.1, 'range': [0.0, 0.5]},
            'activation': {'default': 'relu', 'options': ['relu', 'tanh', 'sigmoid']},
            'max_nodes': {'default': 20, 'range': [5, 100]},
            'bias': {'default': True, 'type': 'bool'}
        },
        
        # Implementation details
        'implementation': {
            'framework': 'custom',
            'class': 'GraphAF',
            'optimization': 'flow',
            'memory_efficient': True
        }
    },
    
    'graphscore': {
        'name': 'GraphScore',
        'description': 'Score-based Model for Graph Generation',
        'paper': 'Various papers on score-based graph generation',
        'category': 'graph_generation',
        'architecture': 'score_based',
        'inductive': False,
        'supported_tasks': ['graph_generation'],
        
        # Model-specific parameters
        'parameters': {
            'hidden_dim': {'default': 64, 'range': [16, 512]},
            'num_layers': {'default': 3, 'range': [1, 10]},
            'dropout': {'default': 0.1, 'range': [0.0, 0.5]},
            'activation': {'default': 'relu', 'options': ['relu', 'tanh', 'sigmoid']},
            'max_nodes': {'default': 20, 'range': [5, 100]},
            'bias': {'default': True, 'type': 'bool'}
        },
        
        # Implementation details
        'implementation': {
            'framework': 'custom',
            'class': 'GraphScore',
            'optimization': 'score_based',
            'memory_efficient': True
        }
    },
    
    'node2vec': {
        'name': 'Node2Vec',
        'description': 'Scalable Feature Learning for Networks',
        'paper': 'Grover & Leskovec, KDD 2016',
        'category': 'embedding',
        'architecture': 'random_walk',
        'inductive': False,
        'supported_tasks': ['graph_embedding_visualization'],
        
        # Model-specific parameters
        'parameters': {
            'embedding_dim': {'default': 128, 'range': [16, 512]},
            'walk_length': {'default': 80, 'range': [10, 200]},
            'num_walks': {'default': 10, 'range': [1, 50]},
            'p': {'default': 1.0, 'range': [0.1, 10.0]},
            'q': {'default': 1.0, 'range': [0.1, 10.0]},
            'window_size': {'default': 10, 'range': [1, 20]},
            'bias': {'default': True, 'type': 'bool'}
        },
        
        # Implementation details
        'implementation': {
            'framework': 'networkx',
            'class': 'Node2Vec',
            'optimization': 'random_walk',
            'memory_efficient': True
        }
    },
    
    'deepwalk': {
        'name': 'DeepWalk',
        'description': 'Online Learning of Social Representations',
        'paper': 'Perozzi et al., KDD 2014',
        'category': 'embedding',
        'architecture': 'random_walk',
        'inductive': False,
        'supported_tasks': ['graph_embedding_visualization'],
        
        # Model-specific parameters
        'parameters': {
            'embedding_dim': {'default': 128, 'range': [16, 512]},
            'walk_length': {'default': 80, 'range': [10, 200]},
            'num_walks': {'default': 10, 'range': [1, 50]},
            'window_size': {'default': 10, 'range': [1, 20]},
            'bias': {'default': True, 'type': 'bool'}
        },
        
        # Implementation details
        'implementation': {
            'framework': 'networkx',
            'class': 'DeepWalk',
            'optimization': 'random_walk',
            'memory_efficient': True
        }
    },
    
    'line': {
        'name': 'LINE',
        'description': 'Large-scale Information Network Embedding',
        'paper': 'Tang et al., WWW 2015',
        'category': 'embedding',
        'architecture': 'edge_sampling',
        'inductive': False,
        'supported_tasks': ['graph_embedding_visualization'],
        
        # Model-specific parameters
        'parameters': {
            'embedding_dim': {'default': 128, 'range': [16, 512]},
            'order': {'default': 2, 'range': [1, 3]},
            'negative': {'default': 5, 'range': [1, 20]},
            'bias': {'default': True, 'type': 'bool'}
        },
        
        # Implementation details
        'implementation': {
            'framework': 'networkx',
            'class': 'LINE',
            'optimization': 'edge_sampling',
            'memory_efficient': True
        }
    },
    
    'sdne': {
        'name': 'SDNE',
        'description': 'Structural Deep Network Embedding',
        'paper': 'Wang et al., KDD 2016',
        'category': 'embedding',
        'architecture': 'autoencoder',
        'inductive': False,
        'supported_tasks': ['graph_embedding_visualization'],
        
        # Model-specific parameters
        'parameters': {
            'embedding_dim': {'default': 128, 'range': [16, 512]},
            'hidden_dim': {'default': 256, 'range': [64, 1024]},
            'dropout': {'default': 0.1, 'range': [0.0, 0.5]},
            'activation': {'default': 'relu', 'options': ['relu', 'tanh', 'sigmoid']},
            'alpha': {'default': 1.0, 'range': [0.1, 10.0]},
            'beta': {'default': 1.0, 'range': [0.1, 10.0]},
            'bias': {'default': True, 'type': 'bool'}
        },
        
        # Implementation details
        'implementation': {
            'framework': 'custom',
            'class': 'SDNE',
            'optimization': 'autoencoder',
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
    },
    'pooling': {
        'name': 'Pooling Models',
        'description': 'Models for graph-level pooling',
        'icon': 'fas fa-layer-group',
        'color': 'dark',
        'models': ['diffpool', 'sortpool']
    },
    'anomaly_detection': {
        'name': 'Anomaly Detection Models',
        'description': 'Models for detecting anomalies in graphs',
        'icon': 'fas fa-exclamation-triangle',
        'color': 'danger',
        'models': ['gcn_ae', 'gat_ae', 'dominant', 'anomalydae', 'ganomaly', 'graphsage_ae']
    },
    'dynamic_graph': {
        'name': 'Dynamic Graph Models',
        'description': 'Models for learning from dynamic graphs',
        'icon': 'fas fa-clock',
        'color': 'info',
        'models': ['dysat', 'tgat', 'evolvegcn', 'dyrep', 'jodie', 'tgn']
    },
    'graph_generation': {
        'name': 'Graph Generation Models',
        'description': 'Models for generating new graphs',
        'icon': 'fas fa-project-diagram',
        'color': 'purple',
        'models': ['graphrnn', 'graphvae', 'graphgan', 'molgan', 'graphaf', 'graphscore']
    },
    'embedding': {
        'name': 'Embedding Models',
        'description': 'Models for learning graph embeddings',
        'icon': 'fas fa-eye',
        'color': 'secondary',
        'models': ['node2vec', 'deepwalk', 'line', 'sdne']
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