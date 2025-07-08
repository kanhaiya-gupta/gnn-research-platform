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
        'formula': 'H^{(l+1)} = σ(D̃^(-1/2)ÃD̃^(-1/2)H^{(l)}W^{(l)})',
        'applications': ['Citation Networks', 'Social Networks', 'Knowledge Graphs', 'Molecular Graphs', 'Recommendation Systems'],
        'category': 'convolutional',
        'architecture': 'spectral',
        'inductive': False,
        'supported_tasks': ['node_classification', 'node_regression', 'link_prediction', 'edge_classification', 'graph_classification', 'graph_regression', 'community_detection', 'graph_embedding_visualization'],
        
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
        'formula': 'α_{ij} = softmax(LeakyReLU(a^T[Wh_i || Wh_j]))',
        'applications': ['Protein Interaction Networks', 'Social Networks', 'Knowledge Graphs', 'Molecular Property Prediction', 'Recommendation Systems'],
        'category': 'attention',
        'architecture': 'attention',
        'inductive': False,
        'supported_tasks': ['node_classification', 'node_regression', 'link_prediction', 'edge_classification', 'graph_classification', 'graph_regression', 'community_detection', 'graph_embedding_visualization'],
        
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
        'formula': 'h_v^{(k)} = σ(W^{(k)} · CONCAT(h_v^{(k-1)}, AGG({h_u^{(k-1)}, ∀u ∈ N(v)})))',
        'applications': ['Large-Scale Social Networks', 'E-commerce Networks', 'Citation Networks', 'Biological Networks', 'Fraud Detection'],
        'category': 'inductive',
        'architecture': 'spatial',
        'inductive': True,
        'supported_tasks': ['node_classification', 'node_regression', 'link_prediction', 'edge_classification', 'graph_classification', 'graph_regression', 'community_detection', 'graph_embedding_visualization'],
        
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
        'formula': 'h_v^{(k)} = MLP^{(k)}((1 + ε^{(k)}) · h_v^{(k-1)} + Σ_{u∈N(v)} h_u^{(k-1)})',
        'applications': ['Molecular Property Prediction', 'Drug Discovery', 'Chemical Networks', 'Graph Classification', 'Bioinformatics'],
        'category': 'convolutional',
        'architecture': 'spatial',
        'inductive': True,
        'supported_tasks': ['node_classification', 'node_regression', 'graph_classification', 'graph_regression', 'community_detection', 'graph_embedding_visualization'],
        
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
        'formula': 'g_θ(L) = Σ_{k=0}^{K-1} θ_k T_k(L̃)',
        'applications': ['Spectral Graph Analysis', 'Signal Processing on Graphs', 'Computer Vision', '3D Shape Analysis', 'Spectral Clustering'],
        'category': 'convolutional',
        'architecture': 'spectral',
        'inductive': False,
        'supported_tasks': ['node_classification', 'node_regression', 'link_prediction', 'edge_classification', 'graph_classification', 'community_detection', 'graph_embedding_visualization'],
        
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
        'formula': 'q(Z|X,A) = Π_{i=1}^N q(z_i|X,A), q(z_i|X,A) = N(z_i|μ_i, diag(σ_i²))',
        'applications': ['Graph Generation', 'Link Prediction', 'Graph Completion', 'Network Reconstruction', 'Anomaly Detection'],
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
    
    'node2vec': {
        'name': 'Node2Vec',
        'description': 'Scalable feature learning for networks using biased random walks',
        'paper': 'Grover & Leskovec, KDD 2016',
        'formula': 'P(c_i = x | c_{i-1} = v) = π_{vx}/Z, π_{vx} = α_{pq}(t,x) · w_{vx}',
        'applications': ['Social Network Analysis', 'Recommendation Systems', 'Community Detection', 'Link Prediction', 'Node Classification'],
        'category': 'embedding',
        'architecture': 'random_walk',
        'inductive': False,
        'supported_tasks': ['graph_embedding_visualization'],
        
        # Model-specific parameters
        'parameters': {
            'embedding_dim': {'default': 64, 'range': [16, 512]},
            'walk_length': {'default': 80, 'range': [10, 200]},
            'num_walks': {'default': 10, 'range': [1, 50]},
            'p': {'default': 1.0, 'range': [0.1, 10.0]},
            'q': {'default': 1.0, 'range': [0.1, 10.0]},
            'window_size': {'default': 10, 'range': [1, 20]},
            'negative_samples': {'default': 5, 'range': [1, 20]},
            'learning_rate': {'default': 0.01, 'range': [0.001, 0.1]},
            'epochs': {'default': 100, 'range': [10, 1000]}
        },
        
        # Implementation details
        'implementation': {
            'framework': 'networkx',
            'class': 'Node2Vec',
            'optimization': 'biased_walks',
            'memory_efficient': True
        }
    },
    
    'deepwalk': {
        'name': 'DeepWalk',
        'description': 'Social representation learning using truncated random walks',
        'paper': 'Perozzi et al., KDD 2014',
        'formula': 'P(c_i = x | c_{i-1} = v) = softmax(W_c · h_v)',
        'applications': ['Social Network Analysis', 'Community Detection', 'Link Prediction', 'Node Classification', 'Network Visualization'],
        'category': 'embedding',
        'architecture': 'random_walk',
        'inductive': False,
        'supported_tasks': ['graph_embedding_visualization'],
        
        # Model-specific parameters
        'parameters': {
            'embedding_dim': {'default': 64, 'range': [16, 512]},
            'walk_length': {'default': 80, 'range': [10, 200]},
            'num_walks': {'default': 10, 'range': [1, 50]},
            'window_size': {'default': 10, 'range': [1, 20]},
            'negative_samples': {'default': 5, 'range': [1, 20]},
            'learning_rate': {'default': 0.01, 'range': [0.001, 0.1]},
            'epochs': {'default': 100, 'range': [10, 1000]}
        },
        
        # Implementation details
        'implementation': {
            'framework': 'networkx',
            'class': 'DeepWalk',
            'optimization': 'random_walks',
            'memory_efficient': True
        }
    },
    
    'line': {
        'name': 'LINE',
        'description': 'Large-scale information network embedding preserving first and second-order proximities',
        'paper': 'Tang et al., WWW 2015',
        'formula': 'p_1(v_i, v_j) = 1/(1 + exp(-u_i^T u_j)), p_2(v_j|v_i) = exp(u_j^T u_i)/Σ_k exp(u_k^T u_i)',
        'applications': ['Large-Scale Networks', 'Social Networks', 'Citation Networks', 'Co-occurrence Networks', 'Knowledge Graphs'],
        'category': 'embedding',
        'architecture': 'proximity',
        'inductive': False,
        'supported_tasks': ['graph_embedding_visualization'],
        
        # Model-specific parameters
        'parameters': {
            'embedding_dim': {'default': 64, 'range': [16, 512]},
            'order': {'default': 2, 'range': [1, 2]},
            'negative_samples': {'default': 5, 'range': [1, 20]},
            'learning_rate': {'default': 0.01, 'range': [0.001, 0.1]},
            'epochs': {'default': 100, 'range': [10, 1000]},
            'batch_size': {'default': 256, 'range': [32, 2048]}
        },
        
        # Implementation details
        'implementation': {
            'framework': 'networkx',
            'class': 'LINE',
            'optimization': 'proximity',
            'memory_efficient': True
        }
    },
    
    'sdne': {
        'name': 'SDNE',
        'description': 'Structural Deep Network Embedding for large-scale networks',
        'paper': 'Wang et al., KDD 2016',
        'formula': 'L = L_2nd + αL_1st + νL_reg, L_2nd = ||(X̂ - X) ⊙ B||_F^2',
        'applications': ['Large-Scale Networks', 'Social Networks', 'Citation Networks', 'Protein Networks', 'Recommendation Systems'],
        'category': 'embedding',
        'architecture': 'autoencoder',
        'inductive': False,
        'supported_tasks': ['graph_embedding_visualization'],
        
        # Model-specific parameters
        'parameters': {
            'embedding_dim': {'default': 64, 'range': [16, 512]},
            'hidden_dim': {'default': 128, 'range': [32, 1024]},
            'alpha': {'default': 1.0, 'range': [0.1, 10.0]},
            'beta': {'default': 1.0, 'range': [0.1, 10.0]},
            'nu': {'default': 1e-5, 'range': [1e-6, 1e-3]},
            'learning_rate': {'default': 0.001, 'range': [0.0001, 0.1]},
            'epochs': {'default': 100, 'range': [10, 1000]},
            'batch_size': {'default': 256, 'range': [32, 2048]}
        },
        
        # Implementation details
        'implementation': {
            'framework': 'pytorch',
            'class': 'SDNE',
            'optimization': 'autoencoder',
            'memory_efficient': False
        }
    },
    
    'seal': {
        'name': 'SEAL',
        'description': 'Subgraphs, Embeddings and Attributes for Link prediction',
        'paper': 'Zhang & Chen, NeurIPS 2018',
        'formula': 'f(A, X) = MLP([h_i^{(L)} || h_j^{(L)} || h_{i,j}^{(L)}])',
        'applications': ['Link Prediction', 'Network Completion', 'Social Network Analysis', 'Knowledge Graph Completion', 'Recommendation Systems'],
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
        'formula': 'H^{(k)} = S^k X W^{(k)}',
        'applications': ['Large-Scale Networks', 'Citation Networks', 'Social Networks', 'Fast Node Classification', 'Pre-computed Features'],
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
        'formula': 'H^{(k)} = (1-α) · ÃH^{(k-1)} + α · H^{(0)}',
        'applications': ['Semi-supervised Learning', 'Node Classification', 'Personalized PageRank', 'Graph Neural Networks', 'Large-scale Networks'],
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
        'formula': 'S^{(l)} = softmax(GNN_{pool}^{(l)}(A^{(l)}, X^{(l)}))',
        'applications': ['Graph Classification', 'Molecular Property Prediction', 'Hierarchical Graph Representation', 'Graph Pooling', 'Bioinformatics'],
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
        'formula': 'y = CONV1D(sort(h_1, h_2, ..., h_n)[:k])',
        'applications': ['Graph Classification', 'Graph Regression', 'Molecular Property Prediction', 'Graph Pooling', 'Fixed-size Representations'],
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
        'formula': 'L = ||X - X̂||² + λ||Z||²',
        'applications': ['Anomaly Detection', 'Fraud Detection', 'Network Security', 'Outlier Detection', 'Graph Reconstruction'],
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
        'formula': 'L = ||X - X̂||² + λ||Z||² + βKL(q(Z|X)||p(Z))',
        'applications': ['Anomaly Detection', 'Attention-based Reconstruction', 'Network Security', 'Fraud Detection', 'Graph Attention'],
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
        'formula': 'L = αL_{attr} + βL_{struct} + γL_{reg}',
        'applications': ['Anomaly Detection', 'Fraud Detection', 'Network Security', 'Attributed Networks', 'Deep Learning'],
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
        'formula': 'L = L_{recon} + βKL(q(Z|X,A)||p(Z))',
        'applications': ['Anomaly Detection', 'Variational Autoencoders', 'Attributed Networks', 'Graph Generation', 'Network Security'],
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
        'formula': 'L = L_{adv} + λL_{recon} + γL_{feature}',
        'applications': ['Anomaly Detection', 'Generative Adversarial Networks', 'Fraud Detection', 'Network Security', 'Graph Generation'],
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
        'formula': 'L = ||X - X̂||² + λ||Z||² + μL_{sage}',
        'applications': ['Anomaly Detection', 'Inductive Learning', 'Large-scale Networks', 'GraphSAGE', 'Autoencoder'],
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
        'formula': 'h_v^{(t)} = MultiHead(Q_v^{(t)}, K_v^{(t)}, V_v^{(t)})',
        'applications': ['Dynamic Graph Learning', 'Temporal Networks', 'Social Network Evolution', 'Time-series Graphs', 'Temporal Embeddings'],
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
        'formula': 'α_{ij}^{(t)} = softmax(LeakyReLU(a^T[Wh_i^{(t)} || Wh_j^{(t)} || Φ(t-t_{ij})]))',
        'applications': ['Dynamic Graph Learning', 'Temporal Attention', 'Time-evolving Networks', 'Temporal Link Prediction', 'Dynamic Embeddings'],
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
        'formula': 'W^{(t)} = GRU(W^{(t-1)}, ΔW^{(t)})',
        'applications': ['Dynamic Graph Learning', 'Evolving Networks', 'Temporal GCN', 'Network Evolution', 'Dynamic Convolution'],
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
        'applications': ['Molecular Graph Generation', 'Social Network Generation', 'Chemical Compound Design', 'Drug Discovery', 'Network Topology Generation'],
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
        'applications': ['Molecular Graph Generation', 'Chemical Compound Design', 'Drug Discovery', 'Network Reconstruction', 'Graph Completion'],
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
        'applications': ['Social Network Generation', 'Citation Network Generation', 'Protein Interaction Networks', 'Network Topology Generation', 'Graph Data Augmentation'],
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
        'applications': ['Molecular Graph Generation', 'Drug Discovery', 'Chemical Compound Design', 'Molecular Property Optimization', 'De Novo Drug Design'],
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
        'applications': ['Molecular Graph Generation', 'Chemical Compound Design', 'Drug Discovery', 'Molecular Property Optimization', 'Controlled Graph Generation'],
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
        'applications': ['Molecular Graph Generation', 'Chemical Compound Design', 'Drug Discovery', 'Network Topology Generation', 'Controlled Graph Generation'],
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