"""
Graph Embedding Visualization Parameters
This module defines the parameters for graph embedding visualization tasks.
"""

# Parameter categories for organization
PARAMETER_CATEGORIES = {
    'model': {
        'name': 'Model Configuration',
        'description': 'Core model parameters',
        'icon': 'fas fa-brain'
    },
    'embedding': {
        'name': 'Embedding Settings',
        'description': 'Embedding-specific parameters',
        'icon': 'fas fa-eye'
    },
    'training': {
        'name': 'Training Configuration',
        'description': 'Training and optimization parameters',
        'icon': 'fas fa-cogs'
    },
    'visualization': {
        'name': 'Visualization Options',
        'description': 'Visualization and analysis parameters',
        'icon': 'fas fa-chart-line'
    },
    'evaluation': {
        'name': 'Evaluation Metrics',
        'description': 'Evaluation and validation parameters',
        'icon': 'fas fa-chart-bar'
    }
}

# All parameters for graph embedding visualization
GRAPH_EMBEDDING_VISUALIZATION_PARAMETERS = {
    # Model Configuration
    'embedding_dim': {
        'name': 'Embedding Dimension',
        'type': 'int',
        'default': 64,
        'min': 16,
        'max': 512,
        'description': 'Dimension of the learned embeddings',
        'category': 'model'
    },
    'hidden_dim': {
        'name': 'Hidden Dimension',
        'type': 'int',
        'default': 128,
        'min': 32,
        'max': 1024,
        'description': 'Number of hidden units in each layer',
        'category': 'model'
    },
    'num_layers': {
        'name': 'Number of Layers',
        'type': 'int',
        'default': 3,
        'min': 1,
        'max': 10,
        'description': 'Number of graph convolution layers',
        'category': 'model'
    },
    'dropout': {
        'name': 'Dropout Rate',
        'type': 'float',
        'default': 0.1,
        'min': 0.0,
        'max': 0.9,
        'description': 'Dropout rate for regularization',
        'category': 'model'
    },
    'activation': {
        'name': 'Activation Function',
        'type': 'select',
        'default': 'relu',
        'options': ['relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu'],
        'description': 'Activation function for hidden layers',
        'category': 'model'
    },
    
    # Embedding Settings
    'walk_length': {
        'name': 'Walk Length',
        'type': 'int',
        'default': 80,
        'min': 10,
        'max': 200,
        'description': 'Length of random walks for Node2Vec/DeepWalk',
        'category': 'embedding'
    },
    'num_walks': {
        'name': 'Number of Walks',
        'type': 'int',
        'default': 10,
        'min': 1,
        'max': 50,
        'description': 'Number of random walks per node',
        'category': 'embedding'
    },
    'p': {
        'name': 'Return Parameter (p)',
        'type': 'float',
        'default': 1.0,
        'min': 0.1,
        'max': 10.0,
        'description': 'Return parameter for Node2Vec biased walks',
        'category': 'embedding'
    },
    'q': {
        'name': 'In-Out Parameter (q)',
        'type': 'float',
        'default': 1.0,
        'min': 0.1,
        'max': 10.0,
        'description': 'In-out parameter for Node2Vec biased walks',
        'category': 'embedding'
    },
    'window_size': {
        'name': 'Window Size',
        'type': 'int',
        'default': 10,
        'min': 1,
        'max': 20,
        'description': 'Context window size for skip-gram',
        'category': 'embedding'
    },
    'negative_samples': {
        'name': 'Negative Samples',
        'type': 'int',
        'default': 5,
        'min': 1,
        'max': 20,
        'description': 'Number of negative samples for training',
        'category': 'embedding'
    },
    
    # Training Configuration
    'learning_rate': {
        'name': 'Learning Rate',
        'type': 'float',
        'default': 0.001,
        'min': 0.0001,
        'max': 0.1,
        'description': 'Learning rate for optimization',
        'category': 'training'
    },
    'epochs': {
        'name': 'Epochs',
        'type': 'int',
        'default': 100,
        'min': 10,
        'max': 1000,
        'description': 'Number of training epochs',
        'category': 'training'
    },
    'batch_size': {
        'name': 'Batch Size',
        'type': 'int',
        'default': 256,
        'min': 32,
        'max': 2048,
        'description': 'Batch size for training',
        'category': 'training'
    },
    'weight_decay': {
        'name': 'Weight Decay',
        'type': 'float',
        'default': 0.0001,
        'min': 0.0,
        'max': 0.01,
        'description': 'Weight decay for regularization',
        'category': 'training'
    },
    'scheduler': {
        'name': 'Learning Rate Scheduler',
        'type': 'select',
        'default': 'step',
        'options': ['none', 'step', 'cosine', 'exponential', 'plateau'],
        'description': 'Learning rate scheduling strategy',
        'category': 'training'
    },
    'patience': {
        'name': 'Early Stopping Patience',
        'type': 'int',
        'default': 20,
        'min': 5,
        'max': 100,
        'description': 'Patience for early stopping',
        'category': 'training'
    },
    
    # Visualization Options
    'visualization_method': {
        'name': 'Visualization Method',
        'type': 'select',
        'default': 'tsne',
        'options': ['tsne', 'umap', 'pca', 'mds', 'isomap'],
        'description': 'Dimensionality reduction method for visualization',
        'category': 'visualization'
    },
    'perplexity': {
        'name': 't-SNE Perplexity',
        'type': 'int',
        'default': 30,
        'min': 5,
        'max': 100,
        'description': 'Perplexity parameter for t-SNE',
        'category': 'visualization'
    },
    'n_neighbors': {
        'name': 'UMAP Neighbors',
        'type': 'int',
        'default': 15,
        'min': 5,
        'max': 50,
        'description': 'Number of neighbors for UMAP',
        'category': 'visualization'
    },
    'min_dist': {
        'name': 'UMAP Min Distance',
        'type': 'float',
        'default': 0.1,
        'min': 0.01,
        'max': 1.0,
        'description': 'Minimum distance for UMAP',
        'category': 'visualization'
    },
    'clustering_method': {
        'name': 'Clustering Method',
        'type': 'select',
        'default': 'kmeans',
        'options': ['kmeans', 'dbscan', 'spectral', 'hierarchical', 'gmm'],
        'description': 'Clustering method for embedding analysis',
        'category': 'visualization'
    },
    'n_clusters': {
        'name': 'Number of Clusters',
        'type': 'int',
        'default': 5,
        'min': 2,
        'max': 20,
        'description': 'Number of clusters for clustering analysis',
        'category': 'visualization'
    },
    
    # Evaluation Metrics
    'evaluation_metrics': {
        'name': 'Evaluation Metrics',
        'type': 'multiselect',
        'default': ['reconstruction_loss', 'link_prediction_auc', 'clustering_score'],
        'options': [
            'reconstruction_loss', 'link_prediction_auc', 'node_classification_acc',
            'clustering_score', 'silhouette_score', 'calinski_harabasz_score',
            'davies_bouldin_score', 'embedding_quality', 'visualization_quality'
        ],
        'description': 'Metrics to evaluate embedding quality',
        'category': 'evaluation'
    },
    'validation_split': {
        'name': 'Validation Split',
        'type': 'float',
        'default': 0.2,
        'min': 0.1,
        'max': 0.5,
        'description': 'Fraction of data for validation',
        'category': 'evaluation'
    },
    'test_split': {
        'name': 'Test Split',
        'type': 'float',
        'default': 0.1,
        'min': 0.05,
        'max': 0.3,
        'description': 'Fraction of data for testing',
        'category': 'evaluation'
    }
}

# Default parameters for quick start
DEFAULT_GRAPH_EMBEDDING_VISUALIZATION_PARAMS = {
    'embedding_dim': 64,
    'hidden_dim': 128,
    'num_layers': 3,
    'dropout': 0.1,
    'activation': 'relu',
    'walk_length': 80,
    'num_walks': 10,
    'p': 1.0,
    'q': 1.0,
    'window_size': 10,
    'negative_samples': 5,
    'learning_rate': 0.001,
    'epochs': 100,
    'batch_size': 256,
    'weight_decay': 0.0001,
    'scheduler': 'step',
    'patience': 20,
    'visualization_method': 'tsne',
    'perplexity': 30,
    'n_neighbors': 15,
    'min_dist': 0.1,
    'clustering_method': 'kmeans',
    'n_clusters': 5,
    'evaluation_metrics': ['reconstruction_loss', 'link_prediction_auc', 'clustering_score'],
    'validation_split': 0.2,
    'test_split': 0.1
} 