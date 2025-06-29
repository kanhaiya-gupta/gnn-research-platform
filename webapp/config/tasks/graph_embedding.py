"""
Graph Embedding Task Configuration
This module defines the configuration for graph embedding tasks.
"""

from ..parameters.graph_embedding import (
    GRAPH_EMBEDDING_PARAMETERS,
    DEFAULT_GRAPH_EMBEDDING_PARAMS,
    PARAMETER_CATEGORIES
)

GRAPH_EMBEDDING_TASK = {
    'name': 'Graph Embedding Visualization',
    'description': 'Learn and visualize graph embeddings for nodes, edges, and graphs',
    'category': 'embedding',
    'icon': 'fas fa-eye',
    'color': 'secondary',
    'parameters': GRAPH_EMBEDDING_PARAMETERS,
    'default_params': DEFAULT_GRAPH_EMBEDDING_PARAMS,
    'parameter_categories': PARAMETER_CATEGORIES,
    
    # Task-specific configuration
    'supported_models': [
        'node2vec', 'deepwalk', 'line', 'sdne', 'graphsage', 'gcn'
    ],
    'supported_datasets': [
        'cora', 'citeseer', 'pubmed', 'ppi', 'reddit', 'flickr',
        'ogbn_products', 'ogbn_mag', 'ogbn_papers100m', 'zinc', 'qm9'
    ],
    'metrics': [
        'reconstruction_loss', 'link_prediction_auc', 'node_classification_acc',
        'clustering_coefficient', 'embedding_quality', 'visualization_quality'
    ],
    'visualizations': [
        'embedding_scatter', 't_sne_plot', 'umap_plot', 'pca_plot',
        'embedding_heatmap', 'similarity_matrix', 'clustering_plot'
    ],
    
    # Training configuration
    'training_config': {
        'validation_metric': 'reconstruction_loss',
        'early_stopping_metric': 'val_reconstruction_loss',
        'model_selection_metric': 'val_link_prediction_auc',
        'max_training_time': 3600,  # 1 hour
        'checkpoint_frequency': 10,
        'save_best_model': True,
        'save_embeddings': True
    },
    
    # Evaluation configuration
    'evaluation_config': {
        'link_prediction': True,
        'node_classification': True,
        'graph_classification': True,
        'clustering_evaluation': True,
        'visualization_evaluation': True,
        'compute_statistics': True
    },
    
    # Output configuration
    'output_config': {
        'save_model': True,
        'save_embeddings': True,
        'save_metrics': True,
        'save_visualizations': True,
        'export_format': ['pkl', 'json', 'csv', 'npy', 'h5']
    },
    
    # UI configuration
    'ui_config': {
        'show_parameter_categories': True,
        'show_model_selection': True,
        'show_dataset_selection': True,
        'show_advanced_options': True,
        'show_training_progress': True,
        'show_live_metrics': True,
        'show_visualizations': True,
        'allow_parameter_tuning': True,
        'allow_model_comparison': True,
        'show_embedding_controls': True,
        'show_visualization_options': True
    }
}

# Task metadata for the dashboard
TASK_METADATA = {
    'display_name': 'Graph Embedding Visualization',
    'short_description': 'Learn and visualize graph embeddings',
    'long_description': 'Train embedding models to learn low-dimensional representations of nodes, edges, and graphs. Visualize these embeddings to understand graph structure and relationships.',
    'use_cases': [
        'Graph visualization and exploration',
        'Node similarity analysis',
        'Community detection',
        'Link prediction',
        'Graph clustering',
        'Dimensionality reduction'
    ],
    'difficulty': 'intermediate',
    'estimated_time': '30-60 minutes',
    'required_knowledge': 'Basic understanding of embeddings and visualization',
    'tags': ['embedding', 'visualization', 'unsupervised', 'dimensionality_reduction', 'clustering']
}
