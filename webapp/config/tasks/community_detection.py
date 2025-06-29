"""
Community Detection Task Configuration
This module defines the configuration for community detection tasks.
"""

from ..parameters.community_detection import (
    COMMUNITY_DETECTION_PARAMETERS,
    DEFAULT_COMMUNITY_DETECTION_PARAMS,
    PARAMETER_CATEGORIES
)

COMMUNITY_DETECTION_TASK = {
    'name': 'Community Detection',
    'description': 'Detect communities or clusters in graphs',
    'category': 'community_detection',
    'icon': 'fas fa-users',
    'color': 'warning',
    'parameters': COMMUNITY_DETECTION_PARAMETERS,
    'default_params': DEFAULT_COMMUNITY_DETECTION_PARAMS,
    'parameter_categories': PARAMETER_CATEGORIES,
    
    # Task-specific configuration
    'supported_models': [
        'gcn', 'gat', 'graphsage', 'gin', 'chebnet', 'sgc', 'appnp'
    ],
    'supported_datasets': [
        'karate', 'football', 'polbooks', 'dolphins', 'les_miserables',
        'cora', 'citeseer', 'pubmed', 'ppi', 'reddit', 'flickr'
    ],
    'metrics': [
        'modularity', 'conductance', 'normalized_cut', 'silhouette_score',
        'adjusted_rand_index', 'normalized_mutual_info'
    ],
    'visualizations': [
        'community_network', 'modularity_heatmap', 'community_distribution',
        'node_embeddings', 'community_overlap', 'hierarchical_clustering'
    ],
    
    # Training configuration
    'training_config': {
        'validation_metric': 'modularity',
        'early_stopping_metric': 'val_modularity',
        'model_selection_metric': 'val_conductance',
        'max_training_time': 3600,  # 1 hour
        'checkpoint_frequency': 10,
        'save_best_model': True,
        'save_communities': True
    },
    
    # Evaluation configuration
    'evaluation_config': {
        'cross_validation': True,
        'modularity_evaluation': True,
        'conductance_evaluation': True,
        'community_quality_analysis': True,
        'node_embedding_analysis': True,
        'compute_statistics': True
    },
    
    # Output configuration
    'output_config': {
        'save_model': True,
        'save_communities': True,
        'save_metrics': True,
        'save_visualizations': True,
        'export_format': ['pkl', 'json', 'csv', 'npy']
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
        'show_community_analysis': True,
        'show_network_visualization': True
    }
}

# Task metadata for the dashboard
TASK_METADATA = {
    'display_name': 'Community Detection',
    'short_description': 'Detect communities in graphs',
    'long_description': 'Train models to detect communities or clusters in graphs. This task is useful for social network analysis, biological network analysis, and understanding graph structure.',
    'use_cases': [
        'Social network community detection',
        'Biological network analysis',
        'Recommendation systems',
        'Network security analysis',
        'Market segmentation',
        'Graph structure understanding'
    ],
    'difficulty': 'intermediate',
    'estimated_time': '30-60 minutes',
    'required_knowledge': 'Understanding of graph neural networks and clustering',
    'tags': ['community_detection', 'clustering', 'unsupervised', 'network_analysis', 'social_networks']
}
