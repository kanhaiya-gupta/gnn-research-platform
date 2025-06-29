"""
Link Prediction Task Configuration
This module defines the configuration for link prediction tasks.
"""

from ..parameters.link_prediction import (
    LINK_PREDICTION_PARAMETERS,
    DEFAULT_LINK_PREDICTION_PARAMS,
    PARAMETER_CATEGORIES
)

LINK_PREDICTION_TASK = {
    'name': 'Link Prediction',
    'description': 'Predict missing or future links in a graph',
    'category': 'edge_tasks',
    'icon': 'fas fa-link',
    'color': 'info',
    'parameters': LINK_PREDICTION_PARAMETERS,
    'default_params': DEFAULT_LINK_PREDICTION_PARAMS,
    'parameter_categories': PARAMETER_CATEGORIES,
    
    # Task-specific configuration
    'supported_models': [
        'gcn', 'gat', 'graphsage', 'gin', 'vgae', 'seal', 'n2v'
    ],
    'supported_datasets': [
        'cora', 'citeseer', 'pubmed', 'ogbn_arxiv', 'ogbn_products',
        'ogbn_mag', 'reddit', 'flickr', 'amazon_photo', 'amazon_computers'
    ],
    'metrics': [
        'auc', 'ap', 'precision_at_k', 'recall_at_k', 'f1_score', 'hits_at_k'
    ],
    'visualizations': [
        'link_predictions', 'roc_curve', 'precision_recall_curve',
        'embedding_visualization', 'attention_weights'
    ],
    
    # Training configuration
    'training_config': {
        'validation_metric': 'auc',
        'early_stopping_metric': 'val_auc',
        'model_selection_metric': 'val_ap',
        'max_training_time': 3600,  # 1 hour
        'checkpoint_frequency': 10,
        'save_best_model': True,
        'save_predictions': True
    },
    
    # Evaluation configuration
    'evaluation_config': {
        'cross_validation_folds': 5,
        'test_size': 0.2,
        'random_state': 42,
        'stratify': True,
        'compute_confidence_intervals': True,
        'negative_sampling_ratio': 1.0
    },
    
    # Output configuration
    'output_config': {
        'save_model': True,
        'save_embeddings': True,
        'save_predictions': True,
        'save_metrics': True,
        'save_visualizations': True,
        'export_format': ['pkl', 'json', 'csv']
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
        'allow_model_comparison': True
    }
}

# Task metadata for the dashboard
TASK_METADATA = {
    'display_name': 'Link Prediction',
    'short_description': 'Predict missing or future links',
    'long_description': 'Train Graph Neural Networks to predict missing or future links in a graph. This task is essential for understanding network evolution and discovering hidden relationships.',
    'use_cases': [
        'Friend recommendation in social networks',
        'Product recommendation in e-commerce',
        'Drug-target interaction prediction',
        'Knowledge graph completion',
        'Network evolution prediction'
    ],
    'difficulty': 'intermediate',
    'estimated_time': '45-90 minutes',
    'required_knowledge': 'Understanding of link prediction and graph embeddings',
    'tags': ['prediction', 'links', 'supervised', 'graph', 'embeddings']
}
