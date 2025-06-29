"""
Node Classification Task Configuration
This module defines the configuration for node classification tasks.
"""

from ..parameters.node_classification import (
    NODE_CLASSIFICATION_PARAMETERS,
    DEFAULT_NODE_CLASSIFICATION_PARAMS,
    PARAMETER_CATEGORIES
)

NODE_CLASSIFICATION_TASK = {
    'name': 'Node Classification',
    'description': 'Classify nodes in a graph into predefined categories',
    'category': 'node_tasks',
    'icon': 'fas fa-tags',
    'color': 'primary',
    'parameters': NODE_CLASSIFICATION_PARAMETERS,
    'default_params': DEFAULT_NODE_CLASSIFICATION_PARAMS,
    'parameter_categories': PARAMETER_CATEGORIES,
    
    # Task-specific configuration
    'supported_models': [
        'gcn', 'gat', 'graphsage', 'gin', 'chebnet', 'sgc', 'appnp'
    ],
    'supported_datasets': [
        'cora', 'citeseer', 'pubmed', 'ogbn_arxiv', 'ogbn_products',
        'ogbn_mag', 'reddit', 'flickr', 'amazon_photo', 'amazon_computers'
    ],
    'metrics': [
        'accuracy', 'f1_score', 'precision', 'recall', 'confusion_matrix'
    ],
    'visualizations': [
        'node_embeddings', 'confusion_matrix', 'classification_report',
        'feature_importance', 'attention_weights'
    ],
    
    # Training configuration
    'training_config': {
        'validation_metric': 'accuracy',
        'early_stopping_metric': 'val_accuracy',
        'model_selection_metric': 'val_f1_score',
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
        'compute_confidence_intervals': True
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
    'display_name': 'Node Classification',
    'short_description': 'Classify nodes into categories',
    'long_description': 'Train Graph Neural Networks to classify nodes in a graph into predefined categories. This task is fundamental for understanding node properties and relationships in graph-structured data.',
    'use_cases': [
        'Document classification in citation networks',
        'User classification in social networks',
        'Protein function prediction',
        'Product categorization in e-commerce',
        'Fraud detection in financial networks'
    ],
    'difficulty': 'beginner',
    'estimated_time': '30-60 minutes',
    'required_knowledge': 'Basic understanding of machine learning and graphs',
    'tags': ['classification', 'nodes', 'supervised', 'graph', 'neural_networks']
}
