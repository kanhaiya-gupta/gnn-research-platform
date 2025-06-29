"""
Edge Classification Task Configuration
This module defines the configuration for edge classification tasks.
"""

from ..parameters.edge_classification import (
    EDGE_CLASSIFICATION_PARAMETERS,
    DEFAULT_EDGE_CLASSIFICATION_PARAMS,
    PARAMETER_CATEGORIES
)

EDGE_CLASSIFICATION_TASK = {
    'name': 'Edge Classification',
    'description': 'Classify edges into multiple categories or types',
    'category': 'edge_tasks',
    'icon': 'fas fa-tags',
    'color': 'info',
    'parameters': EDGE_CLASSIFICATION_PARAMETERS,
    'default_params': DEFAULT_EDGE_CLASSIFICATION_PARAMS,
    'parameter_categories': PARAMETER_CATEGORIES,
    
    # Task-specific configuration
    'supported_models': [
        'gcn', 'gat', 'graphsage', 'gin', 'chebnet', 'vgae', 'seal'
    ],
    'supported_datasets': [
        'cora', 'citeseer', 'pubmed', 'ppi', 'reddit', 'flickr',
        'ogbn_products', 'ogbn_mag', 'ogbn_papers100m', 'zinc', 'qm9'
    ],
    'metrics': [
        'accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'auc_pr'
    ],
    'visualizations': [
        'confusion_matrix', 'roc_curve', 'pr_curve', 'edge_distribution',
        'class_distribution', 'feature_importance', 'edge_embeddings'
    ],
    
    # Training configuration
    'training_config': {
        'validation_metric': 'f1_score',
        'early_stopping_metric': 'val_f1_score',
        'model_selection_metric': 'val_accuracy',
        'max_training_time': 3600,  # 1 hour
        'checkpoint_frequency': 10,
        'save_best_model': True,
        'save_predictions': True
    },
    
    # Evaluation configuration
    'evaluation_config': {
        'cross_validation': True,
        'stratified_sampling': True,
        'class_balance_evaluation': True,
        'feature_importance_analysis': True,
        'edge_embedding_analysis': True,
        'compute_statistics': True
    },
    
    # Output configuration
    'output_config': {
        'save_model': True,
        'save_predictions': True,
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
        'show_edge_analysis': True,
        'show_class_distribution': True
    }
}

# Task metadata for the dashboard
TASK_METADATA = {
    'display_name': 'Edge Classification',
    'short_description': 'Classify edges into multiple categories',
    'long_description': 'Train models to classify edges in graphs into multiple categories or types. This task is useful for understanding edge relationships, detecting edge types, and analyzing graph structure.',
    'use_cases': [
        'Social network edge type classification',
        'Molecular bond type prediction',
        'Network traffic classification',
        'Relationship type detection',
        'Edge anomaly detection',
        'Graph structure analysis'
    ],
    'difficulty': 'intermediate',
    'estimated_time': '30-60 minutes',
    'required_knowledge': 'Understanding of graph neural networks and classification',
    'tags': ['edge_tasks', 'classification', 'supervised', 'multi_class', 'graph_analysis']
}
