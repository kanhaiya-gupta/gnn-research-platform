"""
Graph Classification Task Configuration
This module defines the configuration for graph classification tasks.
"""

from ..parameters.graph_classification import (
    GRAPH_CLASSIFICATION_PARAMETERS,
    DEFAULT_GRAPH_CLASSIFICATION_PARAMS,
    PARAMETER_CATEGORIES
)

GRAPH_CLASSIFICATION_TASK = {
    'name': 'Graph Classification',
    'description': 'Classify entire graphs into multiple categories',
    'category': 'graph_tasks',
    'icon': 'fas fa-project-diagram',
    'color': 'success',
    'parameters': GRAPH_CLASSIFICATION_PARAMETERS,
    'default_params': DEFAULT_GRAPH_CLASSIFICATION_PARAMS,
    'parameter_categories': PARAMETER_CATEGORIES,
    
    # Task-specific configuration
    'supported_models': [
        'gcn', 'gat', 'graphsage', 'gin', 'chebnet', 'diffpool', 'sortpool'
    ],
    'supported_datasets': [
        'mutag', 'ptc_mr', 'enzymes', 'proteins', 'nci1', 'nci109',
        'reddit_binary', 'reddit_multi_5k', 'collab', 'imdb_binary', 'imdb_multi'
    ],
    'metrics': [
        'accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'auc_pr'
    ],
    'visualizations': [
        'confusion_matrix', 'roc_curve', 'pr_curve', 'graph_distribution',
        'class_distribution', 'feature_importance', 'graph_embeddings'
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
        'graph_embedding_analysis': True,
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
        'show_graph_analysis': True,
        'show_class_distribution': True
    }
}

# Task metadata for the dashboard
TASK_METADATA = {
    'display_name': 'Graph Classification',
    'short_description': 'Classify entire graphs into categories',
    'long_description': 'Train models to classify entire graphs into multiple categories. This task is useful for molecular property prediction, social network analysis, and graph structure understanding.',
    'use_cases': [
        'Molecular property prediction',
        'Social network classification',
        'Chemical compound classification',
        'Protein function prediction',
        'Network topology classification',
        'Graph structure analysis'
    ],
    'difficulty': 'intermediate',
    'estimated_time': '30-60 minutes',
    'required_knowledge': 'Understanding of graph neural networks and classification',
    'tags': ['graph_tasks', 'classification', 'supervised', 'multi_class', 'graph_analysis']
}
