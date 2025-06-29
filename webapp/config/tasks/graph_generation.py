"""
Graph Generation Task Configuration
This module defines the configuration for graph generation tasks.
"""

from ..parameters.graph_generation import (
    GRAPH_GENERATION_PARAMETERS,
    DEFAULT_GRAPH_GENERATION_PARAMS,
    PARAMETER_CATEGORIES
)

GRAPH_GENERATION_TASK = {
    'name': 'Graph Generation',
    'description': 'Generate new graphs with desired properties',
    'category': 'generation',
    'icon': 'fas fa-magic',
    'color': 'purple',
    'parameters': GRAPH_GENERATION_PARAMETERS,
    'default_params': DEFAULT_GRAPH_GENERATION_PARAMS,
    'parameter_categories': PARAMETER_CATEGORIES,
    
    # Task-specific configuration
    'supported_models': [
        'vae', 'gan', 'flow', 'diffusion', 'autoregressive', 'graphvae', 'graphgan'
    ],
    'supported_datasets': [
        'zinc', 'qm9', 'qm7', 'mutag', 'ptc_mr', 'enzymes',
        'proteins', 'nci1', 'nci109', 'reddit_binary', 'reddit_multi_5k'
    ],
    'metrics': [
        'validity', 'uniqueness', 'novelty', 'diversity', 'fcd', 'mmd'
    ],
    'visualizations': [
        'generated_graphs', 'property_distribution', 'latent_space',
        'training_progress', 'quality_metrics'
    ],
    
    # Training configuration
    'training_config': {
        'validation_metric': 'validity',
        'early_stopping_metric': 'val_validity',
        'model_selection_metric': 'val_uniqueness',
        'max_training_time': 7200,  # 2 hours
        'checkpoint_frequency': 20,
        'save_best_model': True,
        'save_generated_samples': True
    },
    
    # Evaluation configuration
    'evaluation_config': {
        'num_generated_samples': 1000,
        'property_evaluation': True,
        'structural_evaluation': True,
        'diversity_evaluation': True,
        'compute_statistics': True
    },
    
    # Output configuration
    'output_config': {
        'save_model': True,
        'save_generated_graphs': True,
        'save_metrics': True,
        'save_visualizations': True,
        'export_format': ['pkl', 'json', 'csv', 'sdf']
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
        'show_generation_controls': True
    }
}

# Task metadata for the dashboard
TASK_METADATA = {
    'display_name': 'Graph Generation',
    'short_description': 'Generate new graphs with desired properties',
    'long_description': 'Train generative models to create new graphs with specific properties. This task is crucial for drug discovery, molecular design, and synthetic data generation.',
    'use_cases': [
        'Drug discovery and molecular design',
        'Synthetic data generation',
        'Network topology design',
        'Chemical compound generation',
        'Social network simulation'
    ],
    'difficulty': 'advanced',
    'estimated_time': '60-120 minutes',
    'required_knowledge': 'Understanding of generative models and graph theory',
    'tags': ['generation', 'graphs', 'unsupervised', 'generative', 'molecules']
}
