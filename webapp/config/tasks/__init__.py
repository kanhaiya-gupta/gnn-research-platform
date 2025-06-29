"""
Task Registry
This module provides a centralized registry of all available GNN tasks.
"""

# Import all task configurations
from .node_classification import NODE_CLASSIFICATION_TASK, TASK_METADATA as NODE_CLASSIFICATION_METADATA
from .link_prediction import LINK_PREDICTION_TASK, TASK_METADATA as LINK_PREDICTION_METADATA

# Import task configurations for other purposes
# Note: These would be imported when the corresponding files are created
# from .node_regression import NODE_REGRESSION_TASK, TASK_METADATA as NODE_REGRESSION_METADATA
# from .edge_classification import EDGE_CLASSIFICATION_TASK, TASK_METADATA as EDGE_CLASSIFICATION_METADATA
# from .graph_classification import GRAPH_CLASSIFICATION_TASK, TASK_METADATA as GRAPH_CLASSIFICATION_METADATA
# from .graph_regression import GRAPH_REGRESSION_TASK, TASK_METADATA as GRAPH_REGRESSION_METADATA
# from .community_detection import COMMUNITY_DETECTION_TASK, TASK_METADATA as COMMUNITY_DETECTION_METADATA
# from .anomaly_detection import ANOMALY_DETECTION_TASK, TASK_METADATA as ANOMALY_DETECTION_METADATA
# from .graph_generation import GRAPH_GENERATION_TASK, TASK_METADATA as GRAPH_GENERATION_METADATA
# from .graph_embedding import GRAPH_EMBEDDING_TASK, TASK_METADATA as GRAPH_EMBEDDING_METADATA
# from .dynamic_graph_learning import DYNAMIC_GRAPH_LEARNING_TASK, TASK_METADATA as DYNAMIC_GRAPH_LEARNING_METADATA

# Task registry dictionary
TASK_REGISTRY = {
    'node_classification': NODE_CLASSIFICATION_TASK,
    'link_prediction': LINK_PREDICTION_TASK,
    # Add other tasks as they are implemented
    # 'node_regression': NODE_REGRESSION_TASK,
    # 'edge_classification': EDGE_CLASSIFICATION_TASK,
    # 'graph_classification': GRAPH_CLASSIFICATION_TASK,
    # 'graph_regression': GRAPH_REGRESSION_TASK,
    # 'community_detection': COMMUNITY_DETECTION_TASK,
    # 'anomaly_detection': ANOMALY_DETECTION_TASK,
    # 'graph_generation': GRAPH_GENERATION_TASK,
    # 'graph_embedding': GRAPH_EMBEDDING_TASK,
    # 'dynamic_graph_learning': DYNAMIC_GRAPH_LEARNING_TASK,
}

# Metadata registry
TASK_METADATA_REGISTRY = {
    'node_classification': NODE_CLASSIFICATION_METADATA,
    'link_prediction': LINK_PREDICTION_METADATA,
    # Add other task metadata as they are implemented
    # 'node_regression': NODE_REGRESSION_METADATA,
    # 'edge_classification': EDGE_CLASSIFICATION_METADATA,
    # 'graph_classification': GRAPH_CLASSIFICATION_METADATA,
    # 'graph_regression': GRAPH_REGRESSION_METADATA,
    # 'community_detection': COMMUNITY_DETECTION_METADATA,
    # 'anomaly_detection': ANOMALY_DETECTION_METADATA,
    # 'graph_generation': GRAPH_GENERATION_METADATA,
    # 'graph_embedding': GRAPH_EMBEDDING_METADATA,
    # 'dynamic_graph_learning': DYNAMIC_GRAPH_LEARNING_METADATA,
}

# Task categories for organization
TASK_CATEGORIES = {
    'node_tasks': {
        'name': 'Node-Level Tasks',
        'description': 'Tasks that operate on individual nodes',
        'icon': 'fas fa-circle',
        'color': 'primary',
        'tasks': ['node_classification', 'node_regression']
    },
    'edge_tasks': {
        'name': 'Edge-Level Tasks',
        'description': 'Tasks that operate on edges or node pairs',
        'icon': 'fas fa-link',
        'color': 'info',
        'tasks': ['edge_classification', 'link_prediction']
    },
    'graph_tasks': {
        'name': 'Graph-Level Tasks',
        'description': 'Tasks that operate on entire graphs',
        'icon': 'fas fa-project-diagram',
        'color': 'success',
        'tasks': ['graph_classification', 'graph_regression']
    },
    'community_detection': {
        'name': 'Community Detection',
        'description': 'Tasks for detecting communities in graphs',
        'icon': 'fas fa-users',
        'color': 'warning',
        'tasks': ['community_detection']
    },
    'anomaly_detection': {
        'name': 'Anomaly Detection',
        'description': 'Tasks for detecting anomalies in graphs',
        'icon': 'fas fa-exclamation-triangle',
        'color': 'danger',
        'tasks': ['anomaly_detection']
    },
    'generation': {
        'name': 'Graph Generation',
        'description': 'Tasks for generating new graphs',
        'icon': 'fas fa-magic',
        'color': 'purple',
        'tasks': ['graph_generation']
    },
    'embedding': {
        'name': 'Graph Embedding',
        'description': 'Tasks for learning graph embeddings',
        'icon': 'fas fa-eye',
        'color': 'secondary',
        'tasks': ['graph_embedding']
    },
    'dynamic': {
        'name': 'Dynamic Graph Learning',
        'description': 'Tasks for learning from dynamic graphs',
        'icon': 'fas fa-clock',
        'color': 'dark',
        'tasks': ['dynamic_graph_learning']
    }
}

def get_task_config(task_name):
    """Get task configuration by name."""
    return TASK_REGISTRY.get(task_name)

def get_task_metadata(task_name):
    """Get task metadata by name."""
    return TASK_METADATA_REGISTRY.get(task_name)

def get_all_tasks():
    """Get all available tasks."""
    return TASK_REGISTRY

def get_all_metadata():
    """Get all task metadata."""
    return TASK_METADATA_REGISTRY

def get_tasks_by_category(category):
    """Get tasks by category."""
    return TASK_CATEGORIES.get(category, {}).get('tasks', [])

def get_all_categories():
    """Get all task categories."""
    return TASK_CATEGORIES

def is_task_available(task_name):
    """Check if a task is available."""
    return task_name in TASK_REGISTRY

def get_task_parameters(task_name):
    """Get parameters for a specific task."""
    task_config = get_task_config(task_name)
    return task_config.get('parameters', {}) if task_config else {}

def get_task_default_params(task_name):
    """Get default parameters for a specific task."""
    task_config = get_task_config(task_name)
    return task_config.get('default_params', {}) if task_config else {}

def get_task_parameter_categories(task_name):
    """Get parameter categories for a specific task."""
    task_config = get_task_config(task_name)
    return task_config.get('parameter_categories', {}) if task_config else {}
