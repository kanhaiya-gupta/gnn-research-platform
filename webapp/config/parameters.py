"""
Parameters Configuration
This module provides utilities for managing parameters across all GNN tasks.
"""

# Import all parameter modules
from .parameters.node_classification import NODE_CLASSIFICATION_PARAMETERS, DEFAULT_NODE_CLASSIFICATION_PARAMS
from .parameters.node_regression import NODE_REGRESSION_PARAMETERS, DEFAULT_NODE_REGRESSION_PARAMS
from .parameters.edge_classification import EDGE_CLASSIFICATION_PARAMETERS, DEFAULT_EDGE_CLASSIFICATION_PARAMS
from .parameters.link_prediction import LINK_PREDICTION_PARAMETERS, DEFAULT_LINK_PREDICTION_PARAMS
from .parameters.graph_classification import GRAPH_CLASSIFICATION_PARAMETERS, DEFAULT_GRAPH_CLASSIFICATION_PARAMS
from .parameters.graph_regression import GRAPH_REGRESSION_PARAMETERS, DEFAULT_GRAPH_REGRESSION_PARAMS
from .parameters.community_detection import COMMUNITY_DETECTION_PARAMETERS, DEFAULT_COMMUNITY_DETECTION_PARAMS
from .parameters.anomaly_detection import ANOMALY_DETECTION_PARAMETERS, DEFAULT_ANOMALY_DETECTION_PARAMS
from .parameters.graph_generation import GRAPH_GENERATION_PARAMETERS, DEFAULT_GRAPH_GENERATION_PARAMS
from .parameters.graph_embedding import GRAPH_EMBEDDING_PARAMETERS, DEFAULT_GRAPH_EMBEDDING_PARAMS
from .parameters.dynamic_graph_learning import DYNAMIC_GRAPH_LEARNING_PARAMETERS, DEFAULT_DYNAMIC_GRAPH_LEARNING_PARAMS

# Parameter registry for all tasks
PARAMETER_REGISTRY = {
    'node_classification': NODE_CLASSIFICATION_PARAMETERS,
    'node_regression': NODE_REGRESSION_PARAMETERS,
    'edge_classification': EDGE_CLASSIFICATION_PARAMETERS,
    'link_prediction': LINK_PREDICTION_PARAMETERS,
    'graph_classification': GRAPH_CLASSIFICATION_PARAMETERS,
    'graph_regression': GRAPH_REGRESSION_PARAMETERS,
    'community_detection': COMMUNITY_DETECTION_PARAMETERS,
    'anomaly_detection': ANOMALY_DETECTION_PARAMETERS,
    'graph_generation': GRAPH_GENERATION_PARAMETERS,
    'graph_embedding': GRAPH_EMBEDDING_PARAMETERS,
    'dynamic_graph_learning': DYNAMIC_GRAPH_LEARNING_PARAMETERS
}

# Default parameters registry
DEFAULT_PARAMETERS_REGISTRY = {
    'node_classification': DEFAULT_NODE_CLASSIFICATION_PARAMS,
    'node_regression': DEFAULT_NODE_REGRESSION_PARAMS,
    'edge_classification': DEFAULT_EDGE_CLASSIFICATION_PARAMS,
    'link_prediction': DEFAULT_LINK_PREDICTION_PARAMS,
    'graph_classification': DEFAULT_GRAPH_CLASSIFICATION_PARAMS,
    'graph_regression': DEFAULT_GRAPH_REGRESSION_PARAMS,
    'community_detection': DEFAULT_COMMUNITY_DETECTION_PARAMS,
    'anomaly_detection': DEFAULT_ANOMALY_DETECTION_PARAMS,
    'graph_generation': DEFAULT_GRAPH_GENERATION_PARAMS,
    'graph_embedding': DEFAULT_GRAPH_EMBEDDING_PARAMS,
    'dynamic_graph_learning': DEFAULT_DYNAMIC_GRAPH_LEARNING_PARAMS
}

# Common parameter categories across all tasks
COMMON_PARAMETER_CATEGORIES = {
    'architecture': {
        'name': 'Architecture Parameters',
        'description': 'Model architecture configuration',
        'icon': 'fas fa-cogs',
        'color': 'primary'
    },
    'training': {
        'name': 'Training Parameters',
        'description': 'Training configuration and optimization',
        'icon': 'fas fa-graduation-cap',
        'color': 'success'
    },
    'regularization': {
        'name': 'Regularization Parameters',
        'description': 'Regularization techniques to prevent overfitting',
        'icon': 'fas fa-shield-alt',
        'color': 'warning'
    },
    'dataset': {
        'name': 'Dataset Parameters',
        'description': 'Dataset configuration and preprocessing',
        'icon': 'fas fa-database',
        'color': 'secondary'
    },
    'model_specific': {
        'name': 'Model-Specific Parameters',
        'description': 'Parameters specific to certain model architectures',
        'icon': 'fas fa-tools',
        'color': 'dark'
    }
}

# Parameter validation rules
PARAMETER_VALIDATION_RULES = {
    'int': {
        'validate': lambda x, param: isinstance(x, int) and param.get('min', float('-inf')) <= x <= param.get('max', float('inf')),
        'convert': lambda x: int(x) if x is not None else None,
        'error_message': 'Value must be an integer within the specified range'
    },
    'float': {
        'validate': lambda x, param: isinstance(x, (int, float)) and param.get('min', float('-inf')) <= x <= param.get('max', float('inf')),
        'convert': lambda x: float(x) if x is not None else None,
        'error_message': 'Value must be a number within the specified range'
    },
    'bool': {
        'validate': lambda x, param: isinstance(x, bool),
        'convert': lambda x: bool(x) if x is not None else None,
        'error_message': 'Value must be a boolean'
    },
    'select': {
        'validate': lambda x, param: x in param.get('options', []),
        'convert': lambda x: str(x) if x is not None else None,
        'error_message': 'Value must be one of the available options'
    },
    'string': {
        'validate': lambda x, param: isinstance(x, str),
        'convert': lambda x: str(x) if x is not None else None,
        'error_message': 'Value must be a string'
    }
}

def get_task_parameters(task_name):
    """Get parameters for a specific task."""
    return PARAMETER_REGISTRY.get(task_name, {})

def get_task_default_params(task_name):
    """Get default parameters for a specific task."""
    return DEFAULT_PARAMETERS_REGISTRY.get(task_name, {})

def get_parameter_definition(task_name, param_name):
    """Get parameter definition for a specific task and parameter."""
    task_params = get_task_parameters(task_name)
    return task_params.get(param_name, {})

def validate_parameter_value(task_name, param_name, value):
    """Validate a parameter value for a specific task."""
    param_def = get_parameter_definition(task_name, param_name)
    if not param_def:
        return False, f"Parameter '{param_name}' not found for task '{task_name}'"
    
    param_type = param_def.get('type', 'string')
    validation_rule = PARAMETER_VALIDATION_RULES.get(param_type)
    
    if not validation_rule:
        return True, "No validation rule for parameter type"
    
    try:
        converted_value = validation_rule['convert'](value)
        if validation_rule['validate'](converted_value, param_def):
            return True, converted_value
        else:
            return False, validation_rule['error_message']
    except (ValueError, TypeError):
        return False, f"Could not convert value to {param_type}"

def get_parameter_categories(task_name):
    """Get parameter categories for a specific task."""
    task_params = get_task_parameters(task_name)
    categories = {}
    
    for param_name, param_def in task_params.items():
        category = param_def.get('category', 'other')
        if category not in categories:
            categories[category] = []
        categories[category].append(param_name)
    
    return categories

def get_parameters_by_category(task_name, category):
    """Get parameters for a specific task and category."""
    task_params = get_task_parameters(task_name)
    return {name: defn for name, defn in task_params.items() 
            if defn.get('category') == category}

def get_required_parameters(task_name):
    """Get required parameters for a specific task."""
    task_params = get_task_parameters(task_name)
    return {name: defn for name, defn in task_params.items() 
            if defn.get('required', False)}

def get_optional_parameters(task_name):
    """Get optional parameters for a specific task."""
    task_params = get_task_parameters(task_name)
    return {name: defn for name, defn in task_params.items() 
            if not defn.get('required', False)}

def merge_parameters(task_name, user_params, use_defaults=True):
    """Merge user parameters with defaults for a specific task."""
    default_params = get_task_default_params(task_name) if use_defaults else {}
    merged_params = default_params.copy()
    
    for param_name, value in user_params.items():
        is_valid, result = validate_parameter_value(task_name, param_name, value)
        if is_valid:
            merged_params[param_name] = result
        else:
            # Use default value if validation fails
            if param_name in default_params:
                merged_params[param_name] = default_params[param_name]
    
    return merged_params

def get_parameter_ui_config(task_name, param_name):
    """Get UI configuration for a specific parameter."""
    param_def = get_parameter_definition(task_name, param_name)
    if not param_def:
        return {}
    
    ui_config = {
        'name': param_def.get('name', param_name),
        'description': param_def.get('description', ''),
        'type': param_def.get('type', 'string'),
        'required': param_def.get('required', False),
        'default': param_def.get('default'),
        'category': param_def.get('category', 'other')
    }
    
    # Add type-specific UI configuration
    param_type = param_def.get('type', 'string')
    if param_type == 'int' or param_type == 'float':
        ui_config.update({
            'min': param_def.get('min'),
            'max': param_def.get('max'),
            'step': param_def.get('step', 1 if param_type == 'int' else 0.1)
        })
    elif param_type == 'select':
        ui_config['options'] = param_def.get('options', [])
    elif param_type == 'bool':
        ui_config['options'] = [True, False]
    
    return ui_config

def get_all_parameter_categories():
    """Get all parameter categories."""
    return COMMON_PARAMETER_CATEGORIES

def get_parameter_summary(task_name):
    """Get a summary of parameters for a specific task."""
    task_params = get_task_parameters(task_name)
    summary = {
        'total_parameters': len(task_params),
        'required_parameters': len(get_required_parameters(task_name)),
        'optional_parameters': len(get_optional_parameters(task_name)),
        'categories': get_parameter_categories(task_name)
    }
    return summary

def export_parameters(task_name, format='json'):
    """Export parameters for a specific task."""
    task_params = get_task_parameters(task_name)
    default_params = get_task_default_params(task_name)
    
    export_data = {
        'task_name': task_name,
        'parameters': task_params,
        'default_parameters': default_params,
        'categories': get_parameter_categories(task_name)
    }
    
    if format == 'json':
        import json
        return json.dumps(export_data, indent=2)
    elif format == 'yaml':
        import yaml
        return yaml.dump(export_data, default_flow_style=False)
    else:
        return export_data 