"""
Parameters Module
This module provides access to task parameters and validation functions.
"""

from typing import Dict, Any, Optional
import importlib

# Import individual parameter files with fallback for missing DEFAULT_PARAMETERS
def safe_import_parameters(module_name: str):
    """Safely import parameters and defaults from a module."""
    try:
        module = importlib.import_module(f'.{module_name}', package='config.parameters')
        parameters = getattr(module, f'{module_name.upper()}_PARAMETERS', {})
        
        # Try to get DEFAULT_PARAMETERS, fallback to task-specific default
        try:
            defaults = getattr(module, 'DEFAULT_PARAMETERS', {})
        except AttributeError:
            # Try to find task-specific default parameters
            default_attr_name = f'DEFAULT_{module_name.upper()}_PARAMS'
            defaults = getattr(module, default_attr_name, {})
        
        return parameters, defaults
    except Exception as e:
        print(f"Warning: Could not import parameters from {module_name}: {e}")
        return {}, {}

# Import all parameter files safely
node_classification_params, node_classification_defaults = safe_import_parameters('node_classification')
node_regression_params, node_regression_defaults = safe_import_parameters('node_regression')
link_prediction_params, link_prediction_defaults = safe_import_parameters('link_prediction')
edge_classification_params, edge_classification_defaults = safe_import_parameters('edge_classification')
graph_classification_params, graph_classification_defaults = safe_import_parameters('graph_classification')
graph_regression_params, graph_regression_defaults = safe_import_parameters('graph_regression')
community_detection_params, community_detection_defaults = safe_import_parameters('community_detection')
anomaly_detection_params, anomaly_detection_defaults = safe_import_parameters('anomaly_detection')
graph_generation_params, graph_generation_defaults = safe_import_parameters('graph_generation')
graph_embedding_params, graph_embedding_defaults = safe_import_parameters('graph_embedding')
dynamic_graph_learning_params, dynamic_graph_learning_defaults = safe_import_parameters('dynamic_graph_learning')

# Parameter registry
PARAMETER_REGISTRY = {
    'node_classification': node_classification_params,
    'node_regression': node_regression_params,
    'link_prediction': link_prediction_params,
    'edge_classification': edge_classification_params,
    'graph_classification': graph_classification_params,
    'graph_regression': graph_regression_params,
    'community_detection': community_detection_params,
    'anomaly_detection': anomaly_detection_params,
    'graph_generation': graph_generation_params,
    'graph_embedding': graph_embedding_params,
    'dynamic_graph_learning': dynamic_graph_learning_params,
}

# Default parameters registry
DEFAULT_PARAMETERS_REGISTRY = {
    'node_classification': node_classification_defaults,
    'node_regression': node_regression_defaults,
    'link_prediction': link_prediction_defaults,
    'edge_classification': edge_classification_defaults,
    'graph_classification': graph_classification_defaults,
    'graph_regression': graph_regression_defaults,
    'community_detection': community_detection_defaults,
    'anomaly_detection': anomaly_detection_defaults,
    'graph_generation': graph_generation_defaults,
    'graph_embedding': graph_embedding_defaults,
    'dynamic_graph_learning': dynamic_graph_learning_defaults,
}

def get_task_parameters(task_name: str) -> Dict[str, Any]:
    """Get parameters for a specific task."""
    return PARAMETER_REGISTRY.get(task_name, {})

def get_task_default_params(task_name: str) -> Dict[str, Any]:
    """Get default parameters for a specific task."""
    return DEFAULT_PARAMETERS_REGISTRY.get(task_name, {})

def validate_parameter_value(param_name: str, value: Any, param_config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a parameter value against its configuration."""
    validation_result = {
        'valid': True,
        'message': '',
        'value': value
    }
    
    try:
        param_type = param_config.get('type', 'any')
        
        # Type validation
        if param_type == 'int':
            if not isinstance(value, int):
                validation_result['valid'] = False
                validation_result['message'] = f'{param_name} must be an integer'
                return validation_result
        elif param_type == 'float':
            if not isinstance(value, (int, float)):
                validation_result['valid'] = False
                validation_result['message'] = f'{param_name} must be a number'
                return validation_result
        elif param_type == 'bool':
            if not isinstance(value, bool):
                validation_result['valid'] = False
                validation_result['message'] = f'{param_name} must be a boolean'
                return validation_result
        elif param_type == 'str':
            if not isinstance(value, str):
                validation_result['valid'] = False
                validation_result['message'] = f'{param_name} must be a string'
                return validation_result
        
        # Range validation
        if 'min' in param_config and value < param_config['min']:
            validation_result['valid'] = False
            validation_result['message'] = f'{param_name} must be at least {param_config["min"]}'
            return validation_result
        
        if 'max' in param_config and value > param_config['max']:
            validation_result['valid'] = False
            validation_result['message'] = f'{param_name} must be at most {param_config["max"]}'
            return validation_result
        
        # Options validation
        if 'options' in param_config and value not in param_config['options']:
            validation_result['valid'] = False
            validation_result['message'] = f'{param_name} must be one of {param_config["options"]}'
            return validation_result
        
        return validation_result
        
    except Exception as e:
        validation_result['valid'] = False
        validation_result['message'] = f'Validation error: {str(e)}'
        return validation_result

def get_all_parameters() -> Dict[str, Any]:
    """Get all task parameters."""
    return PARAMETER_REGISTRY

def get_all_default_params() -> Dict[str, Any]:
    """Get all default parameters."""
    return DEFAULT_PARAMETERS_REGISTRY

def is_task_parameter_valid(task_name: str, param_name: str, value: Any) -> Dict[str, Any]:
    """Check if a parameter value is valid for a specific task."""
    task_params = get_task_parameters(task_name)
    param_config = task_params.get(param_name, {})
    
    if not param_config:
        return {
            'valid': False,
            'message': f'Parameter {param_name} not found for task {task_name}',
            'value': value
        }
    
    return validate_parameter_value(param_name, value, param_config)
