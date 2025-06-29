# GNN Platform Configuration

This directory contains the configuration system for the Graph Neural Network Platform. The configuration is organized into several key components that work together to provide a comprehensive and flexible system for managing GNN experiments.

## Directory Structure

```
config/
├── README.md                    # This file
├── config.py                    # Main platform configuration
├── models.py                    # Model definitions and configurations
├── parameters.py                # Parameter management utilities
├── parameters/                  # Task-specific parameter definitions
│   ├── __init__.py
│   ├── node_classification.py
│   ├── node_regression.py
│   ├── edge_classification.py
│   ├── link_prediction.py
│   ├── graph_classification.py
│   ├── graph_regression.py
│   ├── community_detection.py
│   ├── anomaly_detection.py
│   ├── graph_generation.py
│   ├── graph_embedding.py
│   └── dynamic_graph_learning.py
└── tasks/                       # Task configurations and metadata
    ├── __init__.py
    ├── node_classification.py
    ├── link_prediction.py
    └── ... (other task files)
```

## Configuration Components

### 1. Main Configuration (`config.py`)

The main configuration file contains platform-wide settings including:

- **Platform Configuration**: Basic platform information, UI settings, feature flags
- **Dataset Configuration**: Available datasets and their metadata
- **Model Configuration**: Supported models and their capabilities
- **Training Configuration**: Default training settings, optimizers, schedulers
- **Visualization Configuration**: Plot settings and interactive features
- **Export Configuration**: Supported export formats and settings
- **Logging Configuration**: Logging levels and file management
- **Development Configuration**: Debug settings and development tools

### 2. Model Configuration (`models.py`)

Defines all available GNN models with their:

- **Model Metadata**: Name, description, paper reference, category
- **Architecture Details**: Type, inductive capabilities, supported tasks
- **Model-Specific Parameters**: Parameters unique to each model
- **Implementation Details**: Framework, class, optimization strategies

**Supported Models:**
- GCN (Graph Convolutional Network)
- GAT (Graph Attention Network)
- GraphSAGE
- GIN (Graph Isomorphism Network)
- ChebNet (Chebyshev Graph Convolution)
- VGAE (Variational Graph Autoencoder)
- SEAL (Subgraphs, Embeddings and Attributes for Link prediction)
- SGC (Simple Graph Convolution)
- APPNP (Approximate Personalized Propagation of Neural Predictions)

### 3. Parameter Management (`parameters.py`)

Provides utilities for managing parameters across all tasks:

- **Parameter Registry**: Central registry of all task parameters
- **Validation Rules**: Type-specific validation and conversion
- **Parameter Categories**: Organization of parameters by category
- **Utility Functions**: Helper functions for parameter management

### 4. Task-Specific Parameters (`parameters/`)

Each task has its own parameter definition file containing:

- **Task-Specific Parameters**: Parameters unique to each task type
- **Default Values**: Sensible defaults for quick start
- **Parameter Categories**: UI organization categories
- **Validation Rules**: Task-specific validation requirements

**Available Tasks:**
- Node Classification
- Node Regression
- Edge Classification
- Link Prediction
- Graph Classification
- Graph Regression
- Community Detection
- Anomaly Detection
- Graph Generation
- Graph Embedding
- Dynamic Graph Learning

### 5. Task Configuration (`tasks/`)

Each task has a configuration file defining:

- **Task Metadata**: Display information, descriptions, use cases
- **Supported Models**: Models that can be used for this task
- **Supported Datasets**: Datasets appropriate for this task
- **Metrics**: Evaluation metrics for this task
- **Visualizations**: Available visualizations
- **Training Configuration**: Task-specific training settings
- **Evaluation Configuration**: Evaluation protocols
- **Output Configuration**: What to save and export
- **UI Configuration**: Interface settings

## Parameter System

### Parameter Types

The system supports several parameter types:

1. **Integer Parameters**: Numeric parameters with min/max ranges
2. **Float Parameters**: Decimal parameters with precision control
3. **Boolean Parameters**: True/false parameters
4. **Select Parameters**: Dropdown options from a predefined list
5. **String Parameters**: Text input parameters

### Parameter Categories

Parameters are organized into categories for better UI organization:

- **Architecture Parameters**: Model architecture configuration
- **Training Parameters**: Training configuration and optimization
- **Regularization Parameters**: Techniques to prevent overfitting
- **Dataset Parameters**: Dataset configuration and preprocessing
- **Model-Specific Parameters**: Parameters specific to certain models
- **Task-Specific Parameters**: Parameters unique to each task type

### Parameter Validation

The system includes comprehensive parameter validation:

- **Type Validation**: Ensures parameters match their defined types
- **Range Validation**: Validates numeric parameters against min/max ranges
- **Option Validation**: Ensures select parameters use valid options
- **Required Validation**: Checks that required parameters are provided

## Usage Examples

### Getting Task Parameters

```python
from config.parameters import get_task_parameters, get_task_default_params

# Get all parameters for node classification
params = get_task_parameters('node_classification')

# Get default parameters
defaults = get_task_default_params('node_classification')
```

### Validating Parameters

```python
from config.parameters import validate_parameter_value

# Validate a parameter value
is_valid, result = validate_parameter_value('node_classification', 'hidden_dim', 128)
if is_valid:
    print(f"Valid value: {result}")
else:
    print(f"Invalid: {result}")
```

### Getting Model Information

```python
from config.models import get_model_config, get_models_for_task

# Get model configuration
model_config = get_model_config('gcn')

# Get models that support a task
supported_models = get_models_for_task('node_classification')
```

### Getting Task Information

```python
from config.tasks import get_task_config, get_task_metadata

# Get task configuration
task_config = get_task_config('node_classification')

# Get task metadata
metadata = get_task_metadata('node_classification')
```

## Adding New Tasks

To add a new task to the platform:

1. **Create Parameter File**: Add a new file in `parameters/` with task-specific parameters
2. **Create Task File**: Add a new file in `tasks/` with task configuration
3. **Update Registries**: Add the new task to the parameter and task registries
4. **Add Templates**: Create experiment and results templates in the templates directory

### Example: Adding a New Task

```python
# In parameters/new_task.py
NEW_TASK_PARAMETERS = {
    'param1': {
        'name': 'Parameter 1',
        'description': 'Description of parameter 1',
        'type': 'int',
        'default': 64,
        'min': 16,
        'max': 512,
        'step': 16,
        'category': 'architecture',
        'required': True
    }
}

# In tasks/new_task.py
NEW_TASK_CONFIG = {
    'name': 'New Task',
    'description': 'Description of the new task',
    'category': 'custom',
    'icon': 'fas fa-star',
    'color': 'info',
    'parameters': NEW_TASK_PARAMETERS,
    'supported_models': ['gcn', 'gat'],
    'metrics': ['accuracy', 'f1_score']
}
```

## Configuration Best Practices

1. **Consistent Naming**: Use consistent naming conventions across all configuration files
2. **Comprehensive Documentation**: Include detailed descriptions for all parameters
3. **Sensible Defaults**: Provide reasonable default values for all parameters
4. **Validation**: Include proper validation rules for all parameters
5. **Categories**: Organize parameters into logical categories for better UI
6. **Extensibility**: Design the system to be easily extensible for new tasks and models

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all parameter files are properly imported in the registries
2. **Validation Errors**: Check that parameter definitions include proper validation rules
3. **Missing Parameters**: Verify that all required parameters are defined
4. **Type Mismatches**: Ensure parameter types match their validation rules

### Debugging

Enable debug mode in the configuration to get detailed logging:

```python
from config.config import DEV_CONFIG
DEV_CONFIG['debug_mode'] = True
```

## Contributing

When contributing to the configuration system:

1. Follow the existing patterns and conventions
2. Add comprehensive documentation for new parameters
3. Include validation rules for all new parameters
4. Test parameter validation thoroughly
5. Update this README when adding new features

## License

This configuration system is part of the GNN Platform and follows the same license terms. 