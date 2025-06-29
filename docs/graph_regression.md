# Graph Regression

## Overview

Graph Regression is a fundamental graph machine learning task that involves predicting continuous values for entire graphs. Unlike graph classification (which predicts discrete categories), graph regression predicts continuous numerical properties that characterize the whole graph structure and its characteristics.

## Purpose

Graph regression aims to:
- **Predict continuous properties**: Estimate numerical values that characterize entire graphs
- **Model graph relationships**: Learn relationships between graph structure and continuous outcomes
- **Support quantitative analysis**: Provide numerical predictions for graph-based applications
- **Enable property estimation**: Predict physical, chemical, or structural properties of graphs
- **Support decision making**: Provide quantitative insights for graph-based decision processes

## Applications

### Molecular and Chemical Networks
- **Molecular property prediction**: Predict solubility, boiling point, melting point, and other physical properties
- **Drug discovery**: Predict drug efficacy, toxicity, and pharmacokinetic properties
- **Chemical compound analysis**: Estimate chemical reactivity, stability, and other properties
- **Material science**: Predict material properties like conductivity, strength, and thermal properties

### Biological Networks
- **Protein property prediction**: Predict protein stability, binding affinity, and functional properties
- **Drug-target interaction**: Predict binding strength and interaction properties
- **Metabolic pathway analysis**: Predict pathway efficiency and metabolic properties
- **Gene expression prediction**: Predict expression levels and regulatory properties

### Social Networks
- **Network centrality prediction**: Predict influence scores and centrality measures
- **Community strength estimation**: Predict community cohesion and stability measures
- **Information flow prediction**: Predict information propagation rates and reach
- **Network growth modeling**: Predict network evolution and growth patterns

### Computer Vision and Image Analysis
- **Image property prediction**: Predict image complexity, aesthetic scores, and quality measures
- **Scene understanding**: Predict scene depth, complexity, and spatial relationships
- **Object property estimation**: Predict object size, distance, and spatial properties
- **Visual quality assessment**: Predict image quality scores and perceptual measures

### Knowledge Graphs
- **Knowledge completeness**: Predict completeness scores and coverage measures
- **Entity relationship strength**: Predict relationship confidence and strength scores
- **Knowledge quality assessment**: Predict quality scores and reliability measures
- **Semantic similarity**: Predict semantic similarity scores between entities

## Models

### Graph Convolutional Network (GCN)
- **Architecture**: Uses graph convolutions with global pooling for graph-level regression
- **Global Pooling**: Aggregates node features using mean, max, or sum pooling
- **Regression Head**: MLP layers for continuous value prediction
- **Advantages**: Simple, effective, computationally efficient
- **Best for**: Homogeneous graphs with clear structural patterns

### Graph Attention Network (GAT)
- **Architecture**: Uses attention mechanisms with global pooling for graph regression
- **Global Pooling**: Attention-weighted aggregation of node features
- **Regression Head**: Attention-aware regression for continuous prediction
- **Advantages**: Adaptive feature aggregation, handles heterogeneous graphs
- **Best for**: Complex graphs with varying importance of nodes

### GraphSAGE
- **Architecture**: Samples and aggregates neighborhoods with global pooling
- **Global Pooling**: Inductive learning with graph-level aggregation
- **Regression Head**: Scalable regression for large graphs
- **Advantages**: Scalable to large graphs, inductive learning
- **Best for**: Large-scale graphs with millions of nodes

### Graph Isomorphism Network (GIN)
- **Architecture**: Uses injective aggregation functions with global pooling
- **Global Pooling**: Structure-aware aggregation for graph regression
- **Regression Head**: Structure-aware regression for continuous prediction
- **Advantages**: Provably powerful, captures graph structure effectively
- **Best for**: Tasks requiring structural awareness and isomorphism detection

### Chebyshev Graph Convolution (ChebNet)
- **Architecture**: Uses Chebyshev polynomials with global pooling
- **Global Pooling**: Spectral-based aggregation for graph regression
- **Regression Head**: Spectral-aware regression for continuous prediction
- **Advantages**: Efficient spectral filtering, localized convolutions
- **Best for**: Graphs with clear spectral properties

### Simple Graph Convolution (SGC)
- **Architecture**: Simplified graph convolution with global pooling
- **Global Pooling**: Fast and interpretable graph-level regression
- **Regression Head**: Fast regression for continuous prediction
- **Advantages**: Fast training, interpretable, effective baseline
- **Best for**: Quick prototyping and baseline comparisons

## Datasets

### Molecular Property Datasets (MoleculeNet)
- **ESOL**: 1,128 compounds, 1 target (water solubility)
- **FreeSolv**: 642 compounds, 1 target (hydration free energy)
- **Lipophilicity**: 4,200 compounds, 1 target (octanol/water distribution coefficient)
- **QM7**: 7,160 molecules, 1 target (atomization energy)
- **QM8**: 21,786 molecules, 12 targets (electronic properties)
- **QM9**: 133,885 molecules, 12 targets (quantum mechanical properties)
- **Delaney**: 1,127 compounds, 1 target (ESOL solubility)

### GNNBenchmark Datasets
- **ZINC**: 250,000 molecular graphs, 1 target (constrained solubility)
- **Alchemy**: 119,487 molecules, 12 targets (quantum mechanical properties)

### Chemical Property Datasets
- **Toxicity datasets**: Various toxicity prediction tasks
- **Drug discovery datasets**: Drug efficacy and safety prediction
- **Material property datasets**: Physical and chemical properties

### Synthetic Datasets
- **Custom graphs**: Generated with specific properties for testing
- **Controlled experiments**: Graphs with known regression targets

## Usage

### Basic Usage

```python
from graph_tasks.regression.graph_regression import run_graph_regression_experiment

# Default configuration
config = {
    'model_name': 'gcn',
    'dataset_name': 'esol',
    'hidden_channels': 64,
    'learning_rate': 0.01,
    'epochs': 200,
    'batch_size': 32
}

# Run experiment
results = run_graph_regression_experiment(config)
print(f"Test MSE: {results['test_metrics']['overall']['mse']:.4f}")
print(f"Test R²: {results['test_metrics']['overall']['r2']:.4f}")
```

### Advanced Configuration

```python
config = {
    'model_name': 'gat',
    'dataset_name': 'qm9',
    'hidden_channels': 128,
    'num_layers': 4,
    'heads': 8,
    'dropout': 0.2,
    'learning_rate': 0.005,
    'weight_decay': 1e-4,
    'epochs': 300,
    'patience': 50,
    'batch_size': 64,
    'device': 'cuda'
}

results = run_graph_regression_experiment(config)
```

### Custom Dataset

```python
import torch
from torch_geometric.data import Data, DataLoader

# Create custom graph dataset
graphs = []
num_targets = 2

for i in range(1000):
    # Random number of nodes
    num_nodes = torch.randint(10, 50, (1,)).item()
    
    # Node features
    x = torch.randn(num_nodes, 16)
    
    # Create edges (random graph)
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
    
    # Graph target (continuous values)
    target = torch.randn(num_targets, dtype=torch.float)
    
    # Create graph data
    graph = Data(x=x, edge_index=edge_index, y=target)
    graphs.append(graph)

# Use in experiment
config = {
    'model_name': 'gin',
    'custom_data': graphs,
    'hidden_channels': 64,
    'num_targets': num_targets
}
```

## Parameters

### Architecture Parameters
- **hidden_channels**: Dimension of hidden layers (16-512)
- **num_layers**: Number of GNN layers (1-10)
- **dropout**: Dropout rate for regularization (0.0-0.5)
- **activation**: Activation function (relu, tanh, sigmoid, leaky_relu, elu)
- **output_dim**: Number of output values per graph (1-10)
- **batch_norm**: Whether to use batch normalization
- **residual**: Whether to use residual connections

### Graph-Level Parameters
- **readout_method**: Method to aggregate node features (mean, sum, max, attention, sort)
- **pooling_method**: Graph pooling method (diffpool, sortpool, sagpool, edgepool, none)
- **graph_feature_dim**: Dimension of graph-level features (32-512)
- **mlp_layers**: Number of MLP layers for regression (1-5)
- **use_graph_features**: Whether to use additional graph-level features

### Training Parameters
- **learning_rate**: Learning rate for optimization (1e-5 to 0.1)
- **epochs**: Number of training epochs (10-1000)
- **batch_size**: Batch size for training (1-256)
- **optimizer**: Optimization algorithm (adam, sgd, adamw, rmsprop, adagrad)
- **loss_function**: Loss function (mse, mae, huber, smooth_l1, log_cosh)
- **weight_decay**: L2 regularization coefficient (0.0-0.1)
- **scheduler**: Learning rate scheduler (none, step, cosine, plateau, exponential)
- **patience**: Early stopping patience (10-100)

## Evaluation Metrics

### Regression Metrics
- **Mean Squared Error (MSE)**: Average squared difference between predictions and targets
- **Root Mean Squared Error (RMSE)**: Square root of MSE, in same units as target
- **Mean Absolute Error (MAE)**: Average absolute difference between predictions and targets
- **R-squared (R²)**: Proportion of variance explained by the model (0-1, higher is better)
- **Explained Variance Score**: Similar to R² but can be negative for poor models

### Graph-Specific Metrics
- **Per-Target Performance**: Performance for each regression target
- **Overall Performance**: Average performance across all targets
- **Prediction Distribution**: Analysis of prediction accuracy across target ranges
- **Feature Importance**: Importance of different graph features for prediction

## Best Practices

### Model Selection
1. **Start with GCN**: Use GCN as a baseline for most tasks
2. **Try GAT for complex graphs**: Use GAT when node importance varies
3. **Use GraphSAGE for large graphs**: Scale to millions of nodes
4. **Consider GIN for structural tasks**: When graph structure is crucial
5. **Use SGC for fast prototyping**: Quick baseline comparisons

### Global Pooling Strategy
1. **Mean pooling**: Good for most tasks, preserves average node information
2. **Max pooling**: Captures extreme values, good for outlier detection
3. **Sum pooling**: Preserves total information, good for counting tasks
4. **Attention pooling**: Adaptive aggregation, good for complex graphs

### Hyperparameter Tuning
1. **Learning rate**: Start with 0.01, adjust based on convergence
2. **Hidden dimensions**: 64-128 usually works well
3. **Dropout**: 0.1-0.3 for regularization
4. **Number of layers**: 2-4 layers typically sufficient
5. **Batch size**: 32-64 for most datasets

### Data Preparation
1. **Feature normalization**: Normalize node features
2. **Target normalization**: Normalize regression targets
3. **Graph preprocessing**: Remove self-loops, make undirected if appropriate
4. **Validation split**: Use proper train/validation/test splits

### Training Tips
1. **Early stopping**: Use validation MSE for early stopping
2. **Learning rate scheduling**: Reduce learning rate when plateauing
3. **Regularization**: Use dropout and weight decay to prevent overfitting
4. **Monitoring**: Track both training and validation metrics
5. **Data augmentation**: Use graph augmentation techniques if needed

## Examples

### Molecular Property Prediction
```python
# Predict water solubility
config = {
    'model_name': 'gin',
    'dataset_name': 'esol',
    'hidden_channels': 256,
    'num_layers': 4,
    'readout_method': 'attention',
    'learning_rate': 0.001,
    'epochs': 300
}
```

### Drug Discovery
```python
# Predict drug efficacy
config = {
    'model_name': 'gat',
    'dataset_name': 'qm9',
    'hidden_channels': 128,
    'heads': 8,
    'num_targets': 12,  # multiple properties
    'readout_method': 'mean',
    'batch_size': 64
}
```

### Material Science
```python
# Predict material properties
config = {
    'model_name': 'gcn',
    'dataset_name': 'zinc',
    'hidden_channels': 64,
    'readout_method': 'max',
    'learning_rate': 0.005,
    'weight_decay': 1e-4
}
```

### Social Network Analysis
```python
# Predict network properties
config = {
    'model_name': 'graphsage',
    'dataset_name': 'synthetic_social',
    'hidden_channels': 128,
    'readout_method': 'sum',
    'num_targets': 3,  # multiple network metrics
    'use_graph_features': True
}
```

## Troubleshooting

### Common Issues

**High MSE/Low R²**
- Check target normalization
- Increase model capacity (more layers/hidden dimensions)
- Try different readout methods
- Adjust learning rate and training time

**Overfitting**
- Increase dropout rate
- Add weight decay
- Reduce model complexity
- Use early stopping
- Increase training data

**Slow Training**
- Reduce hidden dimensions
- Use fewer layers
- Try SGC for faster training
- Use GPU acceleration
- Increase batch size

**Memory Issues**
- Reduce batch size
- Use smaller hidden dimensions
- Process graphs in smaller batches
- Use sparse operations

### Performance Optimization
1. **GPU Usage**: Ensure CUDA is available and used
2. **Data Loading**: Use efficient data loaders
3. **Model Architecture**: Choose appropriate model complexity
4. **Batch Processing**: Use appropriate batch sizes
5. **Target Normalization**: Normalize regression targets

## Integration with Web Interface

The graph regression functionality is fully integrated with the web interface:

1. **Parameter Configuration**: All parameters can be set through the web UI
2. **Model Selection**: Choose from available models via dropdown
3. **Dataset Selection**: Select from available datasets
4. **Real-time Training**: Monitor training progress with live updates
5. **Result Visualization**: View results and metrics in interactive charts
6. **Model Comparison**: Compare multiple models side-by-side
7. **Prediction Interface**: Make predictions on new data

## Advanced Features

### Multi-Target Regression
- Support for multiple regression targets
- Target-specific loss functions
- Independent target evaluation
- Target correlation analysis

### Uncertainty Quantification
- Prediction confidence intervals
- Uncertainty estimation
- Probabilistic predictions
- Calibration analysis

### Interpretable Regression
- Feature importance analysis
- Attention weight visualization
- Graph structure explanation
- Decision path analysis

### Transfer Learning
- Pre-trained graph models
- Domain adaptation
- Few-shot regression
- Meta-learning approaches

## Future Enhancements

- **Temporal graph regression**: Handle time-evolving graphs
- **Heterogeneous graph regression**: Support for different node and edge types
- **Few-shot graph regression**: Handle scenarios with limited labeled graphs
- **Adversarial graph regression**: Robust against adversarial attacks
- **Federated graph regression**: Privacy-preserving distributed learning
- **Causal graph regression**: Understand causal relationships in graphs
- **Explainable graph regression**: Provide interpretable predictions 