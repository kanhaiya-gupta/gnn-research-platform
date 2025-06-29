# Edge Classification

## Overview

Edge Classification is a fundamental graph machine learning task that involves predicting the class or type of edges in a graph. This task is essential for understanding relationships between entities in various domains such as social networks, biological networks, knowledge graphs, and recommendation systems.

## Purpose

Edge classification aims to:
- **Categorize relationships**: Determine the type or category of connections between nodes
- **Predict edge properties**: Classify edges based on their characteristics or attributes
- **Understand network structure**: Gain insights into the nature of relationships in the graph
- **Support downstream tasks**: Provide labeled edge information for other graph analysis tasks

## Applications

### Social Networks
- **Friend relationship types**: Classify connections as family, colleague, acquaintance, etc.
- **Interaction patterns**: Identify types of interactions (likes, comments, shares)
- **Community detection**: Understand relationship dynamics within communities

### Biological Networks
- **Protein-protein interactions**: Classify interaction types (binding, regulatory, structural)
- **Drug-target interactions**: Predict interaction mechanisms
- **Gene regulatory networks**: Identify regulatory relationship types

### Knowledge Graphs
- **Entity relationships**: Classify semantic relationships between entities
- **Fact verification**: Determine the validity of relationships
- **Knowledge completion**: Predict missing relationship types

### Recommendation Systems
- **User-item interactions**: Classify user preferences and behaviors
- **Product relationships**: Identify product similarity or complementarity
- **Trust networks**: Classify trust levels between users

## Models

### Graph Convolutional Network (GCN)
- **Architecture**: Uses graph convolutions to aggregate neighbor information
- **Advantages**: Simple, effective, computationally efficient
- **Best for**: Homogeneous graphs with clear neighborhood patterns

### Graph Attention Network (GAT)
- **Architecture**: Uses attention mechanisms to weight neighbor contributions
- **Advantages**: Adaptive feature aggregation, handles heterogeneous graphs
- **Best for**: Complex graphs with varying importance of neighbors

### GraphSAGE
- **Architecture**: Samples and aggregates from fixed-size neighborhoods
- **Advantages**: Scalable to large graphs, inductive learning
- **Best for**: Large-scale graphs with millions of nodes

### Graph Isomorphism Network (GIN)
- **Architecture**: Uses injective aggregation functions for maximum discriminative power
- **Advantages**: Provably powerful, captures graph structure effectively
- **Best for**: Tasks requiring structural awareness

### Chebyshev Graph Convolution (ChebNet)
- **Architecture**: Uses Chebyshev polynomials for spectral graph convolution
- **Advantages**: Efficient spectral filtering, localized convolutions
- **Best for**: Graphs with clear spectral properties

### Simple Graph Convolution (SGC)
- **Architecture**: Simplified graph convolution with linear layers
- **Advantages**: Fast training, interpretable, effective baseline
- **Best for**: Quick prototyping and baseline comparisons

## Datasets

### Citation Networks
- **Cora**: 2,708 papers with 5,429 citations, 7 classes
- **Citeseer**: 3,327 papers with 4,732 citations, 6 classes
- **PubMed**: 19,717 papers with 44,338 citations, 3 classes

### Social Networks
- **Reddit**: Large-scale social network with user interactions
- **Flickr**: Photo-sharing network with user relationships

### E-commerce Networks
- **Amazon Photo**: Product co-purchase network
- **Amazon Computers**: Computer product relationships

### Academic Networks
- **Coauthor CS**: Computer science collaboration network
- **Coauthor Physics**: Physics collaboration network

## Usage

### Basic Usage

```python
from edge_tasks.classification.edge_classification import run_edge_classification_experiment

# Default configuration
config = {
    'model_name': 'gcn',
    'dataset_name': 'cora',
    'hidden_channels': 64,
    'learning_rate': 0.01,
    'epochs': 200
}

# Run experiment
results = run_edge_classification_experiment(config)
print(f"Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
```

### Advanced Configuration

```python
config = {
    'model_name': 'gat',
    'dataset_name': 'citeseer',
    'hidden_channels': 128,
    'num_layers': 3,
    'heads': 8,
    'dropout': 0.2,
    'learning_rate': 0.005,
    'weight_decay': 1e-4,
    'epochs': 300,
    'patience': 50,
    'device': 'cuda'
}

results = run_edge_classification_experiment(config)
```

### Custom Dataset

```python
import torch
from torch_geometric.data import Data

# Create custom graph data
num_nodes = 1000
num_features = 16
num_classes = 5

# Node features
x = torch.randn(num_nodes, num_features)

# Edge index (source, target pairs)
edge_index = torch.randint(0, num_nodes, (2, 2000))

# Edge labels
edge_labels = torch.randint(0, num_classes, (edge_index.size(1),))

# Create data object
data = Data(x=x, edge_index=edge_index, edge_labels=edge_labels)

# Use in experiment
config = {
    'model_name': 'gin',
    'custom_data': data,
    'hidden_channels': 64,
    'num_classes': num_classes
}
```

## Parameters

### Architecture Parameters
- **hidden_channels**: Dimension of hidden layers (16-512)
- **num_layers**: Number of GNN layers (1-10)
- **dropout**: Dropout rate for regularization (0.0-0.5)
- **activation**: Activation function (relu, tanh, sigmoid, leaky_relu, elu)
- **batch_norm**: Whether to use batch normalization
- **residual**: Whether to use residual connections

### Training Parameters
- **learning_rate**: Learning rate for optimization (1e-5 to 0.1)
- **epochs**: Number of training epochs (10-1000)
- **batch_size**: Batch size for training (1-256)
- **optimizer**: Optimization algorithm (adam, sgd, adamw, rmsprop, adagrad)
- **weight_decay**: L2 regularization coefficient (0.0-0.1)
- **patience**: Early stopping patience (10-100)

### Edge Classification Specific
- **edge_feature_dim**: Dimension of edge features (8-256)
- **use_edge_features**: Whether to use edge features
- **class_weights**: Weights for handling class imbalance
- **focal_loss_gamma**: Gamma parameter for focal loss (0.0-5.0)

## Evaluation Metrics

### Classification Metrics
- **Accuracy**: Overall classification accuracy
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: Ratio of true positives to predicted positives
- **Recall**: Ratio of true positives to actual positives
- **AUC-ROC**: Area under the ROC curve
- **Confusion Matrix**: Detailed classification breakdown

### Edge-Specific Metrics
- **Edge Classification Accuracy**: Accuracy on edge classification task
- **Per-Class Performance**: Performance for each edge class
- **Edge Feature Importance**: Importance of different edge features

## Best Practices

### Model Selection
1. **Start with GCN**: Use GCN as a baseline for most tasks
2. **Try GAT for complex graphs**: Use GAT when neighbor importance varies
3. **Use GraphSAGE for large graphs**: Scale to millions of nodes
4. **Consider GIN for structural tasks**: When graph structure is crucial

### Hyperparameter Tuning
1. **Learning rate**: Start with 0.01, adjust based on convergence
2. **Hidden dimensions**: 64-128 usually works well
3. **Dropout**: 0.1-0.3 for regularization
4. **Number of layers**: 2-4 layers typically sufficient

### Data Preparation
1. **Feature normalization**: Normalize node features
2. **Edge balance**: Ensure balanced edge classes
3. **Graph preprocessing**: Remove self-loops, make undirected if appropriate
4. **Validation split**: Use proper train/validation/test splits

### Training Tips
1. **Early stopping**: Use validation performance for early stopping
2. **Learning rate scheduling**: Reduce learning rate when plateauing
3. **Regularization**: Use dropout and weight decay to prevent overfitting
4. **Monitoring**: Track both training and validation metrics

## Examples

### Social Network Analysis
```python
# Classify friendship types in a social network
config = {
    'model_name': 'gat',
    'dataset_name': 'synthetic_social',
    'hidden_channels': 128,
    'heads': 8,
    'num_classes': 4,  # family, friend, colleague, acquaintance
    'learning_rate': 0.01,
    'epochs': 200
}
```

### Biological Network Analysis
```python
# Classify protein interaction types
config = {
    'model_name': 'gin',
    'dataset_name': 'protein_interaction',
    'hidden_channels': 256,
    'num_layers': 4,
    'num_classes': 3,  # binding, regulatory, structural
    'use_edge_features': True,
    'edge_feature_dim': 64
}
```

### Knowledge Graph Completion
```python
# Classify entity relationship types
config = {
    'model_name': 'gcn',
    'dataset_name': 'knowledge_graph',
    'hidden_channels': 64,
    'num_classes': 10,  # various relationship types
    'learning_rate': 0.005,
    'weight_decay': 1e-4
}
```

## Troubleshooting

### Common Issues

**Low Accuracy**
- Check class balance in edge labels
- Increase model capacity (more layers/hidden dimensions)
- Try different activation functions
- Adjust learning rate

**Overfitting**
- Increase dropout rate
- Add weight decay
- Reduce model complexity
- Use early stopping

**Slow Training**
- Reduce hidden dimensions
- Use fewer layers
- Try SGC for faster training
- Use GPU acceleration

**Memory Issues**
- Reduce batch size
- Use smaller hidden dimensions
- Process graph in chunks
- Use sparse operations

### Performance Optimization
1. **GPU Usage**: Ensure CUDA is available and used
2. **Data Loading**: Use efficient data loaders
3. **Model Architecture**: Choose appropriate model complexity
4. **Batch Processing**: Use appropriate batch sizes

## Integration with Web Interface

The edge classification functionality is fully integrated with the web interface:

1. **Parameter Configuration**: All parameters can be set through the web UI
2. **Model Selection**: Choose from available models via dropdown
3. **Dataset Selection**: Select from available datasets
4. **Real-time Training**: Monitor training progress with live updates
5. **Result Visualization**: View results and metrics in interactive charts
6. **Model Comparison**: Compare multiple models side-by-side

## Future Enhancements

- **Multi-label classification**: Support for edges with multiple labels
- **Temporal edge classification**: Handle time-evolving edge types
- **Heterogeneous graphs**: Support for different node and edge types
- **Few-shot learning**: Handle scenarios with limited labeled edges
- **Active learning**: Intelligent selection of edges to label
- **Interpretability**: Explain edge classification decisions 