# Graph Classification

## Overview

Graph Classification is a fundamental graph machine learning task that involves classifying entire graphs into multiple categories. Unlike node classification (which classifies individual nodes) or edge classification (which classifies individual edges), graph classification operates at the graph level, making predictions about the entire graph structure and its properties.

## Purpose

Graph classification aims to:
- **Classify graph structures**: Determine the category or type of entire graphs
- **Predict graph properties**: Identify properties that characterize the whole graph
- **Understand graph patterns**: Learn patterns that distinguish different types of graphs
- **Support decision making**: Provide insights for graph-based applications
- **Enable graph comparison**: Compare and categorize different graph structures

## Applications

### Molecular and Chemical Networks
- **Drug discovery**: Classify molecules as active/inactive for drug development
- **Toxicity prediction**: Predict whether chemical compounds are toxic
- **Molecular property prediction**: Predict solubility, bioactivity, and other properties
- **Chemical compound classification**: Categorize compounds by their chemical properties

### Social Networks
- **Community detection**: Classify social networks by community structure
- **Network type classification**: Distinguish between different types of social networks
- **Influence analysis**: Classify networks by their influence patterns
- **Behavior prediction**: Predict collective behavior patterns in networks

### Biological Networks
- **Protein function prediction**: Classify proteins by their functional properties
- **Disease classification**: Classify biological networks by disease association
- **Pathway analysis**: Categorize metabolic or signaling pathways
- **Gene regulatory networks**: Classify regulatory networks by function

### Computer Vision and Image Analysis
- **Image classification**: Classify images represented as graphs
- **Scene understanding**: Understand complex scenes through graph representations
- **Object recognition**: Recognize objects through their graph structures
- **Pattern recognition**: Identify patterns in visual data

### Knowledge Graphs
- **Knowledge base classification**: Classify knowledge graphs by domain
- **Entity relationship analysis**: Understand relationship patterns in knowledge graphs
- **Fact verification**: Classify knowledge graphs by reliability
- **Semantic analysis**: Analyze semantic properties of knowledge structures

## Models

### Graph Convolutional Network (GCN)
- **Architecture**: Uses graph convolutions with global pooling for graph-level classification
- **Global Pooling**: Aggregates node features using mean, max, or sum pooling
- **Advantages**: Simple, effective, computationally efficient
- **Best for**: Homogeneous graphs with clear structural patterns

### Graph Attention Network (GAT)
- **Architecture**: Uses attention mechanisms with global pooling for graph classification
- **Global Pooling**: Attention-weighted aggregation of node features
- **Advantages**: Adaptive feature aggregation, handles heterogeneous graphs
- **Best for**: Complex graphs with varying importance of nodes

### GraphSAGE
- **Architecture**: Samples and aggregates neighborhoods with global pooling
- **Global Pooling**: Inductive learning with graph-level aggregation
- **Advantages**: Scalable to large graphs, inductive learning
- **Best for**: Large-scale graphs with millions of nodes

### Graph Isomorphism Network (GIN)
- **Architecture**: Uses injective aggregation functions with global pooling
- **Global Pooling**: Structure-aware aggregation for graph classification
- **Advantages**: Provably powerful, captures graph structure effectively
- **Best for**: Tasks requiring structural awareness and isomorphism detection

### Chebyshev Graph Convolution (ChebNet)
- **Architecture**: Uses Chebyshev polynomials with global pooling
- **Global Pooling**: Spectral-based aggregation for graph classification
- **Advantages**: Efficient spectral filtering, localized convolutions
- **Best for**: Graphs with clear spectral properties

### Simple Graph Convolution (SGC)
- **Architecture**: Simplified graph convolution with global pooling
- **Global Pooling**: Fast and interpretable graph-level classification
- **Advantages**: Fast training, interpretable, effective baseline
- **Best for**: Quick prototyping and baseline comparisons

## Datasets

### Molecular Datasets (TUDatasets)
- **MUTAG**: 188 mutagenic compounds, 2 classes (mutagenic/non-mutagenic)
- **PTC-MR**: 344 compounds, 2 classes (positive/negative carcinogenicity)
- **ENZYMES**: 600 enzymes, 6 classes (EC top-level classes)
- **PROTEINS**: 1,113 proteins, 2 classes (enzymes/non-enzymes)
- **NCI1**: 4,110 compounds, 2 classes (active/inactive against lung cancer)
- **NCI109**: 4,127 compounds, 2 classes (active/inactive against ovarian cancer)

### Social Network Datasets
- **COLLAB**: 5,000 collaboration networks, 3 classes (physics, high-energy physics, astrophysics)
- **REDDIT-BINARY**: 2,000 Reddit threads, 2 classes (question/discussion)
- **REDDIT-MULTI-5K**: 5,000 Reddit threads, 5 classes
- **REDDIT-MULTI-12K**: 12,000 Reddit threads, 11 classes
- **IMDB-BINARY**: 1,000 movie collaboration networks, 2 classes (action/romance)
- **IMDB-MULTI**: 1,500 movie collaboration networks, 3 classes

### MoleculeNet Datasets
- **BACE**: 1,513 compounds, 2 classes (BACE-1 inhibitors)
- **BBBP**: 2,039 compounds, 2 classes (blood-brain barrier penetration)
- **HIV**: 41,127 compounds, 2 classes (HIV inhibitors)
- **MUV**: 93,087 compounds, 17 classes (multiple targets)
- **TOX21**: 7,831 compounds, 12 classes (toxicity targets)
- **TOXCAST**: 8,575 compounds, 617 classes (toxicity assays)
- **SIDER**: 1,427 compounds, 27 classes (side effects)
- **CLINTOX**: 1,477 compounds, 2 classes (clinical toxicity)

### GNNBenchmark Datasets
- **MNIST**: 70,000 image graphs, 10 classes (digits 0-9)
- **CIFAR10**: 60,000 image graphs, 10 classes (object categories)
- **PATENTS**: 2,000,000 patent graphs, 37 classes (patent categories)
- **ZINC**: 250,000 molecular graphs, regression task (molecular properties)

## Usage

### Basic Usage

```python
from graph_tasks.classification.graph_classification import run_graph_classification_experiment

# Default configuration
config = {
    'model_name': 'gcn',
    'dataset_name': 'mutag',
    'hidden_channels': 64,
    'learning_rate': 0.01,
    'epochs': 200,
    'batch_size': 32
}

# Run experiment
results = run_graph_classification_experiment(config)
print(f"Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
print(f"Test F1 Score: {results['test_metrics']['f1_macro']:.4f}")
```

### Advanced Configuration

```python
config = {
    'model_name': 'gat',
    'dataset_name': 'enzymes',
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

results = run_graph_classification_experiment(config)
```

### Custom Dataset

```python
import torch
from torch_geometric.data import Data, DataLoader

# Create custom graph dataset
graphs = []
num_classes = 3

for i in range(1000):
    # Random number of nodes
    num_nodes = torch.randint(10, 50, (1,)).item()
    
    # Node features
    x = torch.randn(num_nodes, 16)
    
    # Create edges (random graph)
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
    
    # Graph label
    y = torch.randint(0, num_classes, (1,), dtype=torch.long)
    
    # Create graph data
    graph = Data(x=x, edge_index=edge_index, y=y)
    graphs.append(graph)

# Use in experiment
config = {
    'model_name': 'gin',
    'custom_data': graphs,
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
- **num_classes**: Number of graph classes (2-20)
- **batch_norm**: Whether to use batch normalization
- **residual**: Whether to use residual connections

### Graph-Level Parameters
- **readout_method**: Method to aggregate node features (mean, sum, max, attention, sort)
- **pooling_method**: Graph pooling method (diffpool, sortpool, sagpool, edgepool, none)
- **graph_feature_dim**: Dimension of graph-level features (32-512)
- **mlp_layers**: Number of MLP layers for classification (1-5)
- **use_graph_features**: Whether to use additional graph-level features

### Training Parameters
- **learning_rate**: Learning rate for optimization (1e-5 to 0.1)
- **epochs**: Number of training epochs (10-1000)
- **batch_size**: Batch size for training (1-256)
- **optimizer**: Optimization algorithm (adam, sgd, adamw, rmsprop, adagrad)
- **loss_function**: Loss function (cross_entropy, focal_loss, weighted_cross_entropy)
- **weight_decay**: L2 regularization coefficient (0.0-0.1)
- **scheduler**: Learning rate scheduler (none, step, cosine, plateau, exponential)
- **patience**: Early stopping patience (10-100)

## Evaluation Metrics

### Classification Metrics
- **Accuracy**: Overall classification accuracy
- **F1 Score (Macro)**: Harmonic mean of precision and recall (macro-averaged)
- **F1 Score (Weighted)**: Weighted average of F1 scores by class frequency
- **Precision (Macro)**: Macro-averaged precision across classes
- **Recall (Macro)**: Macro-averaged recall across classes
- **AUC-ROC**: Area under the ROC curve (binary) or macro-averaged (multi-class)

### Graph-Specific Metrics
- **Graph Classification Accuracy**: Accuracy on graph classification task
- **Per-Class Performance**: Performance for each graph class
- **Graph Embedding Analysis**: Analysis of learned graph representations
- **Feature Importance**: Importance of different graph features

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
2. **Graph preprocessing**: Remove self-loops, make undirected if appropriate
3. **Class balance**: Ensure balanced graph classes
4. **Validation split**: Use proper train/validation/test splits

### Training Tips
1. **Early stopping**: Use validation performance for early stopping
2. **Learning rate scheduling**: Reduce learning rate when plateauing
3. **Regularization**: Use dropout and weight decay to prevent overfitting
4. **Monitoring**: Track both training and validation metrics
5. **Data augmentation**: Use graph augmentation techniques if needed

## Examples

### Molecular Property Prediction
```python
# Predict drug activity
config = {
    'model_name': 'gin',
    'dataset_name': 'bace',
    'hidden_channels': 256,
    'num_layers': 4,
    'num_classes': 2,  # active/inactive
    'readout_method': 'attention',
    'learning_rate': 0.001,
    'epochs': 300
}
```

### Social Network Classification
```python
# Classify Reddit communities
config = {
    'model_name': 'gat',
    'dataset_name': 'reddit_multi_5k',
    'hidden_channels': 128,
    'heads': 8,
    'num_classes': 5,
    'readout_method': 'mean',
    'batch_size': 64
}
```

### Image Graph Classification
```python
# Classify MNIST digits as graphs
config = {
    'model_name': 'gcn',
    'dataset_name': 'mnist',
    'hidden_channels': 64,
    'num_classes': 10,  # digits 0-9
    'readout_method': 'max',
    'learning_rate': 0.01,
    'epochs': 200
}
```

### Chemical Compound Classification
```python
# Classify chemical compounds
config = {
    'model_name': 'chebnet',
    'dataset_name': 'mutag',
    'hidden_channels': 128,
    'num_layers': 3,
    'K': 3,  # Chebyshev polynomial order
    'num_classes': 2,  # mutagenic/non-mutagenic
    'readout_method': 'sum'
}
```

## Troubleshooting

### Common Issues

**Low Accuracy**
- Check class balance in graph labels
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
5. **Graph Preprocessing**: Optimize graph structure

## Integration with Web Interface

The graph classification functionality is fully integrated with the web interface:

1. **Parameter Configuration**: All parameters can be set through the web UI
2. **Model Selection**: Choose from available models via dropdown
3. **Dataset Selection**: Select from available datasets
4. **Real-time Training**: Monitor training progress with live updates
5. **Result Visualization**: View results and metrics in interactive charts
6. **Model Comparison**: Compare multiple models side-by-side
7. **Graph Analysis**: Analyze individual graphs and their classifications

## Advanced Features

### Multi-Graph Learning
- Support for multiple graphs per sample
- Graph ensemble methods
- Hierarchical graph classification

### Interpretable Graph Classification
- Attention weight visualization
- Feature importance analysis
- Graph structure explanation
- Decision path analysis

### Graph Augmentation
- Edge dropping and addition
- Node feature perturbation
- Subgraph sampling
- Graph transformation techniques

### Transfer Learning
- Pre-trained graph models
- Domain adaptation
- Few-shot graph classification
- Meta-learning approaches

## Future Enhancements

- **Temporal graph classification**: Handle time-evolving graphs
- **Heterogeneous graph classification**: Support for different node and edge types
- **Few-shot graph classification**: Handle scenarios with limited labeled graphs
- **Adversarial graph classification**: Robust against adversarial attacks
- **Federated graph classification**: Privacy-preserving distributed learning
- **Causal graph classification**: Understand causal relationships in graphs
- **Explainable graph classification**: Provide interpretable predictions 