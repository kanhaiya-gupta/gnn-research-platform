# Community Detection

## Overview

Community Detection is a fundamental graph analysis task that involves identifying groups or communities of nodes that are more densely connected internally than with the rest of the network. These communities represent cohesive subgraphs where nodes within the same community have stronger connections than nodes across different communities.

## Purpose

Community detection aims to:
- **Identify cohesive groups**: Find sets of nodes that form tightly-knit communities
- **Understand network structure**: Reveal the hierarchical organization of networks
- **Discover functional modules**: Identify functional units in biological or social networks
- **Support network analysis**: Provide insights for understanding network dynamics
- **Enable targeted interventions**: Identify key communities for targeted actions

## Applications

### Social Networks
- **Friend groups**: Identify circles of friends and social communities
- **Interest groups**: Find communities based on shared interests or activities
- **Influence analysis**: Understand how information spreads within communities
- **Recommendation systems**: Suggest connections within communities

### Biological Networks
- **Protein complexes**: Identify functional protein modules
- **Metabolic pathways**: Find metabolic pathway communities
- **Gene regulatory networks**: Discover co-regulated gene communities
- **Disease modules**: Identify disease-related protein communities

### Information Networks
- **Topic communities**: Find communities discussing similar topics
- **Citation networks**: Identify research communities and collaborations
- **Web communities**: Discover related websites and content clusters
- **Knowledge graphs**: Find related entity communities

### Infrastructure Networks
- **Power grids**: Identify power distribution communities
- **Transportation networks**: Find transportation hubs and communities
- **Communication networks**: Identify communication clusters
- **Supply chains**: Find supply chain communities

### Financial Networks
- **Trading communities**: Identify groups of related traders
- **Banking networks**: Find banking communities and clusters
- **Market segments**: Identify market communities and sectors
- **Risk assessment**: Understand risk propagation through communities

## Models

### Graph Convolutional Network (GCN)
- **Architecture**: Uses graph convolutions to learn node representations
- **Community Detection**: Learns node embeddings and predicts community assignments
- **Advantages**: Simple, effective, computationally efficient
- **Best for**: Homogeneous graphs with clear community structure

### Graph Attention Network (GAT)
- **Architecture**: Uses attention mechanisms to weight neighbor contributions
- **Community Detection**: Attention-weighted embeddings for community prediction
- **Advantages**: Adaptive feature aggregation, handles heterogeneous graphs
- **Best for**: Complex graphs with varying importance of neighbors

### GraphSAGE
- **Architecture**: Samples and aggregates from fixed-size neighborhoods
- **Community Detection**: Inductive learning for community detection on unseen nodes
- **Advantages**: Scalable to large graphs, inductive learning
- **Best for**: Large-scale graphs with millions of nodes

### Graph Isomorphism Network (GIN)
- **Architecture**: Uses injective aggregation functions for maximum discriminative power
- **Community Detection**: Structure-aware embeddings for community detection
- **Advantages**: Provably powerful, captures graph structure effectively
- **Best for**: Tasks requiring structural awareness

### Chebyshev Graph Convolution (ChebNet)
- **Architecture**: Uses Chebyshev polynomials for spectral graph convolution
- **Community Detection**: Spectral-based embeddings for community detection
- **Advantages**: Efficient spectral filtering, localized convolutions
- **Best for**: Graphs with clear spectral properties

### Simple Graph Convolution (SGC)
- **Architecture**: Simplified graph convolution with linear layers
- **Community Detection**: Fast and interpretable community detection
- **Advantages**: Fast training, interpretable, effective baseline
- **Best for**: Quick prototyping and baseline comparisons

## Traditional Algorithms

### Louvain Algorithm
- **Method**: Modularity optimization with hierarchical clustering
- **Advantages**: Fast, scalable, no predefined number of communities
- **Best for**: Large networks with unknown community structure

### Leiden Algorithm
- **Method**: Improved modularity optimization with better local optimization
- **Advantages**: Better quality communities, faster convergence
- **Best for**: High-quality community detection in large networks

### Spectral Clustering
- **Method**: Uses graph Laplacian eigenvectors for clustering
- **Advantages**: Theoretical guarantees, works well with clear community structure
- **Best for**: Networks with well-defined community boundaries

### K-means Clustering
- **Method**: Clusters node embeddings using K-means
- **Advantages**: Simple, interpretable, works with any embeddings
- **Best for**: When node embeddings are already available

## Datasets

### Social Network Datasets
- **Karate Club**: 34 nodes, 2 communities (Zachary's karate club)
- **Football**: 115 nodes, 12 communities (American football teams)
- **Polbooks**: 105 nodes, 3 communities (Political books)
- **Dolphins**: 62 nodes, 2 communities (Dolphin social network)
- **Les Misérables**: 77 nodes, 11 communities (Character co-appearances)

### Academic Networks
- **Cora**: 2,708 papers with 5,429 citations, 7 communities
- **Citeseer**: 3,327 papers with 4,732 citations, 6 communities
- **PubMed**: 19,717 papers with 44,338 citations, 3 communities

### Social Media Networks
- **Reddit**: Large-scale social network with user interactions
- **Flickr**: Photo-sharing network with user relationships

### E-commerce Networks
- **Amazon Photo**: Product co-purchase network
- **Amazon Computers**: Computer product relationships

## Usage

### Basic Usage

```python
from community_detection.community_detection import run_community_detection_experiment

# Default configuration
config = {
    'model_name': 'gcn',
    'dataset_name': 'karate',
    'hidden_channels': 64,
    'num_communities': 2,
    'learning_rate': 0.01,
    'epochs': 200
}

# Run experiment
results = run_community_detection_experiment(config)
print("Community detection completed")
```

### Advanced Configuration

```python
config = {
    'model_name': 'gat',
    'dataset_name': 'football',
    'hidden_channels': 128,
    'num_layers': 3,
    'heads': 8,
    'dropout': 0.2,
    'num_communities': 12,
    'learning_rate': 0.005,
    'weight_decay': 1e-4,
    'epochs': 300,
    'patience': 50,
    'device': 'cuda'
}

results = run_community_detection_experiment(config)
```

### Using Traditional Algorithms

```python
from community_detection.community_detection import detect_communities

# Detect communities using different methods
methods = ['louvain', 'leiden', 'spectral', 'kmeans']

for method in methods:
    communities = detect_communities(model, data, device, method=method)
    print(f"{method}: {len(np.unique(communities))} communities")
```

### Custom Dataset

```python
import torch
from torch_geometric.data import Data

# Create custom graph data
num_nodes = 1000
num_features = 16
num_communities = 5

# Node features
x = torch.randn(num_nodes, num_features)

# Edge index (source, target pairs)
edge_index = torch.randint(0, num_nodes, (2, 2000))

# Create data object
data = Data(x=x, edge_index=edge_index)

# Use in experiment
config = {
    'model_name': 'gin',
    'custom_data': data,
    'hidden_channels': 64,
    'num_communities': num_communities
}
```

## Parameters

### Architecture Parameters
- **hidden_channels**: Dimension of hidden layers (16-512)
- **num_layers**: Number of GNN layers (1-10)
- **dropout**: Dropout rate for regularization (0.0-0.5)
- **activation**: Activation function (relu, tanh, sigmoid, leaky_relu, elu)
- **num_communities**: Number of communities to detect (2-50)
- **batch_norm**: Whether to use batch normalization
- **residual**: Whether to use residual connections

### Community Detection Specific Parameters
- **detection_method**: Method for community detection (modularity, spectral, label_propagation, louvain, infomap)
- **resolution**: Resolution parameter for modularity optimization (0.1-10.0)
- **overlap**: Whether to allow overlapping communities
- **min_community_size**: Minimum size of detected communities (1-100)
- **max_community_size**: Maximum size of detected communities (10-1000)
- **use_edge_weights**: Whether to use edge weights in community detection

### Training Parameters
- **learning_rate**: Learning rate for optimization (1e-5 to 0.1)
- **epochs**: Number of training epochs (10-1000)
- **batch_size**: Batch size for training (1-256)
- **optimizer**: Optimization algorithm (adam, sgd, adamw, rmsprop, adagrad)
- **loss_function**: Loss function (modularity_loss, conductance_loss, normalized_cut_loss)
- **weight_decay**: L2 regularization coefficient (0.0-0.1)
- **scheduler**: Learning rate scheduler (none, step, cosine, plateau, exponential)
- **patience**: Early stopping patience (10-100)

## Evaluation Metrics

### Community Quality Metrics
- **Modularity**: Measures the quality of community structure (-1 to 1, higher is better)
- **Conductance**: Measures the quality of community boundaries (0 to 1, lower is better)
- **Normalized Cut**: Measures the quality of community separation (0 to 1, lower is better)
- **Silhouette Score**: Measures how well-separated communities are (-1 to 1, higher is better)

### Clustering Metrics (with ground truth)
- **Normalized Mutual Information (NMI)**: Measures similarity between predicted and true communities (0-1, higher is better)
- **Adjusted Rand Index (ARI)**: Measures clustering similarity (-1 to 1, higher is better)
- **Homogeneity**: Measures if communities contain only members of a single class (0-1, higher is better)
- **Completeness**: Measures if all members of a given class are assigned to the same community (0-1, higher is better)

### Community-Specific Metrics
- **Community Size Distribution**: Analysis of community size distribution
- **Community Overlap**: Analysis of overlapping communities
- **Community Stability**: Stability of communities across different runs
- **Community Cohesion**: Internal cohesion of detected communities

## Best Practices

### Model Selection
1. **Start with GCN**: Use GCN as a baseline for most tasks
2. **Try GAT for complex graphs**: Use GAT when neighbor importance varies
3. **Use GraphSAGE for large graphs**: Scale to millions of nodes
4. **Consider GIN for structural tasks**: When graph structure is crucial
5. **Use traditional algorithms**: Louvain and Leiden for quick results

### Algorithm Selection
1. **Louvain/Leiden**: For large networks with unknown community structure
2. **Spectral clustering**: For networks with clear community boundaries
3. **K-means on embeddings**: When node embeddings are available
4. **GNN models**: For learning-based community detection

### Hyperparameter Tuning
1. **Number of communities**: Start with domain knowledge, then tune
2. **Resolution parameter**: Higher values for smaller communities
3. **Learning rate**: Start with 0.01, adjust based on convergence
4. **Hidden dimensions**: 64-128 usually works well
5. **Dropout**: 0.1-0.3 for regularization

### Data Preparation
1. **Graph preprocessing**: Remove self-loops, make undirected if appropriate
2. **Feature normalization**: Normalize node features
3. **Edge weights**: Consider edge weights if available
4. **Community size constraints**: Set appropriate min/max community sizes

### Training Tips
1. **Early stopping**: Use validation modularity for early stopping
2. **Learning rate scheduling**: Reduce learning rate when plateauing
3. **Regularization**: Use dropout and weight decay to prevent overfitting
4. **Monitoring**: Track both training and validation metrics
5. **Multiple runs**: Run multiple times for stability assessment

## Examples

### Social Network Analysis
```python
# Detect friend groups in social network
config = {
    'model_name': 'gat',
    'dataset_name': 'karate',
    'hidden_channels': 128,
    'heads': 8,
    'num_communities': 2,
    'detection_method': 'modularity',
    'resolution': 1.0,
    'learning_rate': 0.01,
    'epochs': 200
}
```

### Biological Network Analysis
```python
# Detect protein complexes
config = {
    'model_name': 'gin',
    'dataset_name': 'ppi',
    'hidden_channels': 256,
    'num_layers': 4,
    'num_communities': 10,
    'detection_method': 'spectral',
    'min_community_size': 5,
    'max_community_size': 50
}
```

### Academic Network Analysis
```python
# Detect research communities
config = {
    'model_name': 'gcn',
    'dataset_name': 'cora',
    'hidden_channels': 64,
    'num_communities': 7,
    'detection_method': 'louvain',
    'resolution': 0.8,
    'learning_rate': 0.005,
    'weight_decay': 1e-4
}
```

### Large-Scale Network Analysis
```python
# Detect communities in large network
config = {
    'model_name': 'graphsage',
    'dataset_name': 'reddit',
    'hidden_channels': 128,
    'num_communities': 20,
    'detection_method': 'leiden',
    'resolution': 1.2,
    'batch_size': 64
}
```

## Troubleshooting

### Common Issues

**Poor Community Quality**
- Check resolution parameter for modularity-based methods
- Adjust number of communities
- Try different detection methods
- Increase model capacity

**Overfitting**
- Increase dropout rate
- Add weight decay
- Reduce model complexity
- Use early stopping
- Increase training data

**Slow Training**
- Reduce hidden dimensions
- Use fewer layers
- Try traditional algorithms for speed
- Use GPU acceleration
- Increase batch size

**Memory Issues**
- Reduce batch size
- Use smaller hidden dimensions
- Process graph in chunks
- Use sparse operations

### Performance Optimization
1. **GPU Usage**: Ensure CUDA is available and used
2. **Data Loading**: Use efficient data loaders
3. **Model Architecture**: Choose appropriate model complexity
4. **Algorithm Selection**: Use appropriate algorithm for your use case
5. **Parallel Processing**: Use multiple runs for stability

## Integration with Web Interface

The community detection functionality is fully integrated with the web interface:

1. **Parameter Configuration**: All parameters can be set through the web UI
2. **Model Selection**: Choose from available models via dropdown
3. **Algorithm Selection**: Choose between GNN models and traditional algorithms
4. **Dataset Selection**: Select from available datasets
5. **Real-time Training**: Monitor training progress with live updates
6. **Result Visualization**: View results and metrics in interactive charts
7. **Community Analysis**: Analyze detected communities and their properties
8. **Network Visualization**: Visualize communities in the network structure

## Advanced Features

### Multi-Level Community Detection
- Hierarchical community detection
- Community refinement
- Multi-scale analysis
- Community evolution tracking

### Overlapping Communities
- Fuzzy community membership
- Overlapping community detection
- Community hierarchy analysis
- Multi-membership analysis

### Dynamic Community Detection
- Temporal community evolution
- Community stability analysis
- Change detection
- Trend analysis

### Interpretable Community Detection
- Feature importance analysis
- Community explanation
- Decision path analysis
- Community characteristics

## Future Enhancements

- **Temporal community detection**: Handle time-evolving networks
- **Heterogeneous community detection**: Support for different node and edge types
- **Few-shot community detection**: Handle scenarios with limited labeled communities
- **Adversarial community detection**: Robust against adversarial attacks
- **Federated community detection**: Privacy-preserving distributed learning
- **Causal community detection**: Understand causal relationships in communities
- **Explainable community detection**: Provide interpretable community assignments 