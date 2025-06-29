# Link Prediction

## Overview

Link Prediction is a fundamental graph machine learning task that involves predicting the existence of edges (links) between nodes in a graph. This task is crucial for understanding network evolution, discovering hidden relationships, and completing missing information in various types of networks.

## Purpose

Link prediction aims to:
- **Predict missing links**: Identify edges that exist but are not observed
- **Forecast future connections**: Predict edges that will form in the future
- **Discover hidden relationships**: Find latent connections between entities
- **Complete networks**: Fill gaps in incomplete graph data
- **Understand network dynamics**: Model how networks evolve over time

## Applications

### Social Networks
- **Friend recommendations**: Suggest potential friends based on common connections
- **Relationship prediction**: Predict romantic or professional relationships
- **Influence prediction**: Identify who will influence whom
- **Community formation**: Predict group membership and community evolution

### Biological Networks
- **Protein-protein interactions**: Predict unknown protein interactions
- **Drug-target interactions**: Identify potential drug targets
- **Gene regulatory networks**: Predict regulatory relationships
- **Metabolic networks**: Predict metabolic pathway connections

### Knowledge Graphs
- **Entity linking**: Connect related entities across different sources
- **Fact completion**: Predict missing facts in knowledge bases
- **Relationship inference**: Infer relationships between entities
- **Knowledge graph expansion**: Add new edges to existing knowledge graphs

### Recommendation Systems
- **Product recommendations**: Suggest products based on user behavior
- **Collaborative filtering**: Predict user-item interactions
- **Content recommendation**: Suggest content based on user preferences
- **Network-based recommendations**: Use graph structure for recommendations

### Academic Networks
- **Collaboration prediction**: Predict future research collaborations
- **Citation prediction**: Predict which papers will cite others
- **Co-authorship networks**: Predict co-author relationships
- **Research interest matching**: Connect researchers with similar interests

## Models

### Graph Convolutional Network (GCN)
- **Architecture**: Uses graph convolutions to learn node representations
- **Link Prediction**: Concatenates node embeddings for edge prediction
- **Advantages**: Simple, effective, computationally efficient
- **Best for**: Homogeneous graphs with clear neighborhood patterns

### Graph Attention Network (GAT)
- **Architecture**: Uses attention mechanisms to weight neighbor contributions
- **Link Prediction**: Attention-weighted node embeddings for edge prediction
- **Advantages**: Adaptive feature aggregation, handles heterogeneous graphs
- **Best for**: Complex graphs with varying importance of neighbors

### GraphSAGE
- **Architecture**: Samples and aggregates from fixed-size neighborhoods
- **Link Prediction**: Inductive learning for link prediction on unseen nodes
- **Advantages**: Scalable to large graphs, inductive learning
- **Best for**: Large-scale graphs with millions of nodes

### Graph Isomorphism Network (GIN)
- **Architecture**: Uses injective aggregation functions for maximum discriminative power
- **Link Prediction**: Structure-aware node embeddings for link prediction
- **Advantages**: Provably powerful, captures graph structure effectively
- **Best for**: Tasks requiring structural awareness

### Chebyshev Graph Convolution (ChebNet)
- **Architecture**: Uses Chebyshev polynomials for spectral graph convolution
- **Link Prediction**: Spectral-based node representations for link prediction
- **Advantages**: Efficient spectral filtering, localized convolutions
- **Best for**: Graphs with clear spectral properties

### Simple Graph Convolution (SGC)
- **Architecture**: Simplified graph convolution with linear layers
- **Link Prediction**: Fast and interpretable link prediction
- **Advantages**: Fast training, interpretable, effective baseline
- **Best for**: Quick prototyping and baseline comparisons

## Datasets

### Citation Networks
- **Cora**: 2,708 papers with 5,429 citations
- **Citeseer**: 3,327 papers with 4,732 citations
- **PubMed**: 19,717 papers with 44,338 citations

### Social Networks
- **Reddit**: Large-scale social network with user interactions
- **Flickr**: Photo-sharing network with user relationships

### E-commerce Networks
- **Amazon Photo**: Product co-purchase network
- **Amazon Computers**: Computer product relationships

### Academic Networks
- **Coauthor CS**: Computer science collaboration network
- **Coauthor Physics**: Physics collaboration network

### Synthetic Datasets
- **Custom graphs**: Generated with specific properties for testing
- **Controlled experiments**: Graphs with known link patterns

## Usage

### Basic Usage

```python
from edge_tasks.link_prediction.link_prediction import run_link_prediction_experiment

# Default configuration
config = {
    'model_name': 'gcn',
    'dataset_name': 'cora',
    'hidden_channels': 64,
    'learning_rate': 0.01,
    'epochs': 200
}

# Run experiment
results = run_link_prediction_experiment(config)
print(f"Test AUC: {results['test_metrics']['auc']:.4f}")
print(f"Test Average Precision: {results['test_metrics']['average_precision']:.4f}")
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
    'device': 'cuda',
    'negative_sampling_ratio': 1.0
}

results = run_link_prediction_experiment(config)
```

### Custom Dataset

```python
import torch
from torch_geometric.data import Data

# Create custom graph data
num_nodes = 1000
num_features = 16

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
    'hidden_channels': 64
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

### Link Prediction Specific Parameters
- **link_prediction_method**: Method for link prediction (dot_product, cosine, euclidean, mlp, bilinear)
- **negative_sampling_ratio**: Ratio of negative samples to positive samples (0.1-10.0)
- **use_edge_features**: Whether to use edge features in prediction
- **edge_feature_dim**: Dimension of edge features (8-256)
- **prediction_threshold**: Threshold for link prediction (0.0-1.0)
- **use_structural_features**: Whether to use structural features (degree, clustering, etc.)

### Training Parameters
- **learning_rate**: Learning rate for optimization (1e-5 to 0.1)
- **epochs**: Number of training epochs (10-1000)
- **batch_size**: Batch size for training (1-256)
- **optimizer**: Optimization algorithm (adam, sgd, adamw, rmsprop, adagrad)
- **loss_function**: Loss function for link prediction (bce, focal_loss, margin_loss, contrastive_loss)
- **weight_decay**: L2 regularization coefficient (0.0-0.1)
- **patience**: Early stopping patience (10-100)

## Evaluation Metrics

### Link Prediction Metrics
- **AUC-ROC**: Area under the ROC curve (0.5-1.0, higher is better)
- **Average Precision**: Area under the precision-recall curve (0.0-1.0, higher is better)
- **Precision@K**: Precision at top-K predictions
- **Recall@K**: Recall at top-K predictions
- **F1 Score**: Harmonic mean of precision and recall
- **Hits@K**: Whether the true link is in top-K predictions

### Ranking Metrics
- **Mean Reciprocal Rank (MRR)**: Average of reciprocal ranks
- **Normalized Discounted Cumulative Gain (NDCG)**: Ranking quality measure
- **Mean Average Precision (MAP)**: Average precision across different queries

## Best Practices

### Model Selection
1. **Start with GCN**: Use GCN as a baseline for most tasks
2. **Try GAT for complex graphs**: Use GAT when neighbor importance varies
3. **Use GraphSAGE for large graphs**: Scale to millions of nodes
4. **Consider GIN for structural tasks**: When graph structure is crucial

### Negative Sampling
1. **Balanced sampling**: Use equal numbers of positive and negative samples
2. **Hard negative mining**: Focus on difficult negative examples
3. **Stratified sampling**: Ensure negative samples are representative
4. **Dynamic sampling**: Adjust negative sampling during training

### Hyperparameter Tuning
1. **Learning rate**: Start with 0.01, adjust based on convergence
2. **Hidden dimensions**: 64-128 usually works well
3. **Dropout**: 0.1-0.3 for regularization
4. **Negative sampling ratio**: 1.0-5.0 typically works well

### Data Preparation
1. **Graph preprocessing**: Remove self-loops, make undirected if appropriate
2. **Feature normalization**: Normalize node features
3. **Edge splitting**: Use proper train/validation/test splits
4. **Temporal splitting**: For temporal graphs, split by time

### Training Tips
1. **Early stopping**: Use validation AUC for early stopping
2. **Learning rate scheduling**: Reduce learning rate when plateauing
3. **Regularization**: Use dropout and weight decay to prevent overfitting
4. **Monitoring**: Track both training and validation metrics

## Examples

### Social Network Link Prediction
```python
# Predict friendship links in a social network
config = {
    'model_name': 'gat',
    'dataset_name': 'synthetic_social',
    'hidden_channels': 128,
    'heads': 8,
    'negative_sampling_ratio': 1.0,
    'learning_rate': 0.01,
    'epochs': 200
}
```

### Biological Network Link Prediction
```python
# Predict protein-protein interactions
config = {
    'model_name': 'gin',
    'dataset_name': 'protein_interaction',
    'hidden_channels': 256,
    'num_layers': 4,
    'use_edge_features': True,
    'edge_feature_dim': 64,
    'link_prediction_method': 'mlp'
}
```

### Knowledge Graph Completion
```python
# Complete missing facts in knowledge graph
config = {
    'model_name': 'gcn',
    'dataset_name': 'knowledge_graph',
    'hidden_channels': 64,
    'link_prediction_method': 'bilinear',
    'learning_rate': 0.005,
    'weight_decay': 1e-4
}
```

### Recommendation System
```python
# Predict user-item interactions
config = {
    'model_name': 'graphsage',
    'dataset_name': 'user_item_graph',
    'hidden_channels': 128,
    'negative_sampling_ratio': 5.0,
    'prediction_threshold': 0.5,
    'use_structural_features': True
}
```

## Troubleshooting

### Common Issues

**Low AUC/AP Scores**
- Check negative sampling strategy
- Increase model capacity (more layers/hidden dimensions)
- Try different link prediction methods
- Adjust learning rate and training time

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
5. **Negative Sampling**: Optimize negative sampling strategy

## Integration with Web Interface

The link prediction functionality is fully integrated with the web interface:

1. **Parameter Configuration**: All parameters can be set through the web UI
2. **Model Selection**: Choose from available models via dropdown
3. **Dataset Selection**: Select from available datasets
4. **Real-time Training**: Monitor training progress with live updates
5. **Result Visualization**: View results and metrics in interactive charts
6. **Model Comparison**: Compare multiple models side-by-side
7. **Prediction Interface**: Make predictions on new data

## Advanced Features

### Multi-relational Link Prediction
- Support for different types of relationships
- Heterogeneous graph processing
- Relation-specific embeddings

### Temporal Link Prediction
- Time-aware link prediction
- Dynamic graph modeling
- Temporal evolution prediction

### Interpretable Link Prediction
- Attention weight visualization
- Feature importance analysis
- Decision explanation

### Active Learning
- Intelligent selection of edges to label
- Uncertainty-based sampling
- Human-in-the-loop learning

## Future Enhancements

- **Multi-modal link prediction**: Incorporate text, image, and graph data
- **Few-shot link prediction**: Handle scenarios with limited labeled edges
- **Adversarial link prediction**: Robust against adversarial attacks
- **Federated link prediction**: Privacy-preserving distributed learning
- **Causal link prediction**: Understand causal relationships in graphs
- **Explainable link prediction**: Provide interpretable predictions 