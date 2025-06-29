# Anomaly Detection

## Overview

Anomaly Detection is a critical graph analysis task that involves identifying unusual patterns, nodes, edges, or subgraphs that deviate from normal behavior in the network. These anomalies can represent fraud, intrusions, defects, or other significant events that require attention.

## Purpose

Anomaly detection aims to:
- **Identify unusual patterns**: Detect nodes, edges, or subgraphs that behave differently from the norm
- **Prevent fraud and attacks**: Identify malicious activities in financial and security networks
- **Quality control**: Detect defects or anomalies in manufacturing and production systems
- **Network security**: Identify intrusions and suspicious activities in computer networks
- **Health monitoring**: Detect disease outbreaks or unusual health patterns
- **Risk assessment**: Identify high-risk entities or connections

## Applications

### Financial Networks
- **Fraud detection**: Identify fraudulent transactions and accounts
- **Money laundering**: Detect suspicious money flow patterns
- **Credit card fraud**: Identify unusual spending patterns
- **Insurance fraud**: Detect fraudulent claims and activities
- **Trading anomalies**: Identify unusual trading patterns

### Cybersecurity
- **Intrusion detection**: Identify network intrusions and attacks
- **Malware detection**: Detect malicious software and activities
- **DDoS attacks**: Identify distributed denial-of-service attacks
- **Data breaches**: Detect unauthorized access and data leaks
- **Bot detection**: Identify automated malicious activities

### Social Networks
- **Fake accounts**: Detect fake or bot accounts
- **Spam detection**: Identify spam messages and content
- **Troll detection**: Detect malicious users and trolls
- **Information manipulation**: Identify coordinated disinformation campaigns
- **Account takeover**: Detect compromised accounts

### Healthcare
- **Disease outbreaks**: Detect unusual disease patterns
- **Medical fraud**: Identify fraudulent medical claims
- **Drug interactions**: Detect unusual drug interaction patterns
- **Patient monitoring**: Identify unusual patient behaviors
- **Healthcare fraud**: Detect billing and insurance fraud

### Manufacturing
- **Quality control**: Detect defective products
- **Equipment failure**: Identify equipment malfunctions
- **Process anomalies**: Detect unusual manufacturing processes
- **Supply chain issues**: Identify supply chain disruptions
- **Safety incidents**: Detect safety violations and incidents

### Transportation
- **Traffic anomalies**: Detect unusual traffic patterns
- **Vehicle malfunctions**: Identify vehicle issues
- **Route anomalies**: Detect unusual routing patterns
- **Accident prediction**: Identify high-risk situations
- **Infrastructure issues**: Detect infrastructure problems

## Models

### Graph Convolutional Network (GCN)
- **Architecture**: Uses graph convolutions to learn node representations
- **Anomaly Detection**: Learns normal patterns and identifies deviations
- **Advantages**: Simple, effective, computationally efficient
- **Best for**: Homogeneous graphs with clear normal patterns

### Graph Attention Network (GAT)
- **Architecture**: Uses attention mechanisms to weight neighbor contributions
- **Anomaly Detection**: Attention-weighted embeddings for anomaly detection
- **Advantages**: Adaptive feature aggregation, handles heterogeneous graphs
- **Best for**: Complex graphs with varying importance of neighbors

### GraphSAGE
- **Architecture**: Samples and aggregates from fixed-size neighborhoods
- **Anomaly Detection**: Inductive learning for anomaly detection on unseen nodes
- **Advantages**: Scalable to large graphs, inductive learning
- **Best for**: Large-scale graphs with millions of nodes

### Graph Isomorphism Network (GIN)
- **Architecture**: Uses injective aggregation functions for maximum discriminative power
- **Anomaly Detection**: Structure-aware embeddings for anomaly detection
- **Advantages**: Provably powerful, captures graph structure effectively
- **Best for**: Tasks requiring structural awareness

### Chebyshev Graph Convolution (ChebNet)
- **Architecture**: Uses Chebyshev polynomials for spectral graph convolution
- **Anomaly Detection**: Spectral-based embeddings for anomaly detection
- **Advantages**: Efficient spectral filtering, localized convolutions
- **Best for**: Graphs with clear spectral properties

### Simple Graph Convolution (SGC)
- **Architecture**: Simplified graph convolution with linear layers
- **Anomaly Detection**: Fast and interpretable anomaly detection
- **Advantages**: Fast training, interpretable, effective baseline
- **Best for**: Quick prototyping and baseline comparisons

### Graph Autoencoder
- **Architecture**: Encoder-decoder architecture for reconstruction
- **Anomaly Detection**: Uses reconstruction error to identify anomalies
- **Advantages**: Unsupervised learning, no need for labeled anomalies
- **Best for**: Scenarios with limited labeled anomaly data

## Traditional Algorithms

### Isolation Forest
- **Method**: Uses isolation to identify anomalies
- **Advantages**: Fast, scalable, works well with high-dimensional data
- **Best for**: Large datasets with many features

### Local Outlier Factor (LOF)
- **Method**: Uses local density to identify outliers
- **Advantages**: Handles clusters of different densities, interpretable
- **Best for**: Data with varying local densities

### One-Class SVM
- **Method**: Learns a boundary around normal data
- **Advantages**: Robust, works well with high-dimensional data
- **Best for**: Well-defined normal regions

### Statistical Methods
- **Method**: Uses statistical measures (Z-score, Mahalanobis distance)
- **Advantages**: Simple, interpretable, fast
- **Best for**: Normally distributed data

## Datasets

### Financial Datasets
- **Credit Card Fraud**: Credit card transaction data with fraud labels
- **Banking Transactions**: Banking transaction networks
- **Insurance Claims**: Insurance claim networks with fraud labels
- **Trading Networks**: Financial trading networks

### Cybersecurity Datasets
- **Network Intrusions**: Network traffic with intrusion labels
- **Malware Networks**: Malware interaction networks
- **DDoS Attacks**: Network traffic during DDoS attacks
- **Bot Networks**: Social media bot networks

### Social Network Datasets
- **Fake News**: Social media networks with fake news labels
- **Spam Networks**: Email or social media spam networks
- **Troll Networks**: Social media troll detection datasets
- **Fake Accounts**: Social media fake account datasets

### Academic Networks
- **Cora**: Citation network with potential anomalies
- **Citeseer**: Citation network with unusual patterns
- **PubMed**: Biomedical literature network
- **Coauthor Networks**: Research collaboration networks

### Synthetic Datasets
- **Synthetic Anomalies**: Generated datasets with controlled anomalies
- **Benchmark Datasets**: Standard anomaly detection benchmarks
- **Simulated Networks**: Simulated networks with known anomalies

## Usage

### Basic Usage

```python
from anomaly_detection.anomaly_detection import run_anomaly_detection_experiment

# Default configuration
config = {
    'model_name': 'gcn',
    'dataset_name': 'synthetic',
    'hidden_channels': 64,
    'learning_rate': 0.01,
    'epochs': 200,
    'model_type': 'supervised'
}

# Run experiment
results = run_anomaly_detection_experiment(config)
print("Anomaly detection completed")
```

### Advanced Configuration

```python
config = {
    'model_name': 'autoencoder',
    'dataset_name': 'cora',
    'hidden_channels': 128,
    'num_layers': 3,
    'dropout': 0.2,
    'learning_rate': 0.005,
    'weight_decay': 1e-4,
    'epochs': 300,
    'patience': 50,
    'device': 'cuda',
    'model_type': 'unsupervised',
    'anomaly_ratio': 0.05,
    'anomaly_type': 'node'
}

results = run_anomaly_detection_experiment(config)
```

### Using Traditional Algorithms

```python
from anomaly_detection.anomaly_detection import detect_anomalies

# Detect anomalies using different methods
methods = ['isolation_forest', 'lof', 'one_class_svm', 'statistical', 'mahalanobis']

for method in methods:
    anomalies = detect_anomalies(model, data, device, method=method)
    print(f"{method}: {anomalies.sum()} anomalies detected")
```

### Custom Dataset

```python
import torch
from torch_geometric.data import Data

# Create custom graph data with anomalies
num_nodes = 1000
num_features = 16
anomaly_ratio = 0.05

# Node features
x = torch.randn(num_nodes, num_features)

# Add anomalies
num_anomalies = int(num_nodes * anomaly_ratio)
anomaly_indices = torch.randperm(num_nodes)[:num_anomalies]
x[anomaly_indices] += torch.randn(num_anomalies, num_features) * 3.0

# Edge index
edge_index = torch.randint(0, num_nodes, (2, 2000))

# Create data object
data = Data(x=x, edge_index=edge_index)

# Use in experiment
config = {
    'model_name': 'gat',
    'custom_data': data,
    'hidden_channels': 64,
    'model_type': 'supervised'
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

### Anomaly Detection Specific Parameters
- **detection_type**: Type of anomaly to detect (node, edge, subgraph, graph)
- **detection_method**: Method for anomaly detection (autoencoder, one_class_svm, isolation_forest, local_outlier_factor, reconstruction_error)
- **anomaly_threshold**: Threshold for anomaly detection (0.5-0.99)
- **contamination**: Expected proportion of anomalies (0.01-0.5)
- **use_reconstruction_error**: Whether to use reconstruction error
- **use_structural_features**: Whether to use structural features
- **embedding_dim**: Dimension of node embeddings (8-256)

### Training Parameters
- **learning_rate**: Learning rate for optimization (1e-5 to 0.1)
- **epochs**: Number of training epochs (10-1000)
- **batch_size**: Batch size for training (1-256)
- **optimizer**: Optimization algorithm (adam, sgd, adamw, rmsprop, adagrad)
- **loss_function**: Loss function (reconstruction_loss, contrastive_loss, margin_loss, one_class_loss)
- **weight_decay**: L2 regularization coefficient (0.0-0.1)
- **scheduler**: Learning rate scheduler (none, step, cosine, plateau, exponential)
- **patience**: Early stopping patience (10-100)

## Evaluation Metrics

### Classification Metrics (with ground truth)
- **AUC (Area Under ROC Curve)**: Measures overall performance (0-1, higher is better)
- **Average Precision (AP)**: Measures precision-recall performance (0-1, higher is better)
- **Precision**: Proportion of detected anomalies that are true anomalies (0-1, higher is better)
- **Recall**: Proportion of true anomalies that are detected (0-1, higher is better)
- **F1-Score**: Harmonic mean of precision and recall (0-1, higher is better)

### Anomaly-Specific Metrics
- **Anomaly Score Distribution**: Analysis of anomaly score distribution
- **Threshold Analysis**: Performance at different thresholds
- **False Positive Rate**: Rate of false alarms
- **False Negative Rate**: Rate of missed anomalies
- **Detection Delay**: Time to detect anomalies

### Statistical Metrics
- **Statistical Significance**: Statistical significance of detected anomalies
- **Confidence Intervals**: Confidence intervals for anomaly scores
- **Outlier Analysis**: Analysis of outlier characteristics
- **Distribution Analysis**: Analysis of score distributions

## Best Practices

### Model Selection
1. **Start with GCN**: Use GCN as a baseline for most tasks
2. **Try Autoencoder for unsupervised**: Use autoencoder when labeled anomalies are scarce
3. **Use GAT for complex graphs**: Use GAT when neighbor importance varies
4. **Consider traditional algorithms**: Use traditional algorithms for quick results
5. **Combine multiple methods**: Ensemble multiple methods for better performance

### Algorithm Selection
1. **Isolation Forest**: For large datasets with many features
2. **Local Outlier Factor**: For data with varying local densities
3. **One-Class SVM**: For well-defined normal regions
4. **Statistical methods**: For normally distributed data
5. **GNN models**: For learning-based anomaly detection

### Hyperparameter Tuning
1. **Anomaly threshold**: Start with 0.95, adjust based on domain knowledge
2. **Contamination**: Estimate based on domain knowledge, then tune
3. **Learning rate**: Start with 0.01, adjust based on convergence
4. **Hidden dimensions**: 64-128 usually works well
5. **Dropout**: 0.1-0.3 for regularization

### Data Preparation
1. **Feature normalization**: Normalize node features
2. **Graph preprocessing**: Remove self-loops, make undirected if appropriate
3. **Anomaly injection**: Inject synthetic anomalies for testing
4. **Data balancing**: Handle class imbalance if present
5. **Feature engineering**: Create relevant features for anomaly detection

### Training Tips
1. **Early stopping**: Use validation loss for early stopping
2. **Learning rate scheduling**: Reduce learning rate when plateauing
3. **Regularization**: Use dropout and weight decay to prevent overfitting
4. **Monitoring**: Track both training and validation metrics
5. **Multiple runs**: Run multiple times for stability assessment

## Examples

### Financial Fraud Detection
```python
# Detect fraud in financial network
config = {
    'model_name': 'gat',
    'dataset_name': 'banking_transactions',
    'hidden_channels': 128,
    'heads': 8,
    'detection_type': 'node',
    'detection_method': 'autoencoder',
    'anomaly_threshold': 0.95,
    'contamination': 0.01,
    'learning_rate': 0.01,
    'epochs': 200
}
```

### Cybersecurity Intrusion Detection
```python
# Detect network intrusions
config = {
    'model_name': 'gin',
    'dataset_name': 'network_traffic',
    'hidden_channels': 256,
    'num_layers': 4,
    'detection_type': 'edge',
    'detection_method': 'reconstruction_error',
    'use_structural_features': True,
    'learning_rate': 0.005,
    'weight_decay': 1e-4
}
```

### Social Network Fake Account Detection
```python
# Detect fake accounts in social network
config = {
    'model_name': 'gcn',
    'dataset_name': 'social_network',
    'hidden_channels': 64,
    'detection_type': 'node',
    'detection_method': 'one_class_svm',
    'anomaly_threshold': 0.9,
    'contamination': 0.05,
    'learning_rate': 0.01,
    'epochs': 100
}
```

### Manufacturing Quality Control
```python
# Detect defects in manufacturing process
config = {
    'model_name': 'autoencoder',
    'dataset_name': 'manufacturing_data',
    'hidden_channels': 128,
    'detection_type': 'subgraph',
    'detection_method': 'reconstruction_error',
    'use_reconstruction_error': True,
    'embedding_dim': 64,
    'learning_rate': 0.001,
    'epochs': 300
}
```

## Troubleshooting

### Common Issues

**High False Positive Rate**
- Increase anomaly threshold
- Use more training data
- Try different detection methods
- Adjust contamination parameter
- Use ensemble methods

**High False Negative Rate**
- Decrease anomaly threshold
- Use more sensitive detection methods
- Increase model capacity
- Add more features
- Use domain-specific features

**Poor Performance**
- Check data quality and preprocessing
- Try different model architectures
- Adjust hyperparameters
- Use ensemble methods
- Consider domain-specific features

**Slow Training**
- Reduce model complexity
- Use smaller hidden dimensions
- Try traditional algorithms for speed
- Use GPU acceleration
- Increase batch size

**Memory Issues**
- Reduce batch size
- Use smaller hidden dimensions
- Process graph in chunks
- Use sparse operations
- Use traditional algorithms

### Performance Optimization
1. **GPU Usage**: Ensure CUDA is available and used
2. **Data Loading**: Use efficient data loaders
3. **Model Architecture**: Choose appropriate model complexity
4. **Algorithm Selection**: Use appropriate algorithm for your use case
5. **Parallel Processing**: Use multiple runs for stability

## Integration with Web Interface

The anomaly detection functionality is fully integrated with the web interface:

1. **Parameter Configuration**: All parameters can be set through the web UI
2. **Model Selection**: Choose from available models via dropdown
3. **Algorithm Selection**: Choose between GNN models and traditional algorithms
4. **Dataset Selection**: Select from available datasets
5. **Real-time Training**: Monitor training progress with live updates
6. **Result Visualization**: View results and metrics in interactive charts
7. **Anomaly Analysis**: Analyze detected anomalies and their characteristics
8. **Network Visualization**: Visualize anomalies in the network structure

## Advanced Features

### Multi-Type Anomaly Detection
- Node-level anomalies
- Edge-level anomalies
- Subgraph-level anomalies
- Graph-level anomalies
- Temporal anomalies

### Ensemble Methods
- Multiple model ensemble
- Different algorithm combination
- Voting mechanisms
- Weighted ensemble
- Stacking methods

### Real-time Anomaly Detection
- Streaming data processing
- Incremental learning
- Online anomaly detection
- Real-time alerts
- Dynamic threshold adjustment

### Interpretable Anomaly Detection
- Feature importance analysis
- Anomaly explanation
- Decision path analysis
- Anomaly characteristics
- Root cause analysis

## Future Enhancements

- **Temporal anomaly detection**: Handle time-evolving networks
- **Heterogeneous anomaly detection**: Support for different node and edge types
- **Few-shot anomaly detection**: Handle scenarios with limited labeled anomalies
- **Adversarial anomaly detection**: Robust against adversarial attacks
- **Federated anomaly detection**: Privacy-preserving distributed learning
- **Causal anomaly detection**: Understand causal relationships in anomalies
- **Explainable anomaly detection**: Provide interpretable anomaly explanations 