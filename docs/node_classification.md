# Node Classification Backend

## Overview

The node classification backend (`node_tasks/classification/nodes_classification.py`) implements multi-class classification for nodes in graphs. It predicts discrete class labels for each node based on node features and graph structure.

## Supported Models

| Model | Description | Paper | Architecture |
|-------|-------------|-------|--------------|
| **GCN** | Graph Convolutional Network | Kipf & Welling, ICLR 2017 | Spectral |
| **GAT** | Graph Attention Network | Veličković et al., ICLR 2018 | Attention |
| **GraphSAGE** | Inductive representation learning | Hamilton et al., NeurIPS 2017 | Spatial |
| **GIN** | Graph Isomorphism Network | Xu et al., ICLR 2019 | Spatial |
| **ChebNet** | Chebyshev Graph Convolution | Defferrard et al., NeurIPS 2016 | Spectral |
| **SGC** | Simple Graph Convolution | Wu et al., ICML 2019 | Spectral |

## Supported Datasets

### Real Datasets
- **Citation Networks**: Cora, CiteSeer, PubMed
- **E-commerce**: Amazon Photo, Amazon Computers
- **Co-authorship**: Coauthor CS, Coauthor Physics
- **Social Networks**: Reddit, Flickr

### Dataset Information

| Dataset | Nodes | Edges | Features | Classes | Type |
|---------|-------|-------|----------|---------|------|
| Cora | 2,708 | 5,429 | 1,433 | 7 | Citation |
| CiteSeer | 3,327 | 4,732 | 3,703 | 6 | Citation |
| PubMed | 19,717 | 44,338 | 500 | 3 | Citation |
| Amazon Photo | 7,650 | 119,081 | 745 | 8 | E-commerce |
| Amazon Computers | 13,752 | 245,861 | 767 | 10 | E-commerce |
| Coauthor CS | 18,333 | 81,894 | 6,805 | 15 | Co-authorship |
| Coauthor Physics | 34,493 | 247,962 | 8,415 | 5 | Co-authorship |
| Reddit | 232,965 | 11,606,919 | 602 | 41 | Social |
| Flickr | 89,250 | 899,756 | 500 | 7 | Social |

## Key Functions

### Main Experiment Runner
```python
def run_node_classification_experiment(config):
    """
    Run a complete node classification experiment
    
    Args:
        config (dict): Experiment configuration
        
    Returns:
        dict: Experiment results including metrics and training history
    """
```

### Model Factory
```python
def get_model(model_name, in_channels, hidden_channels, out_channels, **kwargs):
    """
    Get model instance by name
    
    Args:
        model_name (str): Name of the model architecture
        in_channels (int): Number of input features
        hidden_channels (int): Number of hidden dimensions
        out_channels (int): Number of output classes
        **kwargs: Additional model-specific parameters
        
    Returns:
        torch.nn.Module: Model instance
    """
```

### Training Function
```python
def train_node_classifier(model, data, train_mask, val_mask, device, 
                         learning_rate=0.01, weight_decay=5e-4, epochs=200, 
                         patience=50, save_path=None):
    """
    Train a node classification model
    
    Args:
        model: GNN model
        data: PyTorch Geometric Data object
        train_mask: Boolean mask for training nodes
        val_mask: Boolean mask for validation nodes
        device: Device to train on
        learning_rate: Learning rate
        weight_decay: Weight decay
        epochs: Maximum epochs
        patience: Early stopping patience
        save_path: Path to save best model
        
    Returns:
        dict: Training results
    """
```

### Evaluation Function
```python
def evaluate_node_classifier(model, data, test_mask, device):
    """
    Evaluate a trained node classification model
    
    Args:
        model: Trained GNN model
        data: PyTorch Geometric Data object
        test_mask: Boolean mask for test nodes
        device: Device to evaluate on
        
    Returns:
        dict: Evaluation metrics
    """
```

### Prediction Function
```python
def predict_node_labels(model, data, device, mask=None):
    """
    Predict node labels
    
    Args:
        model: Trained GNN model
        data: PyTorch Geometric Data object
        device: Device to predict on
        mask: Optional mask for specific nodes
        
    Returns:
        dict: Predictions and probabilities
    """
```

## Metrics

The node classification backend provides comprehensive evaluation metrics:

- **Accuracy**: Overall classification accuracy
- **F1 Score**: Macro and micro F1 scores
- **Precision**: Macro precision
- **Recall**: Macro recall
- **Confusion Matrix**: Detailed classification results

## Usage Examples

### Basic Usage
```python
from node_tasks.classification.nodes_classification import run_node_classification_experiment, get_default_config

# Get default configuration
config = get_default_config()

# Run experiment
results = run_node_classification_experiment(config)

# Access results
print(f"Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
print(f"Test F1 Score: {results['test_metrics']['f1_macro']:.4f}")
```

### Custom Configuration
```python
config = {
    'model_name': 'gat',
    'dataset_name': 'cora',
    'hidden_channels': 128,
    'learning_rate': 0.005,
    'weight_decay': 1e-4,
    'epochs': 300,
    'patience': 100,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

results = run_node_classification_experiment(config)
```

### Advanced Usage
```python
from node_tasks.classification.nodes_classification import (
    get_model, load_dataset, create_train_val_test_split,
    train_node_classifier, evaluate_node_classifier
)

# Load dataset
data = load_dataset('pubmed')

# Create splits
train_mask, val_mask, test_mask = create_train_val_test_split(data)

# Get model
model = get_model('gin', data.num_node_features, 256, data.y.max().item() + 1)

# Train model
training_result = train_node_classifier(
    model, data, train_mask, val_mask, device='cuda',
    learning_rate=0.001, epochs=500, patience=100
)

# Evaluate model
test_metrics = evaluate_node_classifier(model, data, test_mask, device='cuda')
```

## Configuration

### Default Configuration
```python
{
    'model_name': 'gcn',
    'dataset_name': 'cora',
    'hidden_channels': 64,
    'learning_rate': 0.01,
    'weight_decay': 5e-4,
    'epochs': 200,
    'patience': 50,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}
```

### Configuration Parameters

| Parameter | Type | Description | Default | Range |
|-----------|------|-------------|---------|-------|
| `model_name` | str | GNN model architecture | 'gcn' | ['gcn', 'gat', 'graphsage', 'gin', 'chebnet', 'sgc'] |
| `dataset_name` | str | Dataset to use | 'cora' | See supported datasets |
| `hidden_channels` | int | Hidden layer dimensions | 64 | [16, 512] |
| `learning_rate` | float | Learning rate | 0.01 | [0.0001, 0.1] |
| `weight_decay` | float | L2 regularization | 5e-4 | [0, 0.1] |
| `epochs` | int | Maximum training epochs | 200 | [10, 1000] |
| `patience` | int | Early stopping patience | 50 | [10, 200] |
| `device` | str | Device to use | 'cuda' | ['cuda', 'cpu'] |

## Model-Specific Parameters

### GAT Parameters
```python
config = {
    'model_name': 'gat',
    'heads': 8,              # Number of attention heads
    'attention_dropout': 0.1  # Attention dropout rate
}
```

### GIN Parameters
```python
config = {
    'model_name': 'gin',
    'epsilon': 0.0,          # Epsilon for GIN
    'train_eps': True        # Whether to train epsilon
}
```

### ChebNet Parameters
```python
config = {
    'model_name': 'chebnet',
    'K': 3                   # Chebyshev polynomial order
}
```

### SGC Parameters
```python
config = {
    'model_name': 'sgc',
    'K': 2                   # Number of hops
}
```

## Output Structure

### Experiment Results
```
results/
└── node_classification/
    └── gcn_cora_20241201_143022/
        ├── config.json          # Experiment configuration
        ├── results.json         # Training and test results
        └── best_model.pt        # Saved model checkpoint
```

### Results JSON Structure
```json
{
    "experiment_name": "gcn_cora_20241201_143022",
    "config": {
        "model_name": "gcn",
        "dataset_name": "cora",
        "hidden_channels": 64,
        "learning_rate": 0.01,
        "weight_decay": 5e-4,
        "epochs": 200,
        "patience": 50,
        "device": "cuda"
    },
    "training_result": {
        "best_val_acc": 0.8234,
        "best_epoch": 156,
        "train_losses": [2.1, 1.8, 1.5, ...],
        "val_accuracies": [0.65, 0.72, 0.78, ...]
    },
    "test_metrics": {
        "accuracy": 0.8156,
        "f1_macro": 0.8123,
        "f1_micro": 0.8156,
        "precision": 0.8145,
        "recall": 0.8156,
        "confusion_matrix": [[...], [...], ...]
    },
    "timestamp": "20241201_143022"
}
```

## Performance Benchmarks

### Typical Performance on Standard Datasets

| Model | Cora | CiteSeer | PubMed | Amazon Photo | Amazon Computers |
|-------|------|----------|--------|--------------|------------------|
| GCN | 81.5% | 70.3% | 79.0% | 92.4% | 86.5% |
| GAT | 83.0% | 72.5% | 79.0% | 93.8% | 87.3% |
| GraphSAGE | 82.2% | 71.4% | 78.6% | 92.7% | 86.8% |
| GIN | 82.7% | 71.8% | 78.8% | 93.1% | 87.1% |
| ChebNet | 81.2% | 69.8% | 78.1% | 91.9% | 86.2% |
| SGC | 81.0% | 71.9% | 78.9% | 92.1% | 86.4% |

*Note: Performance may vary based on hyperparameters and random seeds*

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```python
# Reduce model size
config['hidden_channels'] = 32  # Reduce from 64

# Use smaller dataset
config['dataset_name'] = 'cora'  # Instead of larger datasets
```

#### 2. Training Not Converging
```python
# Reduce learning rate
config['learning_rate'] = 0.001  # Reduce from 0.01

# Increase patience
config['patience'] = 100  # Increase from 50

# Use different model
config['model_name'] = 'gat'  # Try attention-based model
```

#### 3. Poor Performance
```python
# Try different architectures
config['model_name'] = 'gin'  # GIN often performs well

# Increase model capacity
config['hidden_channels'] = 128  # Increase from 64

# Adjust regularization
config['weight_decay'] = 1e-4  # Reduce from 5e-4
```

### Performance Tips

1. **Model Selection**: GAT and GIN typically perform well across datasets
2. **Hyperparameter Tuning**: Learning rate and hidden dimensions are most important
3. **Early Stopping**: Monitor validation accuracy to prevent overfitting
4. **Feature Normalization**: Datasets are automatically normalized
5. **GPU Usage**: Use CUDA for significantly faster training

## API Integration

### FastAPI Route Example
```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from node_tasks.classification.nodes_classification import run_node_classification_experiment

router = APIRouter()

class ClassificationRequest(BaseModel):
    model_name: str
    dataset_name: str
    hidden_channels: int = 64
    learning_rate: float = 0.01
    epochs: int = 200

@router.post("/train")
async def train_classification_model(request: ClassificationRequest):
    try:
        config = request.dict()
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        results = run_node_classification_experiment(config)
        
        return {
            'status': 'success',
            'experiment_name': results['experiment_name'],
            'test_accuracy': results['test_metrics']['accuracy'],
            'test_f1': results['test_metrics']['f1_macro'],
            'training_time': results.get('training_time', 0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## Contributing

To extend the node classification backend:

1. **Add New Model**: Implement in `nodes_classification.py`
2. **Add to Model Factory**: Update `get_model()` function
3. **Add New Dataset**: Implement in `load_dataset()` function
4. **Update Documentation**: Add examples and benchmarks
5. **Add Tests**: Ensure compatibility and performance

## References

- Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. ICLR.
- Veličković, P., et al. (2018). Graph attention networks. ICLR.
- Hamilton, W., et al. (2017). Inductive representation learning on large graphs. NeurIPS.
- Xu, K., et al. (2019). How powerful are graph neural networks? ICLR.
- Defferrard, M., et al. (2016). Convolutional neural networks on graphs with fast localized spectral filtering. NeurIPS.
- Wu, F., et al. (2019). Simplifying graph convolutional networks. ICML. 