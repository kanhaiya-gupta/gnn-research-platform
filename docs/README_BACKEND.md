# Graph Neural Network Backend Documentation

This document provides an overview of the backend implementations for the Graph Neural Network (GNN) platform.

## Overview

The backend system provides complete implementations for training, evaluating, and deploying Graph Neural Networks across multiple tasks. Built on PyTorch Geometric, it supports various GNN architectures and datasets with a unified interface. All backend implementations are production-ready with comprehensive training, evaluation, and visualization capabilities.

## Architecture

```
graph-neural-network/
├── node_tasks/
│   ├── classification/
│   │   └── nodes_classification.py    # ✅ Node classification backend
│   └── regression/
│       └── nodes_regression.py        # ✅ Node regression backend
├── edge_tasks/
│   ├── classification/
│   │   └── edge_classification.py     # ✅ Edge classification backend
│   └── link_prediction/
│       └── link_prediction.py         # ✅ Link prediction backend
├── graph_tasks/
│   ├── classification/
│   │   └── graph_classification.py    # ✅ Graph classification backend
│   └── regression/
│       └── graph_regression.py        # ✅ Graph regression backend
├── community_detection/
│   └── community_detection.py         # ✅ Community detection backend
├── anomaly_detection/
│   └── anomaly_detection.py           # ✅ Anomaly detection backend
├── graph_generation/
│   └── graph_generation.py            # ✅ Graph generation backend
├── graph_embedding_visualization/
│   └── graph_embedding_visualization.py # ✅ Graph embedding visualization backend
├── dynamic_graph_learning/
│   └── dynamic_graph_learning.py      # ✅ Dynamic graph learning backend
├── utils/
│   ├── loggers.py                     # Logging utilities
│   ├── data_utils.py                  # Data loading/saving utilities
│   ├── model_utils.py                 # Model checkpointing utilities
│   └── train_utils.py                 # Training utilities
├── data/                              # Dataset storage
├── results/                           # Experiment results
└── logs/                              # Training logs
```

## Available Backends

### ✅ Node-Level Tasks
- **[Node Classification](node_classification.md)** - Multi-class classification for nodes
  - Models: GCN, GAT, GraphSAGE, GIN, ChebNet, SGC
  - Datasets: Cora, CiteSeer, PubMed, synthetic
  - Features: Training, evaluation, prediction, visualization

- **[Node Regression](node_regression.md)** - Continuous value prediction for nodes
  - Models: GCN, GAT, GraphSAGE, GIN, ChebNet, SGC
  - Datasets: Synthetic, custom regression datasets
  - Features: MSE/MAE metrics, regression analysis

### ✅ Edge-Level Tasks
- **[Link Prediction](link_prediction.md)** - Predicting missing edges
  - Models: GCN, GAT, GraphSAGE, GIN, ChebNet, SGC
  - Datasets: Cora, CiteSeer, PubMed, synthetic
  - Features: AUC/AP metrics, edge sampling, visualization

- **[Edge Classification](edge_classification.md)** - Classifying edge types
  - Models: GCN, GAT, GraphSAGE, GIN, ChebNet, SGC
  - Datasets: Synthetic, custom edge-labeled datasets
  - Features: Multi-class edge classification, edge feature support

### ✅ Graph-Level Tasks
- **[Graph Classification](graph_classification.md)** - Classifying entire graphs
  - Models: GCN, GAT, GraphSAGE, GIN, ChebNet, SGC
  - Datasets: MUTAG, PTC-MR, ENZYMES, synthetic
  - Features: Graph-level pooling, classification metrics

- **[Graph Regression](graph_regression.md)** - Predicting graph-level properties
  - Models: GCN, GAT, GraphSAGE, GIN, ChebNet, SGC
  - Datasets: ZINC, QM9, QM7, synthetic
  - Features: Regression metrics, property prediction

### ✅ Specialized Tasks
- **[Community Detection](community_detection.md)** - Detecting communities in graphs
  - Models: GCN, GAT, GraphSAGE, GIN, ChebNet, SGC
  - Traditional: Louvain, Leiden, Spectral Clustering, K-means
  - Datasets: Karate, Football, Polbooks, synthetic
  - Features: Modularity, NMI, ARI metrics, community visualization

- **[Anomaly Detection](anomaly_detection.md)** - Detecting anomalous nodes/edges
  - Models: GCN, GAT, GraphSAGE, GIN, ChebNet, SGC
  - Traditional: Isolation Forest, One-Class SVM, Local Outlier Factor
  - Datasets: Cora, CiteSeer, PubMed, synthetic
  - Features: Anomaly scoring, threshold optimization, visualization

- **[Graph Generation](graph_generation.md)** - Generating new graphs
  - Models: VGAE, GraphVAE, GraphRNN, GraphGAN
  - Datasets: ZINC, QM9, MUTAG, synthetic
  - Features: Graph generation, quality evaluation, diversity metrics

- **[Graph Embedding Visualization](graph_embedding_visualization.md)** - Learning and visualizing graph embeddings
  - Models: GCN, GAT, GraphSAGE, GIN, ChebNet, SGC
  - Datasets: Cora, CiteSeer, PubMed, Amazon, Coauthor, Reddit, Flickr, synthetic
  - Features: Dimensionality reduction (t-SNE, UMAP, PCA, MDS, Isomap), clustering (K-means, DBSCAN, Spectral), interactive visualizations, embedding analysis

- **[Dynamic Graph Learning](dynamic_graph_learning.md)** - Learning from evolving graphs
  - Models: TemporalGCN, TemporalGAT, TemporalGraphSAGE, TemporalGIN, RecurrentGNN, TemporalTransformer
  - Datasets: Enron, UC Irvine, Facebook, synthetic temporal graphs
  - Features: Temporal modeling, future prediction, evolution analysis

## Common Utilities

### Core Utilities (`utils/`)

#### Logging (`utils/loggers.py`)
```python
from utils.loggers import LoggerFactory

logger_factory = LoggerFactory("experiment_name", "logs", "experiment")
logger = logger_factory.get_logger()
```

#### Data Utilities (`utils/data_utils.py`)
```python
from utils.data_utils import load_json, save_json, ensure_dir

config = load_json("config.json")
save_json(results, "results.json")
ensure_dir("results/experiment")
```

#### Model Utilities (`utils/model_utils.py`)
```python
from utils.model_utils import save_checkpoint, load_checkpoint

save_checkpoint(state, "checkpoint.pt")
checkpoint = load_checkpoint("checkpoint.pt")
```

## Quick Start

### Basic Usage
```python
# Node Classification
from node_tasks.classification.nodes_classification import run_node_classification_experiment, get_default_config

config = get_default_config()
config['model_name'] = 'gat'
config['dataset_name'] = 'cora'
results = run_node_classification_experiment(config)

# Node Regression
from node_tasks.regression.nodes_regression import run_node_regression_experiment, get_default_config

config = get_default_config()
config['model_name'] = 'gcn'
config['dataset_name'] = 'synthetic'
results = run_node_regression_experiment(config)

# Link Prediction
from edge_tasks.link_prediction.link_prediction import run_link_prediction_experiment, get_default_config

config = get_default_config()
config['model_name'] = 'graphsage'
config['dataset_name'] = 'cora'
results = run_link_prediction_experiment(config)

# Graph Embedding Visualization
from graph_embedding_visualization.graph_embedding_visualization import run_embedding_visualization_experiment, get_default_config

config = get_default_config()
config['model_name'] = 'gcn'
config['dataset_name'] = 'cora'
results = run_embedding_visualization_experiment(config)
```

### Running the Backend Server
```bash
python run_backend.py
```

The backend server will start on `http://localhost:8001` with API documentation available at `/api/docs`.

## Supported Models

All backends support the following GNN architectures:
- **GCN** - Graph Convolutional Network (spectral-based)
- **GAT** - Graph Attention Network (attention-based)
- **GraphSAGE** - Inductive representation learning (sampling-based)
- **GIN** - Graph Isomorphism Network (powerful representation learning)
- **ChebNet** - Chebyshev Graph Convolution (spectral with polynomials)
- **SGC** - Simple Graph Convolution (fast and interpretable)

### Specialized Models
- **VGAE/GraphVAE** - Variational autoencoders for graph generation
- **GraphRNN** - Recurrent neural networks for graph generation
- **GraphGAN** - Generative adversarial networks for graph generation
- **TemporalGNN** - Temporal graph neural networks for dynamic graphs

## Configuration

Each backend uses a standardized configuration format:

```python
{
    'model_name': 'gcn',           # GNN architecture
    'dataset_name': 'cora',        # Dataset name
    'hidden_channels': 64,         # Hidden layer dimensions
    'learning_rate': 0.01,         # Learning rate
    'weight_decay': 5e-4,          # L2 regularization
    'epochs': 200,                 # Maximum training epochs
    'patience': 50,                # Early stopping patience
    'device': 'cuda'               # Device (cuda/cpu)
}
```

### Task-Specific Parameters
Each task includes additional parameters:
- **Node/Edge Tasks**: Classification/regression specific metrics
- **Graph Tasks**: Pooling methods, graph-level features
- **Community Detection**: Community detection algorithms, modularity optimization
- **Anomaly Detection**: Anomaly detection algorithms, threshold methods
- **Graph Generation**: Generation methods, quality metrics
- **Embedding Visualization**: Dimensionality reduction, clustering methods
- **Dynamic Learning**: Temporal modeling, prediction horizons

## Output Structure

```
results/
├── node_classification/
│   └── experiment_name/
│       ├── config.json          # Experiment configuration
│       ├── results.json         # Training and test results
│       ├── best_model.pt        # Saved model checkpoint
│       └── visualizations/      # Training curves, confusion matrices
├── node_regression/
│   └── experiment_name/
│       ├── config.json
│       ├── results.json
│       ├── best_model.pt
│       └── visualizations/
├── link_prediction/
│   └── experiment_name/
│       ├── config.json
│       ├── results.json
│       ├── best_model.pt
│       └── visualizations/
├── graph_embedding_visualization/
│   └── experiment_name/
│       ├── config.json
│       ├── results.json
│       ├── embeddings.pkl       # Learned embeddings
│       ├── visualizations/      # 2D/3D plots, interactive visualizations
│       └── clustering_results/  # Clustering analysis
└── [other_tasks]/
    └── experiment_name/
        ├── config.json
        ├── results.json
        ├── best_model.pt
        └── visualizations/
```

## API Integration

All backends can be integrated with FastAPI routes:

```python
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class ExperimentRequest(BaseModel):
    model_name: str
    dataset_name: str
    hidden_channels: int = 64
    learning_rate: float = 0.01
    epochs: int = 200

@router.post("/train")
async def train_model(request: ExperimentRequest):
    config = request.dict()
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = run_experiment(config)  # Task-specific function
    return results
```

## Key Features

### ✅ **Comprehensive Coverage**
- All major GNN tasks implemented
- Multiple model architectures per task
- Extensive dataset support
- Production-ready implementations

### ✅ **Advanced Capabilities**
- **Graph Embedding Visualization**: 6 dimensionality reduction methods, 3 clustering algorithms, interactive visualizations
- **Dynamic Graph Learning**: Temporal modeling with 6 specialized models
- **Community Detection**: 4 traditional algorithms + GNN approaches
- **Anomaly Detection**: 3 traditional algorithms + GNN approaches
- **Graph Generation**: 4 generation models with quality evaluation

### ✅ **Robust Training**
- Early stopping with patience
- Model checkpointing
- Comprehensive logging
- GPU/CPU support
- Validation monitoring

### ✅ **Rich Evaluation**
- Task-specific metrics
- Multiple evaluation criteria
- Statistical analysis
- Visualization tools

### ✅ **Production Features**
- Error handling and recovery
- Memory optimization
- Scalable architectures
- API integration ready

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `hidden_channels` or use smaller datasets
2. **Import Errors**: Install required packages (`torch`, `torch-geometric`, `scikit-learn`, `umap-learn`)
3. **Training Not Converging**: Adjust learning rate and patience
4. **Dataset Issues**: Use synthetic data for testing
5. **Visualization Errors**: Install `plotly`, `seaborn`, `matplotlib`

### Performance Tips

- Use GPU when available for faster training
- Monitor validation metrics to prevent overfitting
- Experiment with different architectures for your specific task
- Use feature normalization for better convergence
- For large graphs, use GraphSAGE or SGC
- For embedding visualization, try UMAP for speed or t-SNE for quality

## Contributing

To add new backends:

1. Create the backend file in the appropriate task directory
2. Implement the standard interface (training, evaluation, prediction)
3. Add comprehensive documentation
4. Include visualization capabilities
5. Add to the task registry and configuration system

## Dependencies

### Core Dependencies
- `torch` >= 1.9.0
- `torch-geometric` >= 2.0.0
- `numpy` >= 1.21.0
- `scikit-learn` >= 1.0.0

### Visualization Dependencies
- `matplotlib` >= 3.5.0
- `seaborn` >= 0.11.0
- `plotly` >= 5.0.0
- `umap-learn` >= 0.5.0

### Optional Dependencies
- `networkx` >= 2.6.0
- `scipy` >= 1.7.0
- `pandas` >= 1.3.0

## Status

✅ **All Backend Implementations Complete**
- Node Classification: ✅ Complete
- Node Regression: ✅ Complete
- Edge Classification: ✅ Complete
- Link Prediction: ✅ Complete
- Graph Classification: ✅ Complete
- Graph Regression: ✅ Complete
- Community Detection: ✅ Complete
- Anomaly Detection: ✅ Complete
- Graph Generation: ✅ Complete
- Graph Embedding Visualization: ✅ Complete
- Dynamic Graph Learning: ✅ Complete

All backends are production-ready with comprehensive documentation, testing, and web interface integration. 