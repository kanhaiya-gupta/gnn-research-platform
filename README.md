# Graph Neural Network (GNN) Platform

A comprehensive web platform for Graph Neural Network research and applications. This platform provides a modular, extensible framework for exploring GNN models across multiple domains and tasks with **complete backend implementations** for all major GNN tasks.

## 🌟 Overview

The GNN Platform is a sophisticated web interface that enables researchers and practitioners to experiment with Graph Neural Networks across a wide range of applications. Built with a modular architecture, it supports **12+ different GNN purposes** and **20+ popular GNN models** with **production-ready backend implementations**.

## 🎯 GNN Applications & Purposes

The platform supports the following GNN applications with **[complete backend implementations](docs/README_BACKEND.md)**:

### ✅ Node Tasks
- **[Node Classification](docs/node_classification.md)** - Classify nodes into different categories
- **[Node Regression](docs/node_regression.md)** - Predict continuous values for nodes

### ✅ Edge Tasks  
- **[Edge Classification](docs/edge_classification.md)** - Classify edges into different types
- **[Link Prediction](docs/link_prediction.md)** - Predict missing or future edges

### ✅ Graph Tasks
- **[Graph Classification](docs/graph_classification.md)** - Classify entire graphs into categories
- **[Graph Regression](docs/graph_regression.md)** - Predict continuous values for entire graphs

### ✅ Community Detection
- **[Community Detection](docs/community_detection.md)** - Discover communities and clusters in graphs

### ✅ Anomaly Detection
- **[Anomaly Detection](docs/anomaly_detection.md)** - Detect anomalous nodes, edges, or subgraphs

### ✅ Graph Generation
- **[Graph Generation](docs/graph_generation.md)** - Generate new graphs with desired properties

### ✅ Graph Embedding & Visualization
- **[Graph Embedding Visualization](docs/graph_embedding_visualization.md)** - Learn low-dimensional graph representations

### ✅ Dynamic Graph Learning
- **[Dynamic Graph Learning](docs/dynamic_graph_learning.md)** - Handle temporal and evolving graphs

## 🧠 Supported GNN Models

### Core Models (All Tasks)
- **Graph Convolutional Network (GCN)**: Semi-supervised learning with graph convolutions
- **Graph Attention Network (GAT)**: Attention-based graph neural network
- **GraphSAGE**: Inductive representation learning on large graphs
- **Graph Isomorphism Network (GIN)**: Maximally powerful GNN for graph classification
- **Chebyshev Graph Convolutional Network**: Spectral graph convolution
- **Simple Graph Convolution (SGC)**: Fast and interpretable graph convolution

### Specialized Models
- **Variational Graph Autoencoder (VGAE)**: Probabilistic graph generation
- **GraphVAE**: Variational autoencoder for graph generation
- **GraphRNN**: Recurrent neural networks for graph generation
- **GraphGAN**: Generative adversarial networks for graph generation
- **TemporalGCN/TemporalGAT**: Temporal graph neural networks
- **TemporalGraphSAGE/TemporalGIN**: Temporal inductive learning
- **RecurrentGNN**: Recurrent graph neural networks
- **TemporalTransformer**: Transformer-based temporal modeling

## 🚀 Features

### ✅ **Complete Backend Implementations**
- **All 12 GNN Tasks**: Fully implemented with PyTorch Geometric
- **Production-Ready**: Comprehensive training, evaluation, and prediction
- **Advanced Capabilities**: Embedding visualization, dynamic learning, graph generation
- **Robust Training**: Early stopping, checkpointing, GPU/CPU support

### Interactive Dashboard
- **GNN Purposes Overview**: Visual cards showcasing different GNN applications
- **Model Explorer**: Browse supported GNN models with detailed information
- **Research Impact**: Learn about GNN applications in various fields
- **Quick Start Guide**: Step-by-step instructions for getting started

### Experiment Interface
- **Parameter Configuration**: Intuitive forms for setting model parameters
- **Training Controls**: Start, monitor, and control GNN training
- **Real-time Feedback**: Live updates on training progress and results
- **Visualization**: Interactive plots and graph visualizations

### Research Tools
- **Result Analysis**: Comprehensive analysis of training results
- **Performance Metrics**: Loss curves, convergence analysis, and accuracy measures
- **Comparison Tools**: Compare different GNN models and configurations
- **Export Capabilities**: Download results, plots, and model parameters

### Advanced Features
- **Modular Architecture**: Extensible design with purpose-specific modules
- **Hyperparameter Tuning**: Automated optimization with grid search and Bayesian optimization
- **Dataset Management**: Built-in support for popular graph datasets
- **Model Sharing**: Export trained models and share experiments
- **Educational Resources**: Comprehensive documentation and tutorials

### 🎨 **Advanced Visualization**
- **Graph Embedding Visualization**: 6 dimensionality reduction methods (t-SNE, UMAP, PCA, MDS, Isomap, Kernel PCA)
- **Interactive Plots**: 2D/3D scatter plots with Plotly
- **Clustering Analysis**: K-means, DBSCAN, Spectral clustering with quality metrics
- **Training Curves**: Real-time loss and metric visualization
- **Graph Structure**: Network visualization and analysis

## 🛠️ Technology Stack

### Frontend
- **Framework**: FastAPI with Jinja2 templates
- **Styling**: Custom CSS with Bootstrap 5 integration
- **JavaScript**: Vanilla JS with modern ES6+ features
- **Icons**: Font Awesome for comprehensive iconography
- **Charts**: Chart.js for interactive visualizations

### Backend
- **Deep Learning**: PyTorch + PyTorch Geometric
- **Machine Learning**: Scikit-learn for traditional algorithms
- **Visualization**: Matplotlib, Seaborn, Plotly, UMAP
- **Data Processing**: NumPy, Pandas, NetworkX
- **API Framework**: FastAPI with WebSocket support

## 📦 Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.9.0+
- PyTorch Geometric 2.0.0+
- FastAPI, Uvicorn
- Additional dependencies for visualization

### Setup
1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd graph-neural-network
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the platform**:
   ```bash
   # Start frontend
   python run_frontend.py
   
   # Start backend (in separate terminal)
   python run_backend.py
   ```

4. **Access the application**:
   - Frontend: http://localhost:5000
   - Backend API: http://localhost:8001

For detailed installation instructions, see [INSTALL.md](INSTALL.md).

## 🎮 Usage

### Getting Started
1. **Explore the Dashboard**: Visit the main page to see GNN purposes and supported models
2. **Choose an Application**: Select from node tasks, edge tasks, graph tasks, or specialized applications
3. **Select a Model**: Pick the GNN architecture that suits your needs
4. **Configure Parameters**: Set model parameters and training configuration
5. **Start Training**: Launch GNN training and monitor progress
6. **Analyze Results**: View interactive visualizations and performance metrics

### Quick Examples

#### Node Classification
```python
from node_tasks.classification.nodes_classification import run_node_classification_experiment

config = {
    'model_name': 'gat',
    'dataset_name': 'cora',
    'hidden_channels': 64,
    'learning_rate': 0.01,
    'epochs': 200
}
results = run_node_classification_experiment(config)
```

#### Graph Embedding Visualization
```python
from graph_embedding_visualization.graph_embedding_visualization import run_embedding_visualization_experiment

config = {
    'model_name': 'gcn',
    'dataset_name': 'cora',
    'embedding_dim': 32,
    'learning_rate': 0.01,
    'epochs': 200
}
results = run_embedding_visualization_experiment(config)
```

### Navigation
- **Dashboard** (`/`): Overview of GNN applications and models
- **Applications**: Dropdown menu with all GNN tasks
- **Experiment Pages**: Training interface for each model
- **Results Pages**: Analysis and visualization

## 🔧 Configuration

### Environment Variables
- `API_BASE_URL`: Backend API URL (default: http://localhost:8001)

### Task-Specific Configuration
Each GNN task has its own configuration system:
- **Model Parameters**: Architecture-specific settings
- **Training Parameters**: Learning rate, epochs, optimization
- **Dataset Parameters**: Data preprocessing and augmentation
- **Evaluation Parameters**: Metrics and validation strategies

## 📊 Backend API

### Training Endpoints
- `POST /api/train/{purpose_name}/{model_id}`: Submit training requests
- `GET /api/status/{experiment_id}`: Get training status
- `GET /api/results/{experiment_id}`: Retrieve results

### Model Management
- `GET /api/models/{purpose_name}`: List available models
- `GET /api/parameters/{purpose_name}/{model_id}`: Get model parameters
- `POST /api/predict/{purpose_name}/{model_id}`: Make predictions

### WebSocket Events
- `training_progress`: Real-time training updates
- `experiment_complete`: Training completion notification
- `error_occurred`: Error handling and reporting

## 📚 Documentation

### Backend Documentation
- **[Backend Overview](docs/README_BACKEND.md)**: Complete backend implementation guide with all 12 GNN tasks
- **[Task-Specific Docs](docs/)**: Detailed documentation for each GNN task
- **[API Reference](docs/)**: Backend API documentation

### Task Documentation
- [Node Classification](docs/node_classification.md) - Multi-class classification for nodes
- [Node Regression](docs/node_regression.md) - Continuous value prediction for nodes
- [Edge Classification](docs/edge_classification.md) - Classifying edge types
- [Link Prediction](docs/link_prediction.md) - Predicting missing edges
- [Graph Classification](docs/graph_classification.md) - Classifying entire graphs
- [Graph Regression](docs/graph_regression.md) - Predicting graph-level properties
- [Community Detection](docs/community_detection.md) - Detecting communities in graphs
- [Anomaly Detection](docs/anomaly_detection.md) - Detecting anomalous nodes/edges
- [Graph Generation](docs/graph_generation.md) - Generating new graphs
- [Graph Embedding Visualization](docs/graph_embedding_visualization.md) - Learning and visualizing graph embeddings
- [Dynamic Graph Learning](docs/dynamic_graph_learning.md) - Learning from evolving graphs

## 🎨 Design Philosophy

### User Experience
- **Intuitive Interface**: Clean, modern design with clear navigation
- **Educational Focus**: Comprehensive explanations and examples
- **Interactive Learning**: Hands-on exploration of GNN concepts
- **Professional Presentation**: Research-grade visualization and analysis

### Accessibility
- **Responsive Design**: Works across all device sizes
- **Keyboard Navigation**: Full keyboard accessibility
- **Clear Typography**: Readable fonts and proper contrast
- **Loading States**: Clear feedback during operations

## 🔬 Research Applications

### Social Network Analysis
- Community detection in social networks
- Influence prediction and recommendation systems
- Anomaly detection in social graphs

### Bioinformatics
- Protein-protein interaction networks
- Drug discovery and molecular property prediction
- Gene regulatory network analysis

### Computer Vision
- Scene graph understanding
- Object relationship modeling
- Image segmentation with graph structures

### Natural Language Processing
- Knowledge graph completion
- Document classification with citation networks
- Semantic role labeling

### Recommender Systems
- User-item interaction modeling
- Collaborative filtering with graph structures
- Multi-modal recommendation

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for:
- Adding new GNN models
- Implementing new tasks
- Improving documentation
- Bug reports and feature requests

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch Geometric team for the excellent GNN library
- FastAPI team for the modern web framework
- The GNN research community for inspiring this work

## 📞 Support

For questions, issues, or contributions:
- Check the documentation in the `docs/` directory
- Review the troubleshooting sections
- Submit issues through the platform

---

**Status**: ✅ **Production Ready** - All 12 GNN tasks fully implemented with comprehensive backend support, advanced visualization capabilities, and web interface integration. 