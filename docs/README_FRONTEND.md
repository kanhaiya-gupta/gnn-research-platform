# Graph Neural Network (GNN) Platform

A comprehensive web platform for Graph Neural Network research and applications, inspired by Physics-Informed Neural Networks (PINNs) architecture. This platform provides a modular, extensible framework for exploring GNN models across multiple domains and tasks.

## 🌟 Overview

The GNN Platform is a sophisticated web interface that enables researchers and practitioners to experiment with Graph Neural Networks across a wide range of applications. Built with a modular architecture, it supports 12+ different GNN purposes and 20+ popular GNN models.

## 🎯 GNN Applications & Purposes

The platform supports the following GNN applications:

### 🔗 Node Tasks
- **Node Classification**: Classify nodes into different categories
- **Node Regression**: Predict continuous values for nodes

### 🔗 Edge Tasks  
- **Edge Classification**: Classify edges into different types
- **Link Prediction**: Predict missing or future edges

### 📊 Graph Tasks
- **Graph Classification**: Classify entire graphs into categories
- **Graph Regression**: Predict continuous values for entire graphs

### 👥 Community Detection
- Discover communities and clusters in graphs
- Modularity optimization with GNN embeddings

### ⚠️ Anomaly Detection
- Detect anomalous nodes, edges, or subgraphs
- Unsupervised and supervised approaches

### 🎨 Graph Generation
- Generate new graphs with desired properties
- Variational and sequential generation methods

### 👁️ Graph Embedding & Visualization
- Learn low-dimensional graph representations
- Interactive visualization of embeddings

### ⏰ Dynamic Graph Learning
- Handle temporal and evolving graphs
- Time-aware graph neural networks

### 🌐 Multi-Relational Graphs
- Handle heterogeneous graphs with multiple edge types
- Relational graph convolutional networks

## 🧠 Supported GNN Models

### Core Models
- **Graph Convolutional Network (GCN)**: Semi-supervised learning with graph convolutions
- **Graph Attention Network (GAT)**: Attention-based graph neural network
- **GraphSAGE**: Inductive representation learning on large graphs
- **Graph Isomorphism Network (GIN)**: Maximally powerful GNN for graph classification
- **Chebyshev Graph Convolutional Network**: Spectral graph convolution

### Specialized Models
- **Graph Autoencoder (GAE)**: Unsupervised learning of graph representations
- **Variational Graph Autoencoder (VGAE)**: Probabilistic graph generation
- **SEAL**: Learning from subgraphs for link prediction
- **DiffPool**: Hierarchical graph representation learning
- **SortPool**: Sorting-based graph pooling
- **Node2Vec**: Scalable feature learning for nodes
- **Graph2Vec**: Learning distributed representations of graphs
- **DynGraph2Vec**: Capturing network dynamics
- **Temporal Graph Network (TGN)**: Deep learning on dynamic graphs
- **Relational GCN (R-GCN)**: Modeling relational data
- **Composition-based GCN**: Multi-relational graph learning

## 🚀 Features

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

## 🛠️ Technology Stack

- **Frontend Framework**: FastAPI with Jinja2 templates
- **Styling**: Custom CSS with Bootstrap 5 integration
- **JavaScript**: Vanilla JS with modern ES6+ features
- **Icons**: Font Awesome for comprehensive iconography
- **Charts**: Chart.js for interactive visualizations
- **Backend Integration**: RESTful API communication

## 📦 Installation

### Prerequisites
- Python 3.8+
- FastAPI
- Uvicorn
- Jinja2
- httpx

### Setup
1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd graph-neural-network
   ```

2. **Install dependencies**:
   ```bash
   pip install fastapi uvicorn jinja2 httpx
   ```

3. **Start the GNN platform**:
   ```bash
   python run_gnn_platform.py
   ```

4. **Access the application**:
   - Frontend: http://localhost:5000
   - Backend API: http://localhost:8000

## 🎮 Usage

### Getting Started
1. **Explore the Dashboard**: Visit the main page to see GNN purposes and supported models
2. **Choose an Application**: Select from node tasks, edge tasks, graph tasks, or specialized applications
3. **Select a Model**: Pick the GNN architecture that suits your needs
4. **Configure Parameters**: Set model parameters and training configuration
5. **Start Training**: Launch GNN training and monitor progress
6. **Analyze Results**: View interactive visualizations and performance metrics

### Navigation
- **Dashboard** (`/`): Overview of GNN applications and models
- **Purpose Pages** (`/purpose/{purpose_name}`): Detailed information about specific purposes
- **Experiment Pages** (`/purpose/{purpose_name}/experiment/{model_id}`): Training interface
- **Results Pages** (`/purpose/{purpose_name}/results/{model_id}`): Analysis and visualization

### Interactive Features
- **Clickable Cards**: Interactive purpose and model cards with hover effects
- **Hover Effects**: Interactive cards with hover animations
- **Keyboard Shortcuts**: 
  - `Ctrl/Cmd + Enter`: Start training
  - `Escape`: Go back
- **Responsive Design**: Works on desktop, tablet, and mobile devices

## 🔧 Configuration

### Environment Variables
- `API_BASE_URL`: Backend API URL (default: http://localhost:8000)

### Customization
- **Adding New Models**: Update `config/models.py` with model definitions
- **Modifying Parameters**: Edit default parameters in the configuration
- **Styling**: Customize CSS variables in `static/css/main.css`
- **Templates**: Modify Jinja2 templates in the `templates/` directory

## 📊 API Integration

The frontend communicates with the backend API through the following endpoints:

### Training Endpoints
- `POST /api/train/{purpose_name}/{model_id}`: Submit training requests
- `POST /api/predict/{purpose_name}/{model_id}`: Make predictions with trained models
- `GET /api/results/{purpose_name}/{model_id}`: Retrieve training results

### Parameter Mapping
The frontend automatically maps user-friendly parameter names to backend API formats for each model type.

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
- Community detection and clustering
- Influence maximization
- Link prediction and recommendation

### Computational Biology
- Protein function prediction
- Drug-target interaction
- Disease gene identification

### Recommendation Systems
- User-item recommendation
- Session-based recommendation
- Multi-modal recommendation

### Cybersecurity
- Anomaly detection
- Malware classification
- Network intrusion detection

### Knowledge Graphs
- Entity linking
- Relation extraction
- Knowledge graph completion

## 🚀 Future Enhancements

### Planned Features
- **Graph Neural Architecture Search (GNAS)**: AutoML for GNNs
- **Graph-to-Graph Translation**: Transform graphs between domains
- **Federated Learning**: Distributed GNN training
- **Explainable AI**: Interpretable GNN predictions
- **Real-time Graph Processing**: Stream processing for dynamic graphs

### Extensibility
- **Plugin System**: Easy addition of new models and purposes
- **Custom Datasets**: Support for user-uploaded graph data
- **Advanced Visualizations**: 3D graph rendering and interactive plots
- **Collaboration Tools**: Multi-user experiments and sharing

## 🤝 Contributing

We welcome contributions to the GNN Platform! Please see our contributing guidelines for more information.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by the PINN platform architecture
- Built with FastAPI and modern web technologies
- Icons provided by Font Awesome
- Charts powered by Chart.js

---

**Built with ❤️ for the GNN research community** 