"""
Graph Embedding Visualization Routes
This module contains all routes for graph embedding visualization tasks.
"""

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import uuid
from datetime import datetime
from typing import Dict, Any

# Import configuration
from webapp.config.tasks import get_task_config, get_task_metadata
from webapp.config.parameters import get_task_parameters, get_task_default_params
from webapp.config.models import get_models_for_task

# Create FastAPI router
router = APIRouter(prefix="/graph_embedding_visualization", tags=["graph_embedding_visualization"])

# Setup templates
templates = Jinja2Templates(directory="templates")

# Task configuration
TASK_NAME = 'graph_embedding_visualization'
task_config = get_task_config(TASK_NAME)
task_metadata = get_task_metadata(TASK_NAME)

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page for graph embedding visualization."""
    return templates.TemplateResponse(
        "graph_embedding_visualization/index.html",
        {
            "request": request,
            "task_name": "Graph Embedding Visualization",
            "description": "Embed graph elements into vector spaces and visualize"
        }
    )

@router.get("/experiment", response_class=HTMLResponse)
async def experiment(request: Request):
    """Experiment page for graph embedding visualization."""
    # Mock parameters and models
    parameters = {
        'embedding_dim': {
            'name': 'Embedding Dimension',
            'type': 'int',
            'default': 64,
            'min': 16,
            'max': 256,
            'description': 'Dimension of the node embeddings'
        },
        'walk_length': {
            'name': 'Walk Length',
            'type': 'int',
            'default': 80,
            'min': 10,
            'max': 200,
            'description': 'Length of random walks'
        },
        'num_walks': {
            'name': 'Number of Walks',
            'type': 'int',
            'default': 10,
            'min': 1,
            'max': 50,
            'description': 'Number of random walks per node'
        }
    }
    
    supported_models = {
        'node2vec': {
            'name': 'Node2Vec',
            'description': 'Scalable feature learning for networks'
        },
        'deepwalk': {
            'name': 'DeepWalk',
            'description': 'Social representations of networks'
        }
    }
    
    return templates.TemplateResponse(
        "graph_embedding_visualization/experiments.html",
        {
            "request": request,
            "task_name": "Graph Embedding Visualization",
            "parameters": parameters,
            "supported_models": supported_models
        }
    )

@router.get("/results", response_class=HTMLResponse)
async def results(request: Request):
    """Results page for graph embedding visualization."""
    experiment_id = uuid.uuid4().hex
    # Mock results data
    results_data = {
        'experiment_id': experiment_id,
        'task_name': 'Graph Embedding Visualization',
        'model_name': 'Node2Vec',
        'dataset_name': 'Cora',
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'embedding_dim': 64,
            'num_nodes': 2708,
            'visualization_method': 't-SNE',
            'clustering_score': 0.78
        },
        'embeddings': np.random.randn(2708, 64).tolist(),
        'labels': np.random.randint(0, 7, 2708).tolist(),
        'node_ids': list(range(2708)),
        'embedding_2d': np.random.randn(2708, 2).tolist()
    }
    return templates.TemplateResponse(
        "graph_embedding_visualization/results.html",
        {
            "request": request,
            "results": results_data
        }
    )

@router.get("/api/get_embeddings/{experiment_id}")
async def get_embeddings(experiment_id: str):
    """Get node embeddings for visualization."""
    # Mock embeddings data
    embeddings_data = {
        'experiment_id': experiment_id,
        'embeddings': np.random.randn(2708, 64).tolist(),
        'labels': np.random.randint(0, 7, 2708).tolist(),
        'node_ids': list(range(2708)),
        'embedding_2d': np.random.randn(2708, 2).tolist()
    }
    return JSONResponse(content=embeddings_data)

@router.get("/api/get_embedding_analysis/{experiment_id}")
async def get_embedding_analysis(experiment_id: str):
    """Get analysis of embedding quality and clustering."""
    # Mock analysis data
    analysis_data = {
        'experiment_id': experiment_id,
        'clustering_score': 0.78,
        'silhouette_score': 0.65,
        'num_clusters': 7,
        'cluster_sizes': [400, 380, 390, 410, 370, 380, 378]
    }
    return JSONResponse(content=analysis_data)

@router.post("/api/start_embedding")
async def start_embedding(request: Request):
    """Start embedding generation process."""
    try:
        data = await request.json()
        experiment_id = data.get('experiment_id', uuid.uuid4().hex)
        
        # Mock response for embedding start
        return JSONResponse(content={
            "status": "success",
            "message": "Embedding generation started",
            "experiment_id": experiment_id
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
