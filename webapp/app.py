#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNN Platform Web Application
Main FastAPI application for the Graph Neural Network platform.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
import uvicorn
import os
import time
from pathlib import Path

# Import configuration
from webapp.config.config import Config

# Create FastAPI app
app = FastAPI(
    title="GNN Platform",
    description="A comprehensive platform for Graph Neural Network experiments",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
current_dir = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(current_dir, "static")), name="static")
app.mount("/css", StaticFiles(directory=os.path.join(current_dir, "static", "css")), name="css")
app.mount("/js", StaticFiles(directory=os.path.join(current_dir, "static", "js")), name="js")

# Setup templates
templates = Jinja2Templates(directory=os.path.join(current_dir, "templates"))

# Initialize configuration
config = Config()

# Import routers from individual purpose modules
from webapp.node_tasks.classification.routes import router as node_classification_router
from webapp.node_tasks.regression.routes import router as node_regression_router
from webapp.edge_tasks.classification.routes import router as edge_classification_router
from webapp.edge_tasks.link_prediction.routes import router as link_prediction_router
from webapp.graph_tasks.classification.routes import router as graph_classification_router
from webapp.graph_tasks.regression.routes import router as graph_regression_router
from webapp.community_detection.routes import router as community_detection_router
from webapp.anomaly_detection.routes import router as anomaly_detection_router
from webapp.dynamic_graph_learning.routes import router as dynamic_graph_learning_router
from webapp.graph_generation.routes import router as graph_generation_router
from webapp.graph_embedding_visualization.routes import router as graph_embedding_visualization_router

# Include routers with proper prefixes
app.include_router(node_classification_router, prefix="/purpose/node_classification", tags=["node_classification"])
app.include_router(node_regression_router, prefix="/purpose/node_regression", tags=["node_regression"])
app.include_router(edge_classification_router, prefix="/purpose/edge_classification", tags=["edge_classification"])
app.include_router(link_prediction_router, prefix="/purpose/link_prediction", tags=["link_prediction"])
app.include_router(graph_classification_router, prefix="/purpose/graph_classification", tags=["graph_classification"])
app.include_router(graph_regression_router, prefix="/purpose/graph_regression", tags=["graph_regression"])
app.include_router(community_detection_router, prefix="/purpose/community_detection", tags=["community_detection"])
app.include_router(anomaly_detection_router, prefix="/purpose/anomaly_detection", tags=["anomaly_detection"])
app.include_router(dynamic_graph_learning_router, prefix="/purpose/dynamic_graph_learning", tags=["dynamic_graph_learning"])
app.include_router(graph_generation_router, prefix="/purpose/graph_generation", tags=["graph_generation"])
app.include_router(graph_embedding_visualization_router, prefix="/purpose/graph_embedding_visualization", tags=["graph_embedding_visualization"])

# Mapping between config purpose names and template paths
PURPOSE_TEMPLATE_MAPPING = {
    'node_classification': 'node_tasks/classification',
    'node_regression': 'node_tasks/regression',
    'edge_classification': 'edge_tasks/classification',
    'link_prediction': 'edge_tasks/link_prediction',
    'graph_classification': 'graph_tasks/classification',
    'graph_regression': 'graph_tasks/regression',
    'community_detection': 'community_detection',
    'anomaly_detection': 'anomaly_detection',
    'dynamic_graph_learning': 'dynamic_graph_learning',
    'graph_generation': 'graph_generation',
    'graph_embedding_visualization': 'graph_embedding_visualization'
}

# Mapping between URL purpose names and task configuration names
PURPOSE_TASK_MAPPING = {
    'node_classification': 'node_classification',
    'node_regression': 'node_regression',
    'edge_classification': 'edge_classification',
    'link_prediction': 'link_prediction',
    'graph_classification': 'graph_classification',
    'graph_regression': 'graph_regression',
    'community_detection': 'community_detection',
    'anomaly_detection': 'anomaly_detection',
    'dynamic_graph_learning': 'dynamic_graph_learning',
    'graph_generation': 'graph_generation',
    'graph_embedding_visualization': 'graph_embedding'
}

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "purposes": config.get_all_purposes(),
            "config": config,
            "title": "GNN Dashboard - Modular Platform"
        }
    )

# Purpose-based routes
@app.get("/purpose/{purpose_name}", response_class=HTMLResponse)
async def purpose_page(request: Request, purpose_name: str):
    """Individual purpose page"""
    purpose_info = config.get_purpose_info(purpose_name)
    if not purpose_info:
        raise HTTPException(status_code=404, detail="Purpose not found")
    
    # Get the correct template path
    template_path = PURPOSE_TEMPLATE_MAPPING.get(purpose_name)
    if not template_path:
        raise HTTPException(status_code=404, detail="Template mapping not found")
    
    # Get the correct task name for configuration
    task_name = PURPOSE_TASK_MAPPING.get(purpose_name, purpose_name)
    
    models = config.get_models_by_purpose(task_name)
    parameters = config.get_parameters_by_purpose(task_name)
    
    return templates.TemplateResponse(
        f"{template_path}/index.html",
        {
            "request": request,
            "purpose": purpose_info,
            "purpose_name": purpose_name,
            "purpose_key": purpose_name,
            "models": models,
            "parameters": parameters,
            "config": config,
            "title": f"{purpose_info['name']} - GNN"
        }
    )

@app.get("/purpose/{purpose_name}/experiment/{model_id}", response_class=HTMLResponse)
async def purpose_experiment_page(request: Request, purpose_name: str, model_id: str):
    """Experiment page for specific purpose and model"""
    purpose_info = config.get_purpose_info(purpose_name)
    if not purpose_info:
        raise HTTPException(status_code=404, detail="Purpose not found")
    
    # Get the correct template path
    template_path = PURPOSE_TEMPLATE_MAPPING.get(purpose_name)
    if not template_path:
        raise HTTPException(status_code=404, detail="Template mapping not found")
    
    # Get the correct task name for configuration
    task_name = PURPOSE_TASK_MAPPING.get(purpose_name, purpose_name)
    
    models = config.get_models_by_purpose(task_name)
    if model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_info = models[model_id]
    
    # Get model-specific parameters
    model_specific_params = config.get_model_specific_parameters(task_name, model_id)
    
    # Create default parameters for the template
    default_params = {
        "hidden_dim": 64,
        "num_layers": 3,
        "learning_rate": 0.001,
        "epochs": 100,
        "dropout": 0.1
    }
    
    # Add model-specific default parameters
    for param_id, param_info in model_specific_params.items():
        if isinstance(param_info, dict) and 'default' in param_info:
            default_params[param_id] = param_info['default']
    
    return templates.TemplateResponse(
        f"{template_path}/experiment.html",
        {
            "request": request,
            "purpose": purpose_info,
            "purpose_name": purpose_name,
            "purpose_key": purpose_name,
            "model": model_info,
            "model_id": model_id,
            "parameters": model_specific_params,
            "default_params": default_params,
            "config": config,
            "title": f"Experiment with {model_info['name']} - {purpose_info['name']}"
        }
    )

@app.get("/purpose/{purpose_name}/results/{experiment_id}", response_class=HTMLResponse)
async def purpose_results_page(request: Request, purpose_name: str, experiment_id: str):
    """Results page for specific purpose and experiment"""
    purpose_info = config.get_purpose_info(purpose_name)
    if not purpose_info:
        raise HTTPException(status_code=404, detail="Purpose not found")
    
    # Get the correct template path
    template_path = PURPOSE_TEMPLATE_MAPPING.get(purpose_name)
    if not template_path:
        raise HTTPException(status_code=404, detail="Template mapping not found")
    
    # Get the correct task name for configuration
    task_name = PURPOSE_TASK_MAPPING.get(purpose_name, purpose_name)
    
    # Try to get experiment results
    try:
        # Import the router to access its functions
        if purpose_name == 'node_classification':
            from webapp.node_tasks.classification.routes import get_experiment_results
        elif purpose_name == 'node_regression':
            from webapp.node_tasks.regression.routes import get_experiment_results
        elif purpose_name == 'edge_classification':
            from webapp.edge_tasks.classification.routes import get_experiment_results
        elif purpose_name == 'link_prediction':
            from webapp.edge_tasks.link_prediction.routes import get_experiment_results
        elif purpose_name == 'graph_classification':
            from webapp.graph_tasks.classification.routes import get_experiment_results
        elif purpose_name == 'graph_regression':
            from webapp.graph_tasks.regression.routes import get_experiment_results
        elif purpose_name == 'community_detection':
            from webapp.community_detection.routes import get_experiment_results
        elif purpose_name == 'anomaly_detection':
            from webapp.anomaly_detection.routes import get_experiment_results
        elif purpose_name == 'dynamic_graph_learning':
            from webapp.dynamic_graph_learning.routes import get_experiment_results
        elif purpose_name == 'graph_generation':
            from webapp.graph_generation.routes import get_experiment_results
        elif purpose_name == 'graph_embedding_visualization':
            from webapp.graph_embedding_visualization.routes import get_experiment_results
        else:
            raise HTTPException(status_code=404, detail="Purpose not supported")
        
        # Get experiment results
        results_data = await get_experiment_results(experiment_id)
        
        return templates.TemplateResponse(
            f"{template_path}/results.html",
            {
                "request": request,
                "purpose": purpose_info,
                "purpose_name": purpose_name,
                "purpose_key": purpose_name,
                "experiment_id": experiment_id,
                "results": results_data,
                "config": config,
                "title": f"Results for {purpose_info['name']}"
            }
        )
        
    except HTTPException as e:
        if e.status_code == 404:
            # If no results found, show a message
            return templates.TemplateResponse(
                f"{template_path}/results.html",
                {
                    "request": request,
                    "purpose": purpose_info,
                    "purpose_name": purpose_name,
                    "purpose_key": purpose_name,
                    "experiment_id": experiment_id,
                    "error": "No results found for this experiment. Please run training first.",
                    "config": config,
                    "title": f"Results for {purpose_info['name']}"
                }
            )
        else:
            raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API endpoints
@app.post("/api/train/{purpose_name}/{model_id}")
async def train_model(purpose_name: str, model_id: str, request: Request):
    """Train a model for a specific purpose"""
    try:
        # Get the correct task name for configuration
        task_name = PURPOSE_TASK_MAPPING.get(purpose_name, purpose_name)
        
        # Get request data
        data = await request.json()
        
        # Here you would call the appropriate backend function
        # For now, return a mock response
        return {
            "status": "success",
            "message": f"Training started for {purpose_name}/{model_id}",
            "experiment_id": f"{purpose_name}_{model_id}_{int(time.time())}",
            "parameters": data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict/{purpose_name}/{model_id}")
async def predict_model(purpose_name: str, model_id: str, request: Request):
    """Make predictions with a trained model"""
    try:
        # Get request data
        data = await request.json()
        
        # Here you would call the appropriate backend function
        # For now, return a mock response
        return {
            "status": "success",
            "message": f"Prediction completed for {purpose_name}/{model_id}",
            "predictions": [0.1, 0.2, 0.3, 0.4, 0.5]  # Mock predictions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/results/{purpose_name}/{model_id}")
async def get_results(purpose_name: str, model_id: str):
    """Get results for a specific model"""
    try:
        # Here you would load actual results
        # For now, return mock results
        return {
            "status": "success",
            "results": {
                "accuracy": 0.85,
                "loss": 0.15,
                "training_time": 120.5,
                "epochs": 100
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000) 