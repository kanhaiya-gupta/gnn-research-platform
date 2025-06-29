"""
Main FastAPI application for the GNN Platform
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from pathlib import Path

# Import configuration
from config.config import Config

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
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/css", StaticFiles(directory="static/css"), name="css")
app.mount("/js", StaticFiles(directory="static/js"), name="js")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Initialize configuration
config = Config()

# Import routers from individual purpose modules
from node_tasks.classification.routes import router as node_classification_router
from node_tasks.regression.routes import router as node_regression_router
from edge_tasks.classification.routes import router as edge_classification_router
from edge_tasks.link_prediction.routes import router as link_prediction_router
from graph_tasks.classification.routes import router as graph_classification_router
from graph_tasks.regression.routes import router as graph_regression_router
from community_detection.routes import router as community_detection_router
from anomaly_detection.routes import router as anomaly_detection_router
from dynamic_graph_learning.routes import router as dynamic_graph_learning_router
from graph_generation.routes import router as graph_generation_router
from graph_embedding_visualization.routes import router as graph_embedding_visualization_router

# Include routers
app.include_router(node_classification_router)
app.include_router(node_regression_router)
app.include_router(edge_classification_router)
app.include_router(link_prediction_router)
app.include_router(graph_classification_router)
app.include_router(graph_regression_router)
app.include_router(community_detection_router)
app.include_router(anomaly_detection_router)
app.include_router(dynamic_graph_learning_router)
app.include_router(graph_generation_router)
app.include_router(graph_embedding_visualization_router)

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
    
    models = config.get_models_by_purpose(purpose_name)
    parameters = config.get_parameters_by_purpose(purpose_name)
    
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
    
    models = config.get_models_by_purpose(purpose_name)
    if model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_info = models[model_id]
    
    # Get model-specific parameters
    model_specific_params = config.get_model_specific_parameters(purpose_name, model_id)
    
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

@app.get("/purpose/{purpose_name}/results/{model_id}", response_class=HTMLResponse)
async def purpose_results_page(request: Request, purpose_name: str, model_id: str):
    """Results page for specific purpose and model"""
    purpose_info = config.get_purpose_info(purpose_name)
    if not purpose_info:
        raise HTTPException(status_code=404, detail="Purpose not found")
    
    # Get the correct template path
    template_path = PURPOSE_TEMPLATE_MAPPING.get(purpose_name)
    if not template_path:
        raise HTTPException(status_code=404, detail="Template mapping not found")
    
    models = config.get_models_by_purpose(purpose_name)
    if model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_info = models[model_id]
    
    return templates.TemplateResponse(
        f"{template_path}/results.html",
        {
            "request": request,
            "purpose": purpose_info,
            "purpose_name": purpose_name,
            "purpose_key": purpose_name,
            "model": model_info,
            "model_id": model_id,
            "config": config,
            "title": f"Results for {model_info['name']} - {purpose_info['name']}"
        }
    )

# API endpoints for training and prediction
@app.post("/api/train/{purpose_name}/{model_id}")
async def train_model(purpose_name: str, model_id: str, request: Request):
    """Train a GNN model for a specific purpose"""
    try:
        data = await request.json()
        
        # Validate purpose and model
        purpose_info = config.get_purpose_info(purpose_name)
        if not purpose_info:
            raise HTTPException(status_code=404, detail="Purpose not found")
        
        models = config.get_models_by_purpose(purpose_name)
        if model_id not in models:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Map frontend parameters to backend format
        backend_params = config.map_parameters_to_backend(purpose_name, model_id, data)
        
        # Here you would call your GNN training backend
        # For now, return a mock response
        return {
            "status": "success",
            "message": f"Training started for {model_id} in {purpose_name}",
            "job_id": f"train_{purpose_name}_{model_id}_{hash(str(backend_params))}",
            "parameters": backend_params
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict/{purpose_name}/{model_id}")
async def predict_model(purpose_name: str, model_id: str, request: Request):
    """Make predictions with a trained GNN model"""
    try:
        data = await request.json()
        
        # Validate purpose and model
        purpose_info = config.get_purpose_info(purpose_name)
        if not purpose_info:
            raise HTTPException(status_code=404, detail="Purpose not found")
        
        models = config.get_models_by_purpose(purpose_name)
        if model_id not in models:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Here you would call your GNN prediction backend
        # For now, return a mock response
        return {
            "status": "success",
            "message": f"Prediction completed for {model_id} in {purpose_name}",
            "predictions": {
                "sample_1": 0.85,
                "sample_2": 0.23,
                "sample_3": 0.67
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/results/{purpose_name}/{model_id}")
async def get_results(purpose_name: str, model_id: str):
    """Get training results for a specific model"""
    try:
        # Validate purpose and model
        purpose_info = config.get_purpose_info(purpose_name)
        if not purpose_info:
            raise HTTPException(status_code=404, detail="Purpose not found")
        
        models = config.get_models_by_purpose(purpose_name)
        if model_id not in models:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Here you would fetch actual results from your backend
        # For now, return mock results
        return {
            "status": "success",
            "results": {
                "accuracy": 0.92,
                "loss": 0.08,
                "training_time": 120.5,
                "epochs_completed": 100,
                "best_epoch": 87
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000) 