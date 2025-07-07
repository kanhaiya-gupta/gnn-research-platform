"""
Node Regression Routes
This module contains all routes for node regression tasks using FastAPI.
"""

from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import json
import os
import uuid
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

# Import configuration
from webapp.config.tasks import get_task_config, get_task_metadata
from webapp.config.parameters import get_task_parameters, get_task_default_params
from webapp.config.models import get_models_for_task, get_model_config

# Create APIRouter
#router = APIRouter(prefix="/api", tags=["Node Regression"])
router = APIRouter(tags=["Node Regression"])

# Task configuration
TASK_NAME = 'node_regression'
task_config = get_task_config(TASK_NAME)
task_metadata = get_task_metadata(TASK_NAME)

# Setup templates
templates = Jinja2Templates(directory="webapp/templates")

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page for node regression."""
    return templates.TemplateResponse(
        "node_tasks/regression/index.html",
        {
            "request": request,
            "task_config": task_config,
            "task_metadata": task_metadata
        }
    )

@router.get("/experiment", response_class=HTMLResponse)
async def experiment(request: Request):
    """Experiment page for node regression."""
    # Get parameters and defaults
    parameters = get_task_parameters(TASK_NAME)
    default_params = get_task_default_params(TASK_NAME)
    supported_models = get_models_for_task(TASK_NAME)
    
    return templates.TemplateResponse(
        "node_tasks/regression/experiment.html",
        {
            "request": request,
            "task_config": task_config,
            "task_metadata": task_metadata,
            "parameters": parameters,
            "default_params": default_params,
            "supported_models": supported_models
        }
    )

@router.get("/results", response_class=HTMLResponse)
async def results(request: Request, experiment_id: str = "demo"):
    """Results page for node regression."""
    # Mock results data for demonstration
    results_data = {
        'experiment_id': experiment_id,
        'task_name': 'Node Regression',
        'model_name': 'GCN',
        'dataset_name': 'QM9',
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'mse': 0.0234,
            'mae': 0.156,
            'rmse': 0.153,
            'r2_score': 0.892,
            'explained_variance': 0.894
        },
        'training_history': {
            'epochs': list(range(1, 101)),
            'train_loss': [0.5, 0.4, 0.35, 0.3, 0.28, 0.26, 0.25, 0.24, 0.23, 0.22] + [0.21] * 90,
            'val_loss': [0.55, 0.45, 0.4, 0.35, 0.33, 0.31, 0.3, 0.29, 0.28, 0.27] + [0.26] * 90,
            'train_mae': [0.3, 0.25, 0.22, 0.2, 0.19, 0.18, 0.17, 0.16, 0.16, 0.15] + [0.15] * 90,
            'val_mae': [0.35, 0.3, 0.27, 0.25, 0.24, 0.23, 0.22, 0.21, 0.2, 0.19] + [0.18] * 90
        },
        'predictions': {
            'true_values': np.random.normal(0, 1, 1000).tolist(),
            'predicted_values': np.random.normal(0, 1, 1000).tolist(),
            'node_ids': list(range(1000))
        },
        'feature_importance': {
            'feature_names': [f'Feature_{i}' for i in range(50)],
            'importance_scores': np.random.rand(50).tolist()
        }
    }
    
    return templates.TemplateResponse(
        "node_tasks/regression/results.html",
        {
            "request": request,
            "task_config": task_config,
            "task_metadata": task_metadata,
            "results": results_data
        }
    )

@router.post("/api/start_experiment")
async def start_experiment(data: Dict[str, Any]):
    """Start a new node regression experiment."""
    try:
        # Validate parameters
        parameters = get_task_parameters(TASK_NAME)
        default_params = get_task_default_params(TASK_NAME)
        
        # Merge with defaults
        experiment_params = default_params.copy()
        experiment_params.update(data.get('parameters', {}))
        
        # Generate experiment ID
        experiment_id = str(uuid.uuid4())
        
        # Store experiment data
        experiment_data = {
            'experiment_id': experiment_id,
            'task_name': TASK_NAME,
            'parameters': experiment_params,
            'status': 'running',
            'start_time': datetime.now().isoformat(),
            'model_name': experiment_params.get('model', 'gcn'),
            'dataset_name': experiment_params.get('dataset', 'qm9')
        }
        
        # In a real implementation, you would save this to a database
        # and start the training process
        
        return {
            'success': True,
            'experiment_id': experiment_id,
            'message': 'Experiment started successfully'
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/api/experiment_status/{experiment_id}")
async def experiment_status(experiment_id: str):
    """Get the status of an experiment."""
    # Mock status data
    status_data = {
        'experiment_id': experiment_id,
        'status': 'running',
        'progress': 0.65,
        'current_epoch': 65,
        'total_epochs': 100,
        'current_metrics': {
            'train_loss': 0.22,
            'val_loss': 0.26,
            'train_mae': 0.15,
            'val_mae': 0.18
        },
        'estimated_time_remaining': '00:02:30'
    }
    
    return status_data

@router.post("/api/stop_experiment/{experiment_id}")
async def stop_experiment(experiment_id: str):
    """Stop a running experiment."""
    try:
        # In a real implementation, you would stop the training process
        
        return {
            'success': True,
            'message': f'Experiment {experiment_id} stopped successfully'
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/api/get_parameters")
async def get_parameters():
    """Get parameters for node regression."""
    parameters = get_task_parameters(TASK_NAME)
    default_params = get_task_default_params(TASK_NAME)
    supported_models = get_models_for_task(TASK_NAME)
    
    return {
        'parameters': parameters,
        'default_params': default_params,
        'supported_models': supported_models
    }

@router.post("/api/validate_parameters")
async def validate_parameters(data: Dict[str, Any]):
    """Validate experiment parameters."""
    try:
        parameters = data.get('parameters', {})
        
        # Import validation function
        from webapp.config.parameters import validate_parameter_value
        
        validation_results = {}
        is_valid = True
        
        for param_name, value in parameters.items():
            valid, result = validate_parameter_value(TASK_NAME, param_name, value)
            validation_results[param_name] = {
                'valid': valid,
                'result': result
            }
            if not valid:
                is_valid = False
        
        return {
            'valid': is_valid,
            'validation_results': validation_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/api/get_experiment_results/{experiment_id}")
async def get_experiment_results(experiment_id: str):
    """Get results for a completed experiment."""
    # Mock results data
    results_data = {
        'experiment_id': experiment_id,
        'task_name': 'Node Regression',
        'model_name': 'GCN',
        'dataset_name': 'QM9',
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'hidden_dim': 64,
            'num_layers': 3,
            'dropout': 0.1,
            'learning_rate': 0.001,
            'epochs': 100
        },
        'metrics': {
            'mse': 0.0234,
            'mae': 0.156,
            'rmse': 0.153,
            'r2_score': 0.892,
            'explained_variance': 0.894
        },
        'training_history': {
            'epochs': list(range(1, 101)),
            'train_loss': [0.5, 0.4, 0.35, 0.3, 0.28, 0.26, 0.25, 0.24, 0.23, 0.22] + [0.21] * 90,
            'val_loss': [0.55, 0.45, 0.4, 0.35, 0.33, 0.31, 0.3, 0.29, 0.28, 0.27] + [0.26] * 90,
            'train_mae': [0.3, 0.25, 0.22, 0.2, 0.19, 0.18, 0.17, 0.16, 0.16, 0.15] + [0.15] * 90,
            'val_mae': [0.35, 0.3, 0.27, 0.25, 0.24, 0.23, 0.22, 0.21, 0.2, 0.19] + [0.18] * 90
        }
    }
    
    return results_data

@router.get("/api/get_predictions/{experiment_id}")
async def get_predictions(experiment_id: str):
    """Get predictions vs true values."""
    # Mock predictions data
    predictions_data = {
        'experiment_id': experiment_id,
        'true_values': np.random.normal(0, 1, 1000).tolist(),
        'predicted_values': np.random.normal(0, 1, 1000).tolist(),
        'node_ids': list(range(1000)),
        'residuals': np.random.normal(0, 0.2, 1000).tolist()
    }
    
    return predictions_data

@router.get("/api/get_residual_analysis/{experiment_id}")
async def get_residual_analysis(experiment_id: str):
    """Get residual analysis data."""
    # Mock residual analysis data
    residual_data = {
        'experiment_id': experiment_id,
        'residuals': np.random.normal(0, 0.2, 1000).tolist(),
        'predicted_values': np.random.normal(0, 1, 1000).tolist(),
        'true_values': np.random.normal(0, 1, 1000).tolist(),
        'residual_stats': {
            'mean': 0.001,
            'std': 0.198,
            'skewness': 0.05,
            'kurtosis': 3.1
        }
    }
    
    return residual_data

@router.get("/api/export_results/{experiment_id}")
async def export_results(experiment_id: str, format_type: str = "json"):
    """Export experiment results."""
    # Get results data
    results_data = {
        'experiment_id': experiment_id,
        'task_name': 'Node Regression',
        'model_name': 'GCN',
        'dataset_name': 'QM9',
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'mse': 0.0234,
            'mae': 0.156,
            'rmse': 0.153,
            'r2_score': 0.892,
            'explained_variance': 0.894
        }
    }
    
    if format_type == 'csv':
        # Convert to CSV format
        import io
        output = io.StringIO()
        df = pd.DataFrame([results_data])
        df.to_csv(output, index=False)
        output.seek(0)
        
        from fastapi.responses import Response
        return Response(
            content=output.getvalue(),
            media_type='text/csv',
            headers={'Content-Disposition': f'attachment; filename=node_regression_results_{experiment_id}.csv'}
        )
    else:
        # Return JSON
        return results_data

@router.get("/api/get_feature_importance/{experiment_id}")
async def get_feature_importance(experiment_id: str):
    """Get feature importance scores."""
    # Mock feature importance data
    feature_importance = {
        'experiment_id': experiment_id,
        'feature_names': [f'Feature_{i}' for i in range(50)],
        'importance_scores': np.random.rand(50).tolist(),
        'top_features': [f'Feature_{i}' for i in range(10)]
    }
    
    return feature_importance

@router.get("/api/get_node_embeddings/{experiment_id}")
async def get_node_embeddings(experiment_id: str):
    """Get node embeddings for visualization."""
    # Mock embeddings data
    embeddings_data = {
        'experiment_id': experiment_id,
        'embeddings': np.random.randn(1000, 64).tolist(),
        'target_values': np.random.normal(0, 1, 1000).tolist(),
        'node_ids': list(range(1000))
    }
    
    return embeddings_data
