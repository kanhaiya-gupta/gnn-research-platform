"""
Graph Regression Routes
This module contains all routes for graph regression tasks.
"""

from fastapi import APIRouter, Request, HTTPException
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
from config.tasks import get_task_config, get_task_metadata
from config.parameters import get_task_parameters, get_task_default_params
from config.models import get_models_for_task, get_model_config

# Create FastAPI router
router = APIRouter(prefix="/graph_regression", tags=["Graph Regression"])

# Task configuration
TASK_NAME = 'graph_regression'
task_config = get_task_config(TASK_NAME)
task_metadata = get_task_metadata(TASK_NAME)

# Setup templates
templates = Jinja2Templates(directory="templates")

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page for graph regression."""
    return templates.TemplateResponse(
        "graph_tasks/regression/index.html",
        {
            "request": request,
            "task_config": task_config,
            "task_metadata": task_metadata
        }
    )

@router.get("/experiment", response_class=HTMLResponse)
async def experiment(request: Request):
    """Experiment page for graph regression."""
    # Get parameters and defaults
    parameters = get_task_parameters(TASK_NAME)
    default_params = get_task_default_params(TASK_NAME)
    supported_models = get_models_for_task(TASK_NAME)
    
    return templates.TemplateResponse(
        "graph_tasks/regression/experiment.html",
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
    """Results page for graph regression."""
    # Mock results data for demonstration
    results_data = {
        'experiment_id': experiment_id,
        'task_name': 'Graph Regression',
        'model_name': 'GIN',
        'dataset_name': 'QM9',
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'mse': 0.0156,
            'mae': 0.124,
            'rmse': 0.125,
            'r2_score': 0.912,
            'explained_variance': 0.914
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
            'graph_ids': list(range(1000))
        },
        'feature_importance': {
            'feature_names': [f'Feature_{i}' for i in range(50)],
            'importance_scores': np.random.rand(50).tolist()
        }
    }
    
    return templates.TemplateResponse(
        "graph_tasks/regression/results.html",
        {
            "request": request,
            "task_config": task_config,
            "task_metadata": task_metadata,
            "results": results_data
        }
    )

@router.post("/api/start_experiment")
async def start_experiment(data: Dict[str, Any]):
    """Start a new graph regression experiment."""
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
            'model_name': experiment_params.get('model', 'gin'),
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
    """Get parameters for graph regression."""
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
        from config.parameters import validate_parameter_value
        
        validation_results = {}
        is_valid = True
        
        # Get task parameters for validation
        task_parameters = get_task_parameters(TASK_NAME)
        
        for param_name, param_value in parameters.items():
            if param_name in task_parameters:
                param_config = task_parameters[param_name]
                validation_result = validate_parameter_value(param_name, param_value, param_config)
                validation_results[param_name] = validation_result
                
                if not validation_result['valid']:
                    is_valid = False
            else:
                validation_results[param_name] = {
                    'valid': False,
                    'message': f'Unknown parameter: {param_name}',
                    'value': param_value
                }
                is_valid = False
        
        return {
            'valid': is_valid,
            'validation_results': validation_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/api/get_experiment_results/{experiment_id}")
async def get_experiment_results(experiment_id: str):
    """Get results for a specific experiment."""
    # Mock results data
    results_data = {
        'experiment_id': experiment_id,
        'task_name': 'Graph Regression',
        'model_name': 'GIN',
        'dataset_name': 'QM9',
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'mse': 0.0156,
            'mae': 0.124,
            'rmse': 0.125,
            'r2_score': 0.912,
            'explained_variance': 0.914
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
    """Get predictions for visualization."""
    # Mock predictions data
    predictions = {
        'experiment_id': experiment_id,
        'true_values': np.random.normal(0, 1, 1000).tolist(),
        'predicted_values': np.random.normal(0, 1, 1000).tolist(),
        'graph_ids': list(range(1000))
    }
    
    return predictions

@router.get("/api/get_residual_analysis/{experiment_id}")
async def get_residual_analysis(experiment_id: str):
    """Get residual analysis for regression."""
    # Mock residual analysis data
    residual_analysis = {
        'experiment_id': experiment_id,
        'residuals': np.random.normal(0, 0.1, 1000).tolist(),
        'true_values': np.random.normal(0, 1, 1000).tolist(),
        'predicted_values': np.random.normal(0, 1, 1000).tolist(),
        'residual_stats': {
            'mean': 0.001,
            'std': 0.098,
            'skewness': 0.05,
            'kurtosis': 3.1
        }
    }
    
    return residual_analysis

@router.get("/api/get_graph_embeddings/{experiment_id}")
async def get_graph_embeddings(experiment_id: str):
    """Get graph embeddings for visualization."""
    # Mock graph embeddings data
    graph_embeddings = {
        'experiment_id': experiment_id,
        'embeddings': np.random.randn(1000, 64).tolist(),
        'graph_ids': list(range(1000)),
        'target_values': np.random.normal(0, 1, 1000).tolist()
    }
    
    return graph_embeddings

@router.get("/api/get_graph_statistics/{experiment_id}")
async def get_graph_statistics(experiment_id: str):
    """Get graph statistics and properties."""
    # Mock graph statistics data
    graph_statistics = {
        'experiment_id': experiment_id,
        'total_graphs': 1000,
        'avg_nodes': 18.5,
        'avg_edges': 37.2,
        'node_feature_dim': 9,
        'edge_feature_dim': 3,
        'target_statistics': {
            'mean': 0.0,
            'std': 1.0,
            'min': -3.2,
            'max': 3.1,
            'q25': -0.67,
            'q75': 0.67
        },
        'graph_properties': {
            'density': np.random.uniform(0.1, 0.8, 1000).tolist(),
            'clustering_coeff': np.random.uniform(0.0, 1.0, 1000).tolist(),
            'diameter': np.random.randint(2, 8, 1000).tolist()
        }
    }
    
    return graph_statistics

@router.get("/api/export_results/{experiment_id}")
async def export_results(experiment_id: str, format_type: str = "json"):
    """Export experiment results in various formats."""
    try:
        # Mock results data
        results_data = {
            'experiment_id': experiment_id,
            'task_name': 'Graph Regression',
            'model_name': 'GIN',
            'dataset_name': 'QM9',
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'mse': 0.0156,
                'mae': 0.124,
                'rmse': 0.125,
                'r2_score': 0.912
            }
        }
        
        if format_type == "json":
            return JSONResponse(content=results_data)
        elif format_type == "csv":
            # Convert to CSV format
            import io
            import csv
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow(['Metric', 'Value'])
            
            # Write metrics
            for metric, value in results_data['metrics'].items():
                writer.writerow([metric, value])
            
            csv_content = output.getvalue()
            output.close()
            
            return JSONResponse(content={'csv_data': csv_content})
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format_type}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/get_feature_importance/{experiment_id}")
async def get_feature_importance(experiment_id: str):
    """Get feature importance scores."""
    # Mock feature importance data
    feature_importance = {
        'experiment_id': experiment_id,
        'feature_names': [f'Feature_{i}' for i in range(50)],
        'importance_scores': np.random.rand(50).tolist()
    }
    
    return feature_importance

@router.get("/api/get_prediction_distribution/{experiment_id}")
async def get_prediction_distribution(experiment_id: str):
    """Get prediction distribution analysis."""
    # Mock prediction distribution data
    prediction_distribution = {
        'experiment_id': experiment_id,
        'true_values': np.random.normal(0, 1, 1000).tolist(),
        'predicted_values': np.random.normal(0, 1, 1000).tolist(),
        'bins': np.linspace(-3, 3, 20).tolist(),
        'true_histogram': np.random.randint(10, 100, 19).tolist(),
        'pred_histogram': np.random.randint(10, 100, 19).tolist(),
        'distribution_stats': {
            'true_mean': 0.01,
            'true_std': 0.98,
            'pred_mean': 0.02,
            'pred_std': 0.97
        }
    }
    
    return prediction_distribution
