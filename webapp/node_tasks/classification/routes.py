"""
Node Classification Routes
This module contains all routes for node classification tasks using FastAPI.
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
from config.tasks import get_task_config, get_task_metadata
from config.parameters import get_task_parameters, get_task_default_params
from config.models import get_models_for_task, get_model_config

# Create APIRouter
router = APIRouter(prefix="/node_classification", tags=["Node Classification"])

# Task configuration
TASK_NAME = 'node_classification'
task_config = get_task_config(TASK_NAME)
task_metadata = get_task_metadata(TASK_NAME)

# Setup templates
templates = Jinja2Templates(directory="templates")

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page for node classification."""
    return templates.TemplateResponse(
        "node_tasks/classification/index.html",
        {
            "request": request,
            "task_config": task_config,
            "task_metadata": task_metadata
        }
    )

@router.get("/experiment", response_class=HTMLResponse)
async def experiment(request: Request):
    """Experiment page for node classification."""
    # Get parameters and defaults
    parameters = get_task_parameters(TASK_NAME)
    default_params = get_task_default_params(TASK_NAME)
    supported_models = get_models_for_task(TASK_NAME)
    
    return templates.TemplateResponse(
        "node_tasks/classification/experiment.html",
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
    """Results page for node classification."""
    # Mock results data for demonstration
    results_data = {
        'experiment_id': experiment_id,
        'task_name': 'Node Classification',
        'model_name': 'GCN',
        'dataset_name': 'Cora',
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'accuracy': 0.85,
            'f1_score': 0.84,
            'precision': 0.86,
            'recall': 0.83
        },
        'training_history': {
            'epochs': list(range(1, 101)),
            'train_loss': [0.8, 0.7, 0.6, 0.5, 0.4, 0.35, 0.32, 0.3, 0.28, 0.26] + [0.25] * 90,
            'val_loss': [0.85, 0.75, 0.65, 0.55, 0.45, 0.4, 0.37, 0.35, 0.33, 0.31] + [0.3] * 90,
            'train_acc': [0.3, 0.45, 0.6, 0.7, 0.78, 0.82, 0.84, 0.85, 0.86, 0.87] + [0.88] * 90,
            'val_acc': [0.25, 0.4, 0.55, 0.65, 0.73, 0.77, 0.79, 0.8, 0.81, 0.82] + [0.83] * 90
        },
        'confusion_matrix': [
            [120, 5, 3, 2, 1, 0, 1],
            [4, 115, 4, 2, 1, 0, 0],
            [2, 3, 118, 3, 1, 0, 0],
            [1, 2, 2, 116, 4, 1, 0],
            [0, 1, 1, 3, 117, 3, 1],
            [0, 0, 0, 1, 2, 115, 4],
            [1, 0, 0, 0, 1, 3, 118]
        ],
        'class_names': ['Case_Based', 'Genetic_Algorithms', 'Neural_Networks', 
                       'Probabilistic_Methods', 'Reinforcement_Learning', 
                       'Rule_Learning', 'Theory'],
        'node_embeddings': {
            'embeddings': np.random.randn(2708, 64).tolist(),
            'labels': np.random.randint(0, 7, 2708).tolist(),
            'node_ids': list(range(2708))
        }
    }
    
    return templates.TemplateResponse(
        "node_tasks/classification/results.html",
        {
            "request": request,
            "task_config": task_config,
            "task_metadata": task_metadata,
            "results": results_data
        }
    )

@router.post("/api/start_experiment")
async def start_experiment(data: Dict[str, Any]):
    """Start a new node classification experiment."""
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
            'dataset_name': experiment_params.get('dataset', 'cora')
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
            'train_loss': 0.28,
            'val_loss': 0.31,
            'train_acc': 0.87,
            'val_acc': 0.84
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
    """Get parameters for node classification."""
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
        'task_name': 'Node Classification',
        'model_name': 'GCN',
        'dataset_name': 'Cora',
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'hidden_dim': 64,
            'num_layers': 3,
            'dropout': 0.1,
            'learning_rate': 0.001,
            'epochs': 100
        },
        'metrics': {
            'accuracy': 0.85,
            'f1_score': 0.84,
            'precision': 0.86,
            'recall': 0.83
        },
        'training_history': {
            'epochs': list(range(1, 101)),
            'train_loss': [0.8, 0.7, 0.6, 0.5, 0.4, 0.35, 0.32, 0.3, 0.28, 0.26] + [0.25] * 90,
            'val_loss': [0.85, 0.75, 0.65, 0.55, 0.45, 0.4, 0.37, 0.35, 0.33, 0.31] + [0.3] * 90,
            'train_acc': [0.3, 0.45, 0.6, 0.7, 0.78, 0.82, 0.84, 0.85, 0.86, 0.87] + [0.88] * 90,
            'val_acc': [0.25, 0.4, 0.55, 0.65, 0.73, 0.77, 0.79, 0.8, 0.81, 0.82] + [0.83] * 90
        },
        'confusion_matrix': [
            [120, 5, 3, 2, 1, 0, 1],
            [4, 115, 4, 2, 1, 0, 0],
            [2, 3, 118, 3, 1, 0, 0],
            [1, 2, 2, 116, 4, 1, 0],
            [0, 1, 1, 3, 117, 3, 1],
            [0, 0, 0, 1, 2, 115, 4],
            [1, 0, 0, 0, 1, 3, 118]
        ],
        'class_names': ['Case_Based', 'Genetic_Algorithms', 'Neural_Networks', 
                       'Probabilistic_Methods', 'Reinforcement_Learning', 
                       'Rule_Learning', 'Theory']
    }
    
    return results_data

@router.get("/api/get_node_embeddings/{experiment_id}")
async def get_node_embeddings(experiment_id: str):
    """Get node embeddings for visualization."""
    # Mock embeddings data
    embeddings_data = {
        'experiment_id': experiment_id,
        'embeddings': np.random.randn(2708, 64).tolist(),
        'labels': np.random.randint(0, 7, 2708).tolist(),
        'node_ids': list(range(2708)),
        'class_names': ['Case_Based', 'Genetic_Algorithms', 'Neural_Networks', 
                       'Probabilistic_Methods', 'Reinforcement_Learning', 
                       'Rule_Learning', 'Theory']
    }
    
    return embeddings_data

@router.get("/api/export_results/{experiment_id}")
async def export_results(experiment_id: str, format_type: str = "json"):
    """Export experiment results."""
    # Get results data
    results_data = {
        'experiment_id': experiment_id,
        'task_name': 'Node Classification',
        'model_name': 'GCN',
        'dataset_name': 'Cora',
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'accuracy': 0.85,
            'f1_score': 0.84,
            'precision': 0.86,
            'recall': 0.83
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
            headers={'Content-Disposition': f'attachment; filename=node_classification_results_{experiment_id}.csv'}
        )
    else:
        # Return JSON
        return results_data

@router.get("/api/get_attention_weights/{experiment_id}")
async def get_attention_weights(experiment_id: str):
    """Get attention weights for GAT models."""
    # Mock attention weights data
    attention_data = {
        'experiment_id': experiment_id,
        'attention_weights': np.random.rand(100, 8).tolist(),  # 100 nodes, 8 attention heads
        'node_ids': list(range(100)),
        'head_names': [f'Head_{i+1}' for i in range(8)]
    }
    
    return attention_data

@router.get("/api/get_feature_importance/{experiment_id}")
async def get_feature_importance(experiment_id: str):
    """Get feature importance scores."""
    # Mock feature importance data
    feature_importance = {
        'experiment_id': experiment_id,
        'feature_names': [f'Feature_{i}' for i in range(1433)],
        'importance_scores': np.random.rand(1433).tolist(),
        'top_features': [f'Feature_{i}' for i in range(20)]
    }
    
    return feature_importance
