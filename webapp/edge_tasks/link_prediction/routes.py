"""
Link Prediction Routes
This module contains all routes for link prediction tasks using FastAPI.
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
router = APIRouter(prefix="/link_prediction", tags=["Link Prediction"])

# Task configuration
TASK_NAME = 'link_prediction'
task_config = get_task_config(TASK_NAME)
task_metadata = get_task_metadata(TASK_NAME)

# Setup templates
templates = Jinja2Templates(directory="templates")

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page for link prediction."""
    return templates.TemplateResponse(
        "edge_tasks/link_prediction/index.html",
        {
            "request": request,
            "task_config": task_config,
            "task_metadata": task_metadata
        }
    )

@router.get("/experiment", response_class=HTMLResponse)
async def experiment(request: Request):
    """Experiment page for link prediction."""
    # Get parameters and defaults
    parameters = get_task_parameters(TASK_NAME)
    default_params = get_task_default_params(TASK_NAME)
    supported_models = get_models_for_task(TASK_NAME)
    
    return templates.TemplateResponse(
        "edge_tasks/link_prediction/experiment.html",
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
    """Results page for link prediction."""
    # Mock results data for demonstration
    results_data = {
        'experiment_id': experiment_id,
        'task_name': 'Link Prediction',
        'model_name': 'VGAE',
        'dataset_name': 'Cora',
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'auc': 0.923,
            'ap': 0.918,
            'precision_at_k': 0.856,
            'recall_at_k': 0.789,
            'f1_score': 0.821,
            'hits_at_k': 0.892
        },
        'training_history': {
            'epochs': list(range(1, 101)),
            'train_loss': [0.8, 0.7, 0.6, 0.5, 0.4, 0.35, 0.32, 0.3, 0.28, 0.26] + [0.25] * 90,
            'val_loss': [0.85, 0.75, 0.65, 0.55, 0.45, 0.4, 0.37, 0.35, 0.33, 0.31] + [0.3] * 90,
            'train_auc': [0.5, 0.6, 0.7, 0.8, 0.85, 0.88, 0.9, 0.91, 0.92, 0.92] + [0.92] * 90,
            'val_auc': [0.45, 0.55, 0.65, 0.75, 0.8, 0.83, 0.85, 0.86, 0.87, 0.88] + [0.89] * 90
        },
        'roc_curve': {
            'fpr': np.linspace(0, 1, 100).tolist(),
            'tpr': (np.linspace(0, 1, 100) ** 0.8).tolist(),
            'thresholds': np.linspace(1, 0, 100).tolist()
        },
        'precision_recall_curve': {
            'precision': (np.linspace(0.5, 1, 100) ** 0.9).tolist(),
            'recall': np.linspace(0, 1, 100).tolist(),
            'thresholds': np.linspace(1, 0, 100).tolist()
        },
        'link_predictions': {
            'positive_links': np.random.rand(500, 2).tolist(),
            'negative_links': np.random.rand(500, 2).tolist(),
            'positive_scores': np.random.uniform(0.7, 1.0, 500).tolist(),
            'negative_scores': np.random.uniform(0.0, 0.3, 500).tolist()
        }
    }
    
    return templates.TemplateResponse(
        "edge_tasks/link_prediction/results.html",
        {
            "request": request,
            "task_config": task_config,
            "task_metadata": task_metadata,
            "results": results_data
        }
    )

@router.post("/api/start_experiment")
async def start_experiment(data: Dict[str, Any]):
    """Start a new link prediction experiment."""
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
            'model_name': experiment_params.get('model', 'vgae'),
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
            'train_loss': 0.26,
            'val_loss': 0.3,
            'train_auc': 0.92,
            'val_auc': 0.89
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
    """Get parameters for link prediction."""
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
        'task_name': 'Link Prediction',
        'model_name': 'VGAE',
        'dataset_name': 'Cora',
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'hidden_dim': 64,
            'num_layers': 3,
            'dropout': 0.1,
            'learning_rate': 0.001,
            'epochs': 100,
            'latent_dim': 32
        },
        'metrics': {
            'auc': 0.923,
            'ap': 0.918,
            'precision_at_k': 0.856,
            'recall_at_k': 0.789,
            'f1_score': 0.821,
            'hits_at_k': 0.892
        },
        'training_history': {
            'epochs': list(range(1, 101)),
            'train_loss': [0.8, 0.7, 0.6, 0.5, 0.4, 0.35, 0.32, 0.3, 0.28, 0.26] + [0.25] * 90,
            'val_loss': [0.85, 0.75, 0.65, 0.55, 0.45, 0.4, 0.37, 0.35, 0.33, 0.31] + [0.3] * 90,
            'train_auc': [0.5, 0.6, 0.7, 0.8, 0.85, 0.88, 0.9, 0.91, 0.92, 0.92] + [0.92] * 90,
            'val_auc': [0.45, 0.55, 0.65, 0.75, 0.8, 0.83, 0.85, 0.86, 0.87, 0.88] + [0.89] * 90
        }
    }
    
    return results_data

@router.get("/api/get_roc_curve/{experiment_id}")
async def get_roc_curve(experiment_id: str):
    """Get ROC curve data."""
    # Mock ROC curve data
    roc_data = {
        'experiment_id': experiment_id,
        'fpr': np.linspace(0, 1, 100).tolist(),
        'tpr': (np.linspace(0, 1, 100) ** 0.8).tolist(),
        'thresholds': np.linspace(1, 0, 100).tolist(),
        'auc': 0.923
    }
    
    return roc_data

@router.get("/api/get_precision_recall_curve/{experiment_id}")
async def get_precision_recall_curve(experiment_id: str):
    """Get precision-recall curve data."""
    # Mock precision-recall curve data
    pr_data = {
        'experiment_id': experiment_id,
        'precision': (np.linspace(0.5, 1, 100) ** 0.9).tolist(),
        'recall': np.linspace(0, 1, 100).tolist(),
        'thresholds': np.linspace(1, 0, 100).tolist(),
        'ap': 0.918
    }
    
    return pr_data

@router.get("/api/get_link_predictions/{experiment_id}")
async def get_link_predictions(experiment_id: str):
    """Get link predictions data."""
    # Mock link predictions data
    predictions_data = {
        'experiment_id': experiment_id,
        'positive_links': np.random.rand(500, 2).tolist(),
        'negative_links': np.random.rand(500, 2).tolist(),
        'positive_scores': np.random.uniform(0.7, 1.0, 500).tolist(),
        'negative_scores': np.random.uniform(0.0, 0.3, 500).tolist(),
        'all_predictions': np.random.rand(1000, 3).tolist()  # [node1, node2, score]
    }
    
    return predictions_data

@router.get("/api/get_embedding_visualization/{experiment_id}")
async def get_embedding_visualization(experiment_id: str):
    """Get embeddings for visualization."""
    # Mock embeddings data
    embeddings_data = {
        'experiment_id': experiment_id,
        'embeddings': np.random.randn(2708, 64).tolist(),
        'node_ids': list(range(2708)),
        'link_predictions': np.random.rand(1000, 3).tolist()  # [node1, node2, score]
    }
    
    return embeddings_data

@router.get("/api/export_results/{experiment_id}")
async def export_results(experiment_id: str, format_type: str = "json"):
    """Export experiment results."""
    # Get results data
    results_data = {
        'experiment_id': experiment_id,
        'task_name': 'Link Prediction',
        'model_name': 'VGAE',
        'dataset_name': 'Cora',
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'auc': 0.923,
            'ap': 0.918,
            'precision_at_k': 0.856,
            'recall_at_k': 0.789,
            'f1_score': 0.821,
            'hits_at_k': 0.892
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
            headers={'Content-Disposition': f'attachment; filename=link_prediction_results_{experiment_id}.csv'}
        )
    else:
        # Return JSON
        return results_data

@router.get("/api/get_top_predictions/{experiment_id}")
async def get_top_predictions(experiment_id: str, k: int = 10):
    """Get top-k link predictions."""
    # Mock top predictions data
    top_predictions = {
        'experiment_id': experiment_id,
        'k': k,
        'predictions': [
            {
                'node1': i,
                'node2': i + 100,
                'score': 0.95 - i * 0.01,
                'rank': i + 1
            }
            for i in range(k)
        ]
    }
    
    return top_predictions

@router.get("/api/get_negative_sampling_analysis/{experiment_id}")
async def get_negative_sampling_analysis(experiment_id: str):
    """Get negative sampling analysis."""
    # Mock negative sampling analysis
    analysis_data = {
        'experiment_id': experiment_id,
        'sampling_method': 'uniform',
        'negative_ratio': 1.0,
        'sample_sizes': {
            'positive': 5000,
            'negative': 5000,
            'total': 10000
        },
        'sampling_distribution': {
            'degree_based': 0.3,
            'random': 0.7
        }
    }
    
    return analysis_data

# WebSocket events for real-time updates
def register_socketio_events(socketio):
    """Register WebSocket events for real-time updates."""
    
    @socketio.on('connect', namespace='/link_prediction')
    def handle_connect():
        emit('connected', {'message': 'Connected to Link Prediction'})
    
    @socketio.on('start_training', namespace='/link_prediction')
    def handle_start_training(data):
        experiment_id = data.get('experiment_id')
        parameters = data.get('parameters', {})
        
        # Emit training progress updates
        for epoch in range(1, 101):
            progress = epoch / 100
            metrics = {
                'epoch': epoch,
                'train_loss': 0.8 * (0.9 ** epoch),
                'val_loss': 0.85 * (0.9 ** epoch),
                'train_auc': 0.5 + 0.4 * (1 - 0.9 ** epoch),
                'val_auc': 0.45 + 0.4 * (1 - 0.9 ** epoch)
            }
            
            emit('training_progress', {
                'experiment_id': experiment_id,
                'progress': progress,
                'metrics': metrics
            })
            
            # Simulate training time
            import time
            time.sleep(0.1)
        
        # Emit completion
        emit('training_complete', {
            'experiment_id': experiment_id,
            'final_metrics': {
                'auc': 0.923,
                'ap': 0.918,
                'precision_at_k': 0.856,
                'recall_at_k': 0.789
            }
        })
    
    @socketio.on('disconnect', namespace='/link_prediction')
    def handle_disconnect():
        print('Client disconnected from Link Prediction')
