"""
Graph Classification Routes
This module contains all routes for graph classification tasks.
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
router = APIRouter(prefix="/graph_classification", tags=["Graph Classification"])

# Task configuration
TASK_NAME = 'graph_classification'
task_config = get_task_config(TASK_NAME)
task_metadata = get_task_metadata(TASK_NAME)

# Setup templates
templates = Jinja2Templates(directory="templates")

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page for graph classification."""
    return templates.TemplateResponse(
        "graph_tasks/classification/index.html",
        {
            "request": request,
            "task_config": task_config,
            "task_metadata": task_metadata
        }
    )

@router.get("/experiment", response_class=HTMLResponse)
async def experiment(request: Request):
    """Experiment page for graph classification."""
    # Get parameters and defaults
    parameters = get_task_parameters(TASK_NAME)
    default_params = get_task_default_params(TASK_NAME)
    supported_models = get_models_for_task(TASK_NAME)
    
    return templates.TemplateResponse(
        "graph_tasks/classification/experiment.html",
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
    """Results page for graph classification."""
    # Mock results data for demonstration
    results_data = {
        'experiment_id': experiment_id,
        'task_name': 'Graph Classification',
        'model_name': 'GraphSAGE',
        'dataset_name': 'MUTAG',
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'accuracy': 0.89,
            'f1_score': 0.88,
            'precision': 0.90,
            'recall': 0.87,
            'micro_f1': 0.89,
            'macro_f1': 0.88
        },
        'training_history': {
            'epochs': list(range(1, 101)),
            'train_loss': [0.8, 0.7, 0.6, 0.5, 0.4, 0.35, 0.32, 0.3, 0.28, 0.26] + [0.25] * 90,
            'val_loss': [0.85, 0.75, 0.65, 0.55, 0.45, 0.4, 0.37, 0.35, 0.33, 0.31] + [0.3] * 90,
            'train_acc': [0.3, 0.45, 0.6, 0.7, 0.78, 0.82, 0.84, 0.85, 0.86, 0.87] + [0.88] * 90,
            'val_acc': [0.25, 0.4, 0.55, 0.65, 0.73, 0.77, 0.79, 0.8, 0.81, 0.82] + [0.83] * 90
        },
        'confusion_matrix': [
            [85, 5, 3, 2],
            [4, 82, 4, 2],
            [2, 3, 88, 3],
            [1, 2, 2, 86]
        ],
        'class_names': ['Mutagenic', 'Non-mutagenic', 'Carcinogenic', 'Non-carcinogenic'],
        'graph_predictions': {
            'graph_ids': list(range(100)),
            'true_labels': np.random.randint(0, 4, 100).tolist(),
            'predicted_labels': np.random.randint(0, 4, 100).tolist(),
            'confidence_scores': np.random.uniform(0.7, 1.0, 100).tolist()
        }
    }
    
    return templates.TemplateResponse(
        "graph_tasks/classification/results.html",
        {
            "request": request,
            "task_config": task_config,
            "task_metadata": task_metadata,
            "results": results_data
        }
    )

@router.post("/api/start_experiment")
async def start_experiment(data: Dict[str, Any]):
    """Start a new graph classification experiment."""
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
            'model_name': experiment_params.get('model', 'graphsage'),
            'dataset_name': experiment_params.get('dataset', 'mutag')
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
    """Get parameters for graph classification."""
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
        'task_name': 'Graph Classification',
        'model_name': 'GraphSAGE',
        'dataset_name': 'MUTAG',
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'accuracy': 0.89,
            'f1_score': 0.88,
            'precision': 0.90,
            'recall': 0.87,
            'micro_f1': 0.89,
            'macro_f1': 0.88
        },
        'training_history': {
            'epochs': list(range(1, 101)),
            'train_loss': [0.8, 0.7, 0.6, 0.5, 0.4, 0.35, 0.32, 0.3, 0.28, 0.26] + [0.25] * 90,
            'val_loss': [0.85, 0.75, 0.65, 0.55, 0.45, 0.4, 0.37, 0.35, 0.33, 0.31] + [0.3] * 90,
            'train_acc': [0.3, 0.45, 0.6, 0.7, 0.78, 0.82, 0.84, 0.85, 0.86, 0.87] + [0.88] * 90,
            'val_acc': [0.25, 0.4, 0.55, 0.65, 0.73, 0.77, 0.79, 0.8, 0.81, 0.82] + [0.83] * 90
        }
    }
    
    return results_data

@router.get("/api/get_graph_predictions/{experiment_id}")
async def get_graph_predictions(experiment_id: str):
    """Get graph predictions for visualization."""
    # Mock graph predictions data
    graph_predictions = {
        'experiment_id': experiment_id,
        'graph_ids': list(range(100)),
        'true_labels': np.random.randint(0, 4, 100).tolist(),
        'predicted_labels': np.random.randint(0, 4, 100).tolist(),
        'confidence_scores': np.random.uniform(0.7, 1.0, 100).tolist()
    }
    
    return graph_predictions

@router.get("/api/get_class_performance/{experiment_id}")
async def get_class_performance(experiment_id: str):
    """Get performance metrics for each class."""
    # Mock class performance data
    class_performance = {
        'experiment_id': experiment_id,
        'class_names': ['Mutagenic', 'Non-mutagenic', 'Carcinogenic', 'Non-carcinogenic'],
        'precision': [0.91, 0.89, 0.92, 0.88],
        'recall': [0.89, 0.87, 0.90, 0.86],
        'f1_score': [0.90, 0.88, 0.91, 0.87]
    }
    
    return class_performance

@router.get("/api/get_graph_embeddings/{experiment_id}")
async def get_graph_embeddings(experiment_id: str):
    """Get graph embeddings for visualization."""
    # Mock graph embeddings data
    graph_embeddings = {
        'experiment_id': experiment_id,
        'embeddings': np.random.randn(100, 64).tolist(),
        'graph_ids': list(range(100)),
        'labels': np.random.randint(0, 4, 100).tolist()
    }
    
    return graph_embeddings

@router.get("/api/get_graph_statistics/{experiment_id}")
async def get_graph_statistics(experiment_id: str):
    """Get graph statistics and properties."""
    # Mock graph statistics data
    graph_statistics = {
        'experiment_id': experiment_id,
        'total_graphs': 100,
        'avg_nodes': 25.3,
        'avg_edges': 52.7,
        'node_feature_dim': 7,
        'edge_feature_dim': 4,
        'class_distribution': {
            'Mutagenic': 25,
            'Non-mutagenic': 25,
            'Carcinogenic': 25,
            'Non-carcinogenic': 25
        },
        'graph_properties': {
            'density': np.random.uniform(0.1, 0.5, 100).tolist(),
            'clustering_coeff': np.random.uniform(0.0, 1.0, 100).tolist(),
            'diameter': np.random.randint(3, 10, 100).tolist()
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
            'task_name': 'Graph Classification',
            'model_name': 'GraphSAGE',
            'dataset_name': 'MUTAG',
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'accuracy': 0.89,
                'f1_score': 0.88,
                'precision': 0.90,
                'recall': 0.87
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
        'feature_names': ['node_degree', 'clustering_coeff', 'betweenness', 'eigenvector_centrality'],
        'importance_scores': [0.30, 0.25, 0.25, 0.20]
    }
    
    return feature_importance

@router.get("/api/get_attention_weights/{experiment_id}")
async def get_attention_weights(experiment_id: str):
    """Get attention weights for attention-based models."""
    # Mock attention weights data
    attention_weights = {
        'experiment_id': experiment_id,
        'attention_weights': np.random.uniform(0, 1, (100, 8)).tolist(),  # 100 graphs, 8 attention heads
        'graph_ids': list(range(100))
    }
    
    return attention_weights
