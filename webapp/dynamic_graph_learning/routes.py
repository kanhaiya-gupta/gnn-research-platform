"""
Dynamic Graph Learning Routes
This module contains all routes for dynamic graph learning tasks.
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
router = APIRouter(prefix="/dynamic_graph_learning", tags=["Dynamic Graph Learning"])

# Task configuration
TASK_NAME = 'dynamic_graph_learning'
task_config = get_task_config(TASK_NAME)
task_metadata = get_task_metadata(TASK_NAME)

# Setup templates
templates = Jinja2Templates(directory="templates")

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page for dynamic graph learning."""
    return templates.TemplateResponse(
        "dynamic_graph_learning/index.html",
        {
            "request": request,
            "task_config": task_config,
            "task_metadata": task_metadata
        }
    )

@router.get("/experiment", response_class=HTMLResponse)
async def experiment(request: Request):
    """Experiment page for dynamic graph learning."""
    # Get parameters and defaults
    parameters = get_task_parameters(TASK_NAME)
    default_params = get_task_default_params(TASK_NAME)
    supported_models = get_models_for_task(TASK_NAME)
    
    return templates.TemplateResponse(
        "dynamic_graph_learning/experiment.html",
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
    """Results page for dynamic graph learning."""
    # Mock results data for demonstration
    results_data = {
        'experiment_id': experiment_id,
        'task_name': 'Dynamic Graph Learning',
        'model_name': 'DySAT',
        'dataset_name': 'Reddit',
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'link_prediction_auc': 0.856,
            'node_classification_acc': 0.823,
            'temporal_consistency': 0.789,
            'prediction_horizon': 0.745,
            'evolution_accuracy': 0.812,
            'temporal_embedding_quality': 0.834
        },
        'temporal_stats': {
            'num_snapshots': 50,
            'time_span': '365 days',
            'snapshot_interval': '7 days',
            'total_nodes': 1000,
            'total_edges': 5000,
            'avg_edges_per_snapshot': 100
        },
        'temporal_evolution': {
            'timestamps': list(range(50)),
            'node_counts': np.random.randint(950, 1050, 50).tolist(),
            'edge_counts': np.random.randint(90, 110, 50).tolist(),
            'density_evolution': np.random.uniform(0.08, 0.12, 50).tolist()
        },
        'prediction_results': {
            'future_timestamps': list(range(51, 61)),
            'predicted_edges': np.random.randint(85, 115, 10).tolist(),
            'predicted_nodes': np.random.randint(945, 1055, 10).tolist(),
            'confidence_scores': np.random.uniform(0.7, 0.9, 10).tolist()
        }
    }
    
    return templates.TemplateResponse(
        "dynamic_graph_learning/results.html",
        {
            "request": request,
            "task_config": task_config,
            "task_metadata": task_metadata,
            "results": results_data
        }
    )

@router.post("/api/start_experiment")
async def start_experiment(data: Dict[str, Any]):
    """Start a new dynamic graph learning experiment."""
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
            'model_name': experiment_params.get('model', 'dysat'),
            'dataset_name': experiment_params.get('dataset', 'reddit')
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
        'current_snapshot': 32,
        'total_snapshots': 50,
        'current_metrics': {
            'train_loss': 0.28,
            'val_loss': 0.31,
            'temporal_loss': 0.25,
            'prediction_accuracy': 0.82
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
    """Get parameters for dynamic graph learning."""
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
        'task_name': 'Dynamic Graph Learning',
        'model_name': 'DySAT',
        'dataset_name': 'Reddit',
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'link_prediction_auc': 0.856,
            'node_classification_acc': 0.823,
            'temporal_consistency': 0.789,
            'prediction_horizon': 0.745,
            'evolution_accuracy': 0.812,
            'temporal_embedding_quality': 0.834
        },
        'temporal_stats': {
            'num_snapshots': 50,
            'time_span': '365 days',
            'snapshot_interval': '7 days',
            'total_nodes': 1000,
            'total_edges': 5000,
            'avg_edges_per_snapshot': 100
        }
    }
    
    return results_data

@router.get("/api/get_temporal_evolution/{experiment_id}")
async def get_temporal_evolution(experiment_id: str):
    """Get temporal evolution data."""
    # Mock temporal evolution data
    temporal_evolution = {
        'experiment_id': experiment_id,
        'timestamps': list(range(50)),
        'node_counts': np.random.randint(950, 1050, 50).tolist(),
        'edge_counts': np.random.randint(90, 110, 50).tolist(),
        'density_evolution': np.random.uniform(0.08, 0.12, 50).tolist(),
        'clustering_evolution': np.random.uniform(0.1, 0.3, 50).tolist()
    }
    
    return temporal_evolution

@router.get("/api/get_prediction_results/{experiment_id}")
async def get_prediction_results(experiment_id: str):
    """Get prediction results for future timestamps."""
    # Mock prediction results data
    prediction_results = {
        'experiment_id': experiment_id,
        'future_timestamps': list(range(51, 61)),
        'predicted_edges': np.random.randint(85, 115, 10).tolist(),
        'predicted_nodes': np.random.randint(945, 1055, 10).tolist(),
        'confidence_scores': np.random.uniform(0.7, 0.9, 10).tolist(),
        'prediction_accuracy': np.random.uniform(0.75, 0.85, 10).tolist()
    }
    
    return prediction_results

@router.get("/api/get_temporal_embeddings/{experiment_id}")
async def get_temporal_embeddings(experiment_id: str):
    """Get temporal embeddings for nodes."""
    # Mock temporal embeddings data
    temporal_embeddings = {
        'experiment_id': experiment_id,
        'node_ids': list(range(100)),
        'timestamps': list(range(50)),
        'embeddings': np.random.randn(100, 50, 64).tolist(),
        'embedding_quality': np.random.uniform(0.7, 0.9, 100).tolist()
    }
    
    return temporal_embeddings

@router.get("/api/get_snapshot_analysis/{experiment_id}")
async def get_snapshot_analysis(experiment_id: str):
    """Get analysis of individual snapshots."""
    # Mock snapshot analysis data
    snapshot_analysis = {
        'experiment_id': experiment_id,
        'snapshots': [
            {
                'snapshot_id': i,
                'timestamp': i * 7,  # 7 days interval
                'num_nodes': np.random.randint(950, 1050),
                'num_edges': np.random.randint(90, 110),
                'density': np.random.uniform(0.08, 0.12),
                'clustering_coefficient': np.random.uniform(0.1, 0.3),
                'avg_degree': np.random.uniform(8, 12)
            }
            for i in range(50)
        ]
    }
    
    return snapshot_analysis

@router.get("/api/export_results/{experiment_id}")
async def export_results(experiment_id: str, format_type: str = "json"):
    """Export experiment results in various formats."""
    try:
        # Mock results data
        results_data = {
            'experiment_id': experiment_id,
            'task_name': 'Dynamic Graph Learning',
            'model_name': 'DySAT',
            'dataset_name': 'Reddit',
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'link_prediction_auc': 0.856,
                'node_classification_acc': 0.823,
                'temporal_consistency': 0.789,
                'prediction_horizon': 0.745
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

@router.get("/api/get_temporal_visualization/{experiment_id}")
async def get_temporal_visualization(experiment_id: str):
    """Get temporal visualization data."""
    # Mock temporal visualization data
    temporal_visualization = {
        'experiment_id': experiment_id,
        'timestamps': list(range(50)),
        'node_counts': np.random.randint(950, 1050, 50).tolist(),
        'edge_counts': np.random.randint(90, 110, 50).tolist(),
        'density_evolution': np.random.uniform(0.08, 0.12, 50).tolist(),
        'clustering_evolution': np.random.uniform(0.1, 0.3, 50).tolist(),
        'snapshot_graphs': [
            {
                'snapshot_id': i,
                'nodes': list(range(np.random.randint(950, 1050))),
                'edges': np.random.rand(np.random.randint(90, 110), 2).tolist()
            }
            for i in range(10)  # Sample 10 snapshots
        ]
    }
    
    return temporal_visualization

@router.get("/api/get_evolution_patterns/{experiment_id}")
async def get_evolution_patterns(experiment_id: str):
    """Get patterns in graph evolution."""
    # Mock evolution patterns data
    evolution_patterns = {
        'experiment_id': experiment_id,
        'growth_patterns': {
            'linear_growth': 0.45,
            'exponential_growth': 0.25,
            'periodic_growth': 0.20,
            'stable_growth': 0.10
        },
        'community_evolution': {
            'merging_communities': 0.35,
            'splitting_communities': 0.25,
            'stable_communities': 0.40
        },
        'temporal_metrics': {
            'avg_growth_rate': 0.05,
            'volatility': 0.12,
            'predictability': 0.78,
            'seasonality': 0.34
        },
        'evolution_timeline': {
            'timestamps': list(range(50)),
            'growth_rates': np.random.uniform(0.02, 0.08, 50).tolist(),
            'stability_scores': np.random.uniform(0.6, 0.9, 50).tolist(),
            'complexity_scores': np.random.uniform(0.3, 0.7, 50).tolist()
        }
    }
    
    return evolution_patterns
