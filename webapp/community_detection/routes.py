"""
Community Detection Routes
This module contains all routes for community detection tasks.
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
from webapp.config.tasks import get_task_config, get_task_metadata
from webapp.config.parameters import get_task_parameters, get_task_default_params
from webapp.config.models import get_models_for_task, get_model_config

# Create FastAPI router
router = APIRouter(prefix="/community_detection", tags=["Community Detection"])

# Task configuration
TASK_NAME = 'community_detection'
task_config = get_task_config(TASK_NAME)
task_metadata = get_task_metadata(TASK_NAME)

# Setup templates
templates = Jinja2Templates(directory="webapp/templates")

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page for community detection."""
    return templates.TemplateResponse(
        "community_detection/index.html",
        {
            "request": request,
            "task_config": task_config,
            "task_metadata": task_metadata
        }
    )

@router.get("/experiment", response_class=HTMLResponse)
async def experiment(request: Request):
    """Experiment page for community detection."""
    # Get parameters and defaults
    parameters = get_task_parameters(TASK_NAME)
    default_params = get_task_default_params(TASK_NAME)
    supported_models = get_models_for_task(TASK_NAME)
    
    return templates.TemplateResponse(
        "community_detection/experiment.html",
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
    """Results page for community detection."""
    # Mock results data for demonstration
    results_data = {
        'experiment_id': experiment_id,
        'task_name': 'Community Detection',
        'model_name': 'Louvain',
        'dataset_name': 'Karate Club',
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'modularity': 0.412,
            'conductance': 0.234,
            'coverage': 0.856,
            'performance': 0.789,
            'silhouette_score': 0.623,
            'nmi': 0.745
        },
        'community_stats': {
            'num_communities': 4,
            'avg_community_size': 8.5,
            'largest_community': 12,
            'smallest_community': 5,
            'community_distribution': [12, 8, 7, 5]
        },
        'node_assignments': {
            'node_ids': list(range(34)),
            'community_ids': np.random.randint(0, 4, 34).tolist(),
            'confidence_scores': np.random.uniform(0.7, 1.0, 34).tolist()
        },
        'community_network': {
            'nodes': list(range(34)),
            'edges': np.random.rand(78, 2).tolist(),
            'node_communities': np.random.randint(0, 4, 34).tolist()
        }
    }
    
    return templates.TemplateResponse(
        "community_detection/results.html",
        {
            "request": request,
            "task_config": task_config,
            "task_metadata": task_metadata,
            "results": results_data
        }
    )

@router.post("/api/start_experiment")
async def start_experiment(data: Dict[str, Any]):
    """Start a new community detection experiment."""
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
            'model_name': experiment_params.get('model', 'louvain'),
            'dataset_name': experiment_params.get('dataset', 'karate')
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
        'current_iteration': 65,
        'total_iterations': 100,
        'current_metrics': {
            'modularity': 0.38,
            'num_communities': 4,
            'avg_community_size': 8.5
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
    """Get parameters for community detection."""
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
        'task_name': 'Community Detection',
        'model_name': 'Louvain',
        'dataset_name': 'Karate Club',
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'modularity': 0.412,
            'conductance': 0.234,
            'coverage': 0.856,
            'performance': 0.789,
            'silhouette_score': 0.623,
            'nmi': 0.745
        },
        'community_stats': {
            'num_communities': 4,
            'avg_community_size': 8.5,
            'largest_community': 12,
            'smallest_community': 5,
            'community_distribution': [12, 8, 7, 5]
        }
    }
    
    return results_data

@router.get("/api/get_community_assignments/{experiment_id}")
async def get_community_assignments(experiment_id: str):
    """Get community assignments for nodes."""
    # Mock community assignments data
    community_assignments = {
        'experiment_id': experiment_id,
        'node_ids': list(range(34)),
        'community_ids': np.random.randint(0, 4, 34).tolist(),
        'confidence_scores': np.random.uniform(0.7, 1.0, 34).tolist()
    }
    
    return community_assignments

@router.get("/api/get_community_network/{experiment_id}")
async def get_community_network(experiment_id: str):
    """Get community network visualization data."""
    # Mock community network data
    community_network = {
        'experiment_id': experiment_id,
        'nodes': list(range(34)),
        'edges': np.random.rand(78, 2).tolist(),
        'node_communities': np.random.randint(0, 4, 34).tolist()
    }
    
    return community_network

@router.get("/api/get_community_metrics/{experiment_id}")
async def get_community_metrics(experiment_id: str):
    """Get detailed community metrics."""
    # Mock community metrics data
    community_metrics = {
        'experiment_id': experiment_id,
        'modularity': 0.412,
        'conductance': 0.234,
        'coverage': 0.856,
        'performance': 0.789,
        'silhouette_score': 0.623,
        'nmi': 0.745,
        'community_sizes': [12, 8, 7, 5],
        'community_densities': [0.85, 0.72, 0.68, 0.91]
    }
    
    return community_metrics

@router.get("/api/get_community_analysis/{experiment_id}")
async def get_community_analysis(experiment_id: str):
    """Get detailed community analysis."""
    # Mock community analysis data
    community_analysis = {
        'experiment_id': experiment_id,
        'num_communities': 4,
        'avg_community_size': 8.5,
        'largest_community': 12,
        'smallest_community': 5,
        'community_distribution': [12, 8, 7, 5],
        'inter_community_edges': 15,
        'intra_community_edges': 63,
        'community_cohesion': [0.85, 0.72, 0.68, 0.91]
    }
    
    return community_analysis

@router.get("/api/export_results/{experiment_id}")
async def export_results(experiment_id: str, format_type: str = "json"):
    """Export experiment results in various formats."""
    try:
        # Mock results data
        results_data = {
            'experiment_id': experiment_id,
            'task_name': 'Community Detection',
            'model_name': 'Louvain',
            'dataset_name': 'Karate Club',
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'modularity': 0.412,
                'conductance': 0.234,
                'coverage': 0.856,
                'performance': 0.789
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

@router.get("/api/get_community_visualization/{experiment_id}")
async def get_community_visualization(experiment_id: str):
    """Get community visualization data."""
    # Mock community visualization data
    community_visualization = {
        'experiment_id': experiment_id,
        'nodes': list(range(34)),
        'edges': np.random.rand(78, 2).tolist(),
        'node_communities': np.random.randint(0, 4, 34).tolist(),
        'node_positions': np.random.rand(34, 2).tolist(),
        'community_colors': ['#ff0000', '#00ff00', '#0000ff', '#ffff00']
    }
    
    return community_visualization

@router.get("/api/get_community_comparison/{experiment_id}")
async def get_community_comparison(experiment_id: str):
    """Get comparison between different community detection algorithms."""
    # Mock community comparison data
    community_comparison = {
        'experiment_id': experiment_id,
        'algorithms': ['Louvain', 'Girvan-Newman', 'Label Propagation', 'Spectral Clustering'],
        'modularity_scores': [0.412, 0.398, 0.356, 0.423],
        'num_communities': [4, 5, 3, 4],
        'execution_times': [0.15, 2.34, 0.08, 1.23],
        'nmi_scores': [0.745, 0.712, 0.689, 0.756]
    }
    
    return community_comparison
