"""
Graph Generation Routes
This module contains all routes for graph generation tasks.
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
router = APIRouter(prefix="/graph_generation", tags=["Graph Generation"])

# Task configuration
TASK_NAME = 'graph_generation'
task_config = get_task_config(TASK_NAME)
task_metadata = get_task_metadata(TASK_NAME)

# Setup templates
templates = Jinja2Templates(directory="templates")

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page for graph generation."""
    return templates.TemplateResponse(
        "graph_generation/index.html",
        {
            "request": request,
            "task_config": task_config,
            "task_metadata": task_metadata
        }
    )

@router.get("/experiment", response_class=HTMLResponse)
async def experiment(request: Request):
    """Experiment page for graph generation."""
    # Get parameters and defaults
    parameters = get_task_parameters(TASK_NAME)
    default_params = get_task_default_params(TASK_NAME)
    supported_models = get_models_for_task(TASK_NAME)
    
    return templates.TemplateResponse(
        "graph_generation/experiment.html",
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
    """Results page for graph generation."""
    # Mock results data for demonstration
    results_data = {
        'experiment_id': experiment_id,
        'task_name': 'Graph Generation',
        'model_name': 'GraphRNN',
        'dataset_name': 'QM9',
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'validity': 0.945,
            'uniqueness': 0.892,
            'novelty': 0.856,
            'diversity': 0.823,
            'fcd': 0.789,
            'mmd': 0.234
        },
        'generation_stats': {
            'total_generated': 1000,
            'valid_graphs': 945,
            'unique_graphs': 892,
            'novel_graphs': 856,
            'avg_nodes': 18.5,
            'avg_edges': 37.8
        },
        'generated_graphs': {
            'graph_ids': list(range(100)),
            'node_counts': np.random.randint(10, 30, 100).tolist(),
            'edge_counts': np.random.randint(20, 60, 100).tolist(),
            'validity_scores': np.random.uniform(0.8, 1.0, 100).tolist()
        },
        'quality_metrics': {
            'degree_distribution': np.random.poisson(5, 20).tolist(),
            'clustering_coefficient': np.random.uniform(0.1, 0.8, 100).tolist(),
            'diameter_distribution': np.random.randint(2, 8, 100).tolist()
        }
    }
    
    return templates.TemplateResponse(
        "graph_generation/results.html",
        {
            "request": request,
            "task_config": task_config,
            "task_metadata": task_metadata,
            "results": results_data
        }
    )

@router.post("/api/start_experiment")
async def start_experiment(data: Dict[str, Any]):
    """Start a new graph generation experiment."""
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
            'model_name': experiment_params.get('model', 'graphrnn'),
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
            'train_loss': 0.28,
            'val_loss': 0.31,
            'generation_loss': 0.25,
            'validity_rate': 0.92
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
    """Get parameters for graph generation."""
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
        'task_name': 'Graph Generation',
        'model_name': 'GraphRNN',
        'dataset_name': 'QM9',
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'validity': 0.945,
            'uniqueness': 0.892,
            'novelty': 0.856,
            'diversity': 0.823,
            'fcd': 0.789,
            'mmd': 0.234
        },
        'generation_stats': {
            'total_generated': 1000,
            'valid_graphs': 945,
            'unique_graphs': 892,
            'novel_graphs': 856,
            'avg_nodes': 18.5,
            'avg_edges': 37.8
        }
    }
    
    return results_data

@router.post("/api/generate_graphs/{experiment_id}")
async def generate_graphs(experiment_id: str, data: Dict[str, Any]):
    """Generate new graphs using a trained model."""
    try:
        num_graphs = data.get('num_graphs', 10)
        
        # Mock generation process
        generated_graphs = {
            'experiment_id': experiment_id,
            'num_requested': num_graphs,
            'num_generated': num_graphs,
            'graphs': []
        }
        
        for i in range(num_graphs):
            graph = {
                'graph_id': f"{experiment_id}_graph_{i}",
                'num_nodes': np.random.randint(10, 30),
                'num_edges': np.random.randint(20, 60),
                'validity_score': np.random.uniform(0.8, 1.0),
                'generation_time': np.random.uniform(0.1, 0.5)
            }
            generated_graphs['graphs'].append(graph)
        
        return generated_graphs
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/api/get_generated_graphs/{experiment_id}")
async def get_generated_graphs(experiment_id: str):
    """Get all generated graphs for an experiment."""
    # Mock generated graphs data
    generated_graphs = {
        'experiment_id': experiment_id,
        'graphs': [
            {
                'graph_id': f"{experiment_id}_graph_{i}",
                'num_nodes': np.random.randint(10, 30),
                'num_edges': np.random.randint(20, 60),
                'validity_score': np.random.uniform(0.8, 1.0),
                'generation_time': np.random.uniform(0.1, 0.5)
            }
            for i in range(100)
        ]
    }
    
    return generated_graphs

@router.get("/api/get_quality_metrics/{experiment_id}")
async def get_quality_metrics(experiment_id: str):
    """Get quality metrics for generated graphs."""
    # Mock quality metrics data
    quality_metrics = {
        'experiment_id': experiment_id,
        'validity': 0.945,
        'uniqueness': 0.892,
        'novelty': 0.856,
        'diversity': 0.823,
        'fcd': 0.789,
        'mmd': 0.234,
        'degree_distribution': np.random.poisson(5, 20).tolist(),
        'clustering_coefficient': np.random.uniform(0.1, 0.8, 100).tolist(),
        'diameter_distribution': np.random.randint(2, 8, 100).tolist()
    }
    
    return quality_metrics

@router.get("/api/get_graph_comparison/{experiment_id}")
async def get_graph_comparison(experiment_id: str):
    """Get comparison between generated and real graphs."""
    # Mock graph comparison data
    graph_comparison = {
        'experiment_id': experiment_id,
        'metrics': {
            'degree_distribution_kl': 0.123,
            'clustering_coefficient_kl': 0.234,
            'diameter_kl': 0.345,
            'density_kl': 0.456
        },
        'real_graphs': {
            'avg_nodes': 18.2,
            'avg_edges': 36.8,
            'avg_clustering': 0.45,
            'avg_diameter': 4.2
        },
        'generated_graphs': {
            'avg_nodes': 18.5,
            'avg_edges': 37.8,
            'avg_clustering': 0.43,
            'avg_diameter': 4.1
        }
    }
    
    return graph_comparison

@router.get("/api/export_results/{experiment_id}")
async def export_results(experiment_id: str, format_type: str = "json"):
    """Export experiment results in various formats."""
    try:
        # Mock results data
        results_data = {
            'experiment_id': experiment_id,
            'task_name': 'Graph Generation',
            'model_name': 'GraphRNN',
            'dataset_name': 'QM9',
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'validity': 0.945,
                'uniqueness': 0.892,
                'novelty': 0.856,
                'diversity': 0.823
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

@router.get("/api/get_graph_visualization/{experiment_id}")
async def get_graph_visualization(experiment_id: str):
    """Get graph visualization data."""
    # Mock graph visualization data
    graph_visualization = {
        'experiment_id': experiment_id,
        'sample_graphs': [
            {
                'graph_id': f"{experiment_id}_sample_{i}",
                'nodes': list(range(np.random.randint(10, 20))),
                'edges': np.random.rand(np.random.randint(15, 40), 2).tolist(),
                'node_positions': np.random.rand(np.random.randint(10, 20), 2).tolist()
            }
            for i in range(5)
        ]
    }
    
    return graph_visualization

@router.get("/api/get_generation_analysis/{experiment_id}")
async def get_generation_analysis(experiment_id: str):
    """Get detailed generation analysis."""
    # Mock generation analysis data
    generation_analysis = {
        'experiment_id': experiment_id,
        'total_generated': 1000,
        'valid_graphs': 945,
        'unique_graphs': 892,
        'novel_graphs': 856,
        'avg_nodes': 18.5,
        'avg_edges': 37.8,
        'generation_times': np.random.uniform(0.1, 0.5, 100).tolist(),
        'validity_distribution': np.random.uniform(0.8, 1.0, 100).tolist(),
        'node_count_distribution': np.random.randint(10, 30, 100).tolist(),
        'edge_count_distribution': np.random.randint(20, 60, 100).tolist()
    }
    
    return generation_analysis
