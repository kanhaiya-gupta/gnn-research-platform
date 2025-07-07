"""
Node Classification Routes
This module contains all routes for node classification tasks using FastAPI.
"""

from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse, Response
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
router = APIRouter(tags=["Node Classification"])

# Create API router
api_router = APIRouter(prefix="/api", tags=["Node Classification API"])

# Task configuration
TASK_NAME = 'node_classification'
task_config = get_task_config(TASK_NAME)
task_metadata = get_task_metadata(TASK_NAME)

# Setup templates
templates = Jinja2Templates(directory="webapp/templates")

# Results storage directory
RESULTS_DIR = "webapp/static/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def save_experiment_results(experiment_id: str, results: Dict[str, Any]):
    """Save experiment results to file."""
    filepath = os.path.join(RESULTS_DIR, f"experiment_{experiment_id}.json")
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)

def load_experiment_results(experiment_id: str) -> Optional[Dict[str, Any]]:
    """Load experiment results from file."""
    filepath = os.path.join(RESULTS_DIR, f"experiment_{experiment_id}.json")
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def get_experiment_results(experiment_id: str) -> Optional[Dict[str, Any]]:
    """Get experiment results, returning None if not found."""
    return load_experiment_results(experiment_id)

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
    # Try to load real results
    results_data = load_experiment_results(experiment_id)
    
    if results_data is None:
        # Return error page if no results found
        return templates.TemplateResponse(
            "node_tasks/classification/results.html",
            {
                "request": request,
                "task_config": task_config,
                "task_metadata": task_metadata,
                "error": f"No results found for experiment {experiment_id}",
                "results": None
            }
        )
    
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
        
        # Save initial experiment data
        save_experiment_results(experiment_id, experiment_data)
        
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
    results_data = load_experiment_results(experiment_id)
    
    if results_data is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # Return actual status from saved results
    return {
        'experiment_id': experiment_id,
        'status': results_data.get('status', 'unknown'),
        'progress': results_data.get('progress', 0),
        'current_epoch': results_data.get('current_epoch', 0),
        'total_epochs': results_data.get('total_epochs', 0),
        'current_metrics': results_data.get('current_metrics', {}),
        'estimated_time_remaining': results_data.get('estimated_time_remaining', '00:00:00')
    }

@router.post("/api/stop_experiment/{experiment_id}")
async def stop_experiment(experiment_id: str):
    """Stop a running experiment."""
    try:
        results_data = load_experiment_results(experiment_id)
        
        if results_data is None:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        # Update status to stopped
        results_data['status'] = 'stopped'
        results_data['stop_time'] = datetime.now().isoformat()
        
        # Save updated results
        save_experiment_results(experiment_id, results_data)
        
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
        parameters = get_task_parameters(TASK_NAME)
        experiment_params = data.get('parameters', {})
        
        # Basic validation
        errors = []
        
        # Check required parameters
        for param_name, param_config in parameters.items():
            if param_config.get('required', False) and param_name not in experiment_params:
                errors.append(f"Required parameter '{param_name}' is missing")
        
        # Validate parameter types and ranges
        for param_name, param_value in experiment_params.items():
            if param_name in parameters:
                param_config = parameters[param_name]
                
                # Type validation
                expected_type = param_config.get('type', 'string')
                if expected_type == 'number' and not isinstance(param_value, (int, float)):
                    errors.append(f"Parameter '{param_name}' must be a number")
                elif expected_type == 'integer' and not isinstance(param_value, int):
                    errors.append(f"Parameter '{param_name}' must be an integer")
                
                # Range validation
                if 'min' in param_config and param_value < param_config['min']:
                    errors.append(f"Parameter '{param_name}' must be >= {param_config['min']}")
                if 'max' in param_config and param_value > param_config['max']:
                    errors.append(f"Parameter '{param_name}' must be <= {param_config['max']}")
        
        if errors:
            return {
                'valid': False,
                'errors': errors
            }
        
        return {
            'valid': True,
            'message': 'Parameters are valid'
        }
        
    except Exception as e:
        return {
            'valid': False,
            'errors': [str(e)]
        }

@router.get("/api/get_experiment_results/{experiment_id}")
async def get_experiment_results_api(experiment_id: str):
    """Get experiment results via API."""
    results_data = load_experiment_results(experiment_id)
    
    if results_data is None:
        raise HTTPException(status_code=404, detail="Experiment results not found")
    
    return results_data

@router.get("/api/get_node_embeddings/{experiment_id}")
async def get_node_embeddings(experiment_id: str):
    """Get node embeddings for visualization."""
    results_data = load_experiment_results(experiment_id)
    
    if results_data is None:
        raise HTTPException(status_code=404, detail="Experiment results not found")
    
    # Return embeddings if available
    embeddings = results_data.get('node_embeddings', {})
    if not embeddings:
        raise HTTPException(status_code=404, detail="Node embeddings not available")
    
    return embeddings

@router.post("/api/save_model/{experiment_id}")
async def save_model(experiment_id: str, model_data: dict):
    """Save a trained model."""
    try:
        results_data = load_experiment_results(experiment_id)
        
        if results_data is None:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        # Add model data to results
        results_data['saved_model'] = {
            'model_data': model_data,
            'save_time': datetime.now().isoformat(),
            'model_path': f"models/{experiment_id}_model.pth"
        }
        
        # Save updated results
        save_experiment_results(experiment_id, results_data)
        
        return {
            'success': True,
            'message': f'Model saved successfully for experiment {experiment_id}'
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/api/list_saved_models")
async def list_saved_models():
    """List all saved models."""
    try:
        saved_models = []
        
        # Scan results directory for saved models
        for filename in os.listdir(RESULTS_DIR):
            if filename.startswith('experiment_') and filename.endswith('.json'):
                experiment_id = filename.replace('experiment_', '').replace('.json', '')
                results_data = load_experiment_results(experiment_id)
                
                if results_data and 'saved_model' in results_data:
                    saved_models.append({
                        'experiment_id': experiment_id,
                        'model_name': results_data.get('model_name', 'Unknown'),
                        'dataset_name': results_data.get('dataset_name', 'Unknown'),
                        'save_time': results_data['saved_model']['save_time'],
                        'model_path': results_data['saved_model']['model_path']
                    })
        
        return {
            'saved_models': saved_models,
            'count': len(saved_models)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/api/load_model/{experiment_id}")
async def load_model(experiment_id: str):
    """Load a saved model."""
    try:
        results_data = load_experiment_results(experiment_id)
        
        if results_data is None:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        if 'saved_model' not in results_data:
            raise HTTPException(status_code=404, detail="No saved model found")
        
        return {
            'success': True,
            'model_data': results_data['saved_model']['model_data'],
            'model_path': results_data['saved_model']['model_path']
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/api/delete_model/{experiment_id}")
async def delete_model(experiment_id: str):
    """Delete a saved model."""
    try:
        results_data = load_experiment_results(experiment_id)
        
        if results_data is None:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        if 'saved_model' not in results_data:
            raise HTTPException(status_code=404, detail="No saved model found")
        
        # Remove saved model data
        del results_data['saved_model']
        
        # Save updated results
        save_experiment_results(experiment_id, results_data)
        
        return {
            'success': True,
            'message': f'Model deleted successfully for experiment {experiment_id}'
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/api/export_results/{experiment_id}")
async def export_results(experiment_id: str, format_type: str = "json"):
    """Export experiment results."""
    try:
        results_data = load_experiment_results(experiment_id)
        
        if results_data is None:
            raise HTTPException(status_code=404, detail="Experiment results not found")
        
        if format_type.lower() == "json":
            return JSONResponse(content=results_data)
        elif format_type.lower() == "csv":
            # Convert results to CSV format
            csv_data = []
            for key, value in results_data.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        csv_data.append([f"{key}.{sub_key}", str(sub_value)])
                else:
                    csv_data.append([key, str(value)])
            
            import io
            output = io.StringIO()
            import csv
            writer = csv.writer(output)
            writer.writerow(['Parameter', 'Value'])
            writer.writerows(csv_data)
            
            return Response(
                content=output.getvalue(),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=results_{experiment_id}.csv"}
            )
        else:
            raise HTTPException(status_code=400, detail="Unsupported format type")
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/api/export_config/{experiment_id}")
async def export_config(experiment_id: str, config_data: dict):
    """Export experiment configuration."""
    try:
        results_data = load_experiment_results(experiment_id)
        
        if results_data is None:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        # Create config export
        config_export = {
            'experiment_id': experiment_id,
            'task_name': TASK_NAME,
            'parameters': results_data.get('parameters', {}),
            'model_name': results_data.get('model_name', ''),
            'dataset_name': results_data.get('dataset_name', ''),
            'export_time': datetime.now().isoformat(),
            'additional_config': config_data
        }
        
        return JSONResponse(
            content=config_export,
            headers={"Content-Disposition": f"attachment; filename=config_{experiment_id}.json"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/api/list_configs")
async def list_configs():
    """List all experiment configurations."""
    try:
        configs = []
        
        # Scan results directory for experiments
        for filename in os.listdir(RESULTS_DIR):
            if filename.startswith('experiment_') and filename.endswith('.json'):
                experiment_id = filename.replace('experiment_', '').replace('.json', '')
                results_data = load_experiment_results(experiment_id)
                
                if results_data:
                    configs.append({
                        'experiment_id': experiment_id,
                        'task_name': results_data.get('task_name', 'Unknown'),
                        'model_name': results_data.get('model_name', 'Unknown'),
                        'dataset_name': results_data.get('dataset_name', 'Unknown'),
                        'start_time': results_data.get('start_time', ''),
                        'status': results_data.get('status', 'unknown')
                    })
        
        return {
            'configs': configs,
            'count': len(configs)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/api/results/{experiment_id}")
async def results_page(request: Request, experiment_id: str):
    """Results page for a specific experiment."""
    results_data = load_experiment_results(experiment_id)
    
    if results_data is None:
        # Return error page if no results found
        return templates.TemplateResponse(
            "node_tasks/classification/results.html",
            {
                "request": request,
                "task_config": task_config,
                "task_metadata": task_metadata,
                "error": f"No results found for experiment {experiment_id}",
                "results": None
            }
        )
    
    return templates.TemplateResponse(
        "node_tasks/classification/results.html",
        {
            "request": request,
            "task_config": task_config,
            "task_metadata": task_metadata,
            "results": results_data
        }
    )

@router.post("/api/save_experiment_results/{experiment_id}")
async def save_experiment_results_api(experiment_id: str, results_data: dict):
    """Save experiment results after training."""
    try:
        # Add metadata
        results_data['experiment_id'] = experiment_id
        results_data['save_time'] = datetime.now().isoformat()
        results_data['status'] = 'completed'
        
        # Save results
        save_experiment_results(experiment_id, results_data)
        
        return {
            'success': True,
            'message': f'Results saved successfully for experiment {experiment_id}'
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
