"""
Anomaly Detection Routes
This module contains all routes for anomaly detection tasks.
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
router = APIRouter(prefix="/anomaly_detection", tags=["Anomaly Detection"])

# Task configuration
TASK_NAME = 'anomaly_detection'
task_config = get_task_config(TASK_NAME)
task_metadata = get_task_metadata(TASK_NAME)

# Setup templates
templates = Jinja2Templates(directory="webapp/templates")

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page for anomaly detection."""
    return templates.TemplateResponse(
        "anomaly_detection/index.html",
        {
            "request": request,
            "task_config": task_config,
            "task_metadata": task_metadata
        }
    )

@router.get("/experiment", response_class=HTMLResponse)
async def experiment(request: Request):
    """Experiment page for anomaly detection."""
    # Get parameters and defaults
    parameters = get_task_parameters(TASK_NAME)
    default_params = get_task_default_params(TASK_NAME)
    supported_models = get_models_for_task(TASK_NAME)
    
    return templates.TemplateResponse(
        "anomaly_detection/experiment.html",
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
    """Results page for anomaly detection."""
    # Mock results data for demonstration
    results_data = {
        'experiment_id': experiment_id,
        'task_name': 'Anomaly Detection',
        'model_name': 'GCN-AE',
        'dataset_name': 'Cora',
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'auc': 0.923,
            'ap': 0.918,
            'precision': 0.856,
            'recall': 0.789,
            'f1_score': 0.821,
            'detection_rate': 0.845
        },
        'anomaly_stats': {
            'total_nodes': 2708,
            'anomalous_nodes': 135,
            'normal_nodes': 2573,
            'anomaly_ratio': 0.05,
            'detected_anomalies': 120,
            'false_positives': 15
        },
        'anomaly_scores': {
            'node_ids': list(range(2708)),
            'anomaly_scores': np.random.uniform(0, 1, 2708).tolist(),
            'is_anomaly': np.random.choice([0, 1], 2708, p=[0.95, 0.05]).tolist(),
            'detected_anomalies': np.random.choice([0, 1], 2708, p=[0.96, 0.04]).tolist()
        },
        'roc_curve': {
            'fpr': np.linspace(0, 1, 100).tolist(),
            'tpr': (np.linspace(0, 1, 100) ** 0.8).tolist(),
            'thresholds': np.linspace(1, 0, 100).tolist()
        }
    }
    
    return templates.TemplateResponse(
        "anomaly_detection/results.html",
        {
            "request": request,
            "task_config": task_config,
            "task_metadata": task_metadata,
            "results": results_data
        }
    )

@router.post("/api/start_experiment")
async def start_experiment(data: Dict[str, Any]):
    """Start a new anomaly detection experiment."""
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
            'model_name': experiment_params.get('model', 'gcn_ae'),
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
            'reconstruction_error': 0.25,
            'anomaly_score_threshold': 0.75
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
    """Get parameters for anomaly detection."""
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
        'task_name': 'Anomaly Detection',
        'model_name': 'GCN-AE',
        'dataset_name': 'Cora',
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'auc': 0.923,
            'ap': 0.918,
            'precision': 0.856,
            'recall': 0.789,
            'f1_score': 0.821,
            'detection_rate': 0.845
        },
        'anomaly_stats': {
            'total_nodes': 2708,
            'anomalous_nodes': 135,
            'normal_nodes': 2573,
            'anomaly_ratio': 0.05,
            'detected_anomalies': 120,
            'false_positives': 15
        }
    }
    
    return results_data

@router.get("/api/get_anomaly_scores/{experiment_id}")
async def get_anomaly_scores(experiment_id: str):
    """Get anomaly scores for all nodes."""
    # Mock anomaly scores data
    anomaly_scores = {
        'experiment_id': experiment_id,
        'node_ids': list(range(2708)),
        'anomaly_scores': np.random.uniform(0, 1, 2708).tolist(),
        'is_anomaly': np.random.choice([0, 1], 2708, p=[0.95, 0.05]).tolist(),
        'detected_anomalies': np.random.choice([0, 1], 2708, p=[0.96, 0.04]).tolist(),
        'confidence_scores': np.random.uniform(0.7, 1.0, 2708).tolist()
    }
    
    return anomaly_scores

@router.get("/api/get_roc_curve/{experiment_id}")
async def get_roc_curve(experiment_id: str):
    """Get ROC curve data."""
    # Mock ROC curve data
    roc_curve = {
        'experiment_id': experiment_id,
        'fpr': np.linspace(0, 1, 100).tolist(),
        'tpr': (np.linspace(0, 1, 100) ** 0.8).tolist(),
        'thresholds': np.linspace(1, 0, 100).tolist(),
        'auc': 0.923
    }
    
    return roc_curve

@router.get("/api/get_anomaly_distribution/{experiment_id}")
async def get_anomaly_distribution(experiment_id: str):
    """Get distribution of anomaly scores."""
    # Mock anomaly distribution data
    anomaly_distribution = {
        'experiment_id': experiment_id,
        'score_bins': np.linspace(0, 1, 20).tolist(),
        'normal_counts': np.random.poisson(150, 20).tolist(),
        'anomaly_counts': np.random.poisson(10, 20).tolist(),
        'threshold': 0.75,
        'percentiles': {
            '25': 0.23,
            '50': 0.45,
            '75': 0.67,
            '90': 0.82,
            '95': 0.89
        }
    }
    
    return anomaly_distribution

@router.get("/api/get_top_anomalies/{experiment_id}")
async def get_top_anomalies(experiment_id: str, top_k: int = 20):
    """Get top-k anomalies."""
    # Mock top anomalies data
    top_anomalies = {
        'experiment_id': experiment_id,
        'top_k': top_k,
        'anomalies': [
            {
                'node_id': i,
                'anomaly_score': np.random.uniform(0.8, 1.0),
                'is_anomaly': np.random.choice([0, 1], p=[0.2, 0.8]),
                'detected': np.random.choice([0, 1], p=[0.1, 0.9]),
                'confidence': np.random.uniform(0.8, 1.0),
                'node_features': np.random.randn(10).tolist()
            }
            for i in range(top_k)
        ]
    }
    
    return top_anomalies

@router.get("/api/get_anomaly_network/{experiment_id}")
async def get_anomaly_network(experiment_id: str):
    """Get network visualization with anomaly highlighting."""
    # Mock anomaly network data
    anomaly_network = {
        'experiment_id': experiment_id,
        'nodes': list(range(2708)),
        'edges': np.random.rand(5429, 2).tolist(),
        'node_anomaly_scores': np.random.uniform(0, 1, 2708).tolist(),
        'node_positions': np.random.rand(2708, 2).tolist(),
        'anomaly_threshold': 0.75,
        'highlighted_nodes': np.random.choice(range(2708), 135, replace=False).tolist()
    }
    
    return anomaly_network

@router.get("/api/export_results/{experiment_id}")
async def export_results(experiment_id: str, format_type: str = "json"):
    """Export experiment results in various formats."""
    try:
        # Mock results data
        results_data = {
            'experiment_id': experiment_id,
            'task_name': 'Anomaly Detection',
            'model_name': 'GCN-AE',
            'dataset_name': 'Cora',
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'auc': 0.923,
                'ap': 0.918,
                'precision': 0.856,
                'recall': 0.789
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

@router.get("/api/get_anomaly_visualization/{experiment_id}")
async def get_anomaly_visualization(experiment_id: str):
    """Get anomaly visualization data."""
    # Mock anomaly visualization data
    anomaly_visualization = {
        'experiment_id': experiment_id,
        'nodes': list(range(2708)),
        'edges': np.random.rand(5429, 2).tolist(),
        'node_anomaly_scores': np.random.uniform(0, 1, 2708).tolist(),
        'node_positions': np.random.rand(2708, 2).tolist(),
        'anomaly_threshold': 0.75,
        'color_scheme': {
            'normal_color': '#4CAF50',
            'anomaly_color': '#F44336',
            'threshold_color': '#FF9800'
        }
    }
    
    return anomaly_visualization

@router.get("/api/get_anomaly_analysis/{experiment_id}")
async def get_anomaly_analysis(experiment_id: str):
    """Get detailed anomaly analysis."""
    # Mock anomaly analysis data
    anomaly_analysis = {
        'experiment_id': experiment_id,
        'total_nodes': 2708,
        'anomalous_nodes': 135,
        'normal_nodes': 2573,
        'anomaly_ratio': 0.05,
        'detected_anomalies': 120,
        'false_positives': 15,
        'false_negatives': 20,
        'true_positives': 115,
        'true_negatives': 2558,
        'precision': 0.856,
        'recall': 0.789,
        'f1_score': 0.821,
        'detection_rate': 0.845,
        'anomaly_patterns': {
            'isolated_nodes': 0.35,
            'high_degree_nodes': 0.25,
            'bridge_nodes': 0.20,
            'community_outliers': 0.20
        },
        'temporal_analysis': {
            'anomaly_persistence': 0.67,
            'anomaly_evolution': 0.45,
            'seasonal_patterns': 0.23
        }
    }
    
    return anomaly_analysis
