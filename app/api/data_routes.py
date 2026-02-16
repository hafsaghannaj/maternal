from flask import Blueprint, request, jsonify
from typing import List, Optional, Dict, Any
import json
import os
import asyncio

from app.external.ahr_client import AHRClient
from app.external.cdc_wonder import CDCWonderClient
from app.external.datafenix import DataFenixClient
from app.data.pipeline import run_data_pipeline, _run_pipeline_async

data_bp = Blueprint('data_integration', __name__, url_prefix='/api/v1')

def run_async(coro):
    """Helper to run async code in Flask."""
    return asyncio.run(coro)

@data_bp.route('/benchmarks/ahr', methods=['GET'])
def get_ahr_benchmark():
    measure = request.args.get('measure')
    state = request.args.get('state')
    
    if not measure:
        return jsonify({"error": "Missing measure parameter"}), 400
        
    client = AHRClient()
    results = run_async(client.get_measure_by_state(measure))
    
    if state:
        results = [r.dict() for r in results if r.state == state]
    else:
        results = [r.dict() for r in results]
        
    return jsonify(results)

@data_bp.route('/benchmarks/ahr/rankings', methods=['GET'])
def get_ahr_rankings():
    report = request.args.get('report', 'women_and_children')
    client = AHRClient()
    results = run_async(client.get_rankings(report))
    return jsonify([r.dict() for r in results])

@data_bp.route('/benchmarks/ahr/disparities', methods=['GET'])
def get_ahr_disparities():
    measure = request.args.get('measure')
    if not measure:
        return jsonify({"error": "Missing measure parameter"}), 400
        
    client = AHRClient()
    results = run_async(client.get_measure_with_disparities(measure))
    return jsonify([r.dict() for r in results])

@data_bp.route('/benchmarks/cdc', methods=['GET'])
def get_cdc_benchmark():
    dataset = request.args.get('dataset')
    group_by = request.args.getlist('group_by')
    year = request.args.get('year', '2021-2023')
    
    if not dataset:
        return jsonify({"error": "Missing dataset parameter"}), 400
        
    year_range = year.split("-")
    client = CDCWonderClient()
    
    if dataset == "D66":
        df = run_async(client.get_birth_demographics(year_range, group_by))
        return jsonify(df.to_dict(orient="records"))
    
    return jsonify({"error": "Unsupported dataset"}), 400

@data_bp.route('/data/calibrate', methods=['POST'])
def trigger_calibration():
    is_sync = request.args.get('sync', 'false').lower() == 'true'
    
    if is_sync:
        try:
            result = run_async(_run_pipeline_async())
            return jsonify({"status": "success", "report": result})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
            
    try:
        task = run_data_pipeline.delay()
        return jsonify({"task_id": task.id, "status": "queued"})
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": "Celery/Redis not running. Try adding ?sync=true to your request.",
            "details": str(e)
        }), 503

@data_bp.route('/data/calibration-status', methods=['GET'])
def get_calibration_status():
    path = os.getenv("CALIBRATION_OUTPUT_PATH", "./config/calibration_params.json")
    if not os.path.exists(path):
        return jsonify({"status": "never_run"})
    
    try:
        with open(path, 'r') as f:
            params = json.load(f)
        
        return jsonify({
            "status": "success",
            "last_updated": os.path.getmtime(path),
            "features": list(params.keys())
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@data_bp.route('/self-report/cycle-analysis', methods=['POST'])
def analyze_cycle():
    data = request.get_json()
    dates = data.get('dates', [])
    
    if not dates:
        return jsonify({"error": "Missing dates"}), 400
        
    client = DataFenixClient()
    result = run_async(client.analyze_cycle(dates))
    return jsonify(result)
