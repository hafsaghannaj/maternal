from flask import Flask
from flask_cors import CORS
from app.api.endpoints import api_bp
from config import config


def create_app():
    app = Flask(__name__)
    CORS(app)  # Enable CORS for all routes

    # Register blueprints
    app.register_blueprint(api_bp)

    return app


if __name__ == '__main__':
    app = create_app()

    print("Starting Project Artemis Server")
    print("=" * 50)
    print(f"Device: {config.DEVICE}")
    print(f"Hospitals: {config.NUM_HOSPITALS}")
    print(f"Features: {config.NUM_FEATURES}")
    print("=" * 50)
    print("API Endpoints:")
    print("  POST /api/initialize - Initialize federated learning")
    print("  POST /api/train     - Run federated training")
    print("  GET  /api/evaluate  - Evaluate current model")
    print("  POST /api/predict   - Predict maternal risk")
    print("  GET  /api/history   - Get training history")
    print("=" * 50)

    app.run(debug=True, host='0.0.0.0', port=5001)
