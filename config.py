import os
import torch

class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Data settings
    NUM_HOSPITALS = 3
    NUM_SAMPLES_PER_HOSPITAL = 1000
    NUM_FEATURES = 25
    TEST_SIZE = 0.2
    
    # Model settings
    INPUT_SIZE = 25
    HIDDEN_SIZE = 64
    OUTPUT_SIZE = 1
    DROPOUT_RATE = 0.3
    
    # Training settings
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    FEDERATED_ROUNDS = 5
    
    # Privacy settings
    MAX_GRAD_NORM = 1.0
    NOISE_MULTIPLIER = 1.1
    DELTA = 1e-5
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Storage
    DB_PATH = os.path.join(BASE_DIR, "artemis.sqlite3")
    MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
    
config = Config()
