# train.py

import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
from src.model import ChatbotModel
from src.preprocess import TextPreprocessor
from src.utils import ModelEvaluator, ChatbotUtils

# Load environment variables
load_dotenv('config/.env')

def train_model(args):
    # Set up paths
    training_data = Path(os.getenv('TRAINING_DATA', 'data/training_data.csv'))
    model_path = Path(os.getenv('MODEL_PATH', 'models/chatbot_model.h5'))
    log_dir = Path(os.getenv('LOG_DIR', 'logs'))
    
    # Create directories if they don't exist
    model_path.parent.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    
    # Set up logging
    ChatbotUtils.setup_logging(log_dir)
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(
        max_words=int(os.getenv('MAX_WORDS', 1000)),
        max_len=int(os.getenv('MAX_LENGTH', 100))
    )
    
    # Load and prepare data
    print("Loading training data...")
    texts, intents = preprocessor.load_training_data(training_data)
    
    print("Preparing data...")
    X, y = preprocessor.prepare_data(texts, intents)
    
    # Initialize and build model
    print("Building model...")
    model = ChatbotModel(
        vocab_size=preprocessor.get_vocab_size(),
        num_classes=preprocessor.get_num_classes(),
        max_len=int(os.getenv('MAX_LENGTH', 100)),
        embedding_dim=int(os.getenv('EMBEDDING_DIM', 128))
    )
    model.build_model()
    
    # Train model
    print("Training model...")
    history = model.train(
        X, y,
        epochs=args.epochs or int(os.getenv('EPOCHS', 50)),
        batch_size=args.batch_size or int(os.getenv('BATCH_SIZE', 32))
    )
    
    # Evaluate model
    print("Evaluating model...")
    evaluator = ModelEvaluator()
    plot = evaluator.plot_training_history(history)
    plot.savefig(log_dir / "training_history.png")
    
    # Save model
    print("Saving model...")
    model.save_model(model_path)
    
    print(f"Model saved to {model_path}")
    print(f"Training history plot saved to {log_dir}/training_history.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train chatbot model")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    args = parser.parse_args()
    
    train_model(args)