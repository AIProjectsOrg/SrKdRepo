"""
Main entry point for the super resolution project.
Provides a unified interface for training, inference, evaluation, and benchmarking.
"""
import argparse
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config


def train(args):
    """Run training"""
    from train_modular import SuperResolutionTrainer
    
    # Update config if custom paths provided
    if args.train_data:
        Config.TRAIN_HR_FOLDER = args.train_data
    if args.val_data:
        Config.VAL_HR_FOLDER = args.val_data
    if args.epochs:
        Config.NUM_EPOCHS = args.epochs
    if args.batch_size:
        Config.BATCH_SIZE = args.batch_size
    
    trainer = SuperResolutionTrainer()
    trainer.setup_datasets()
    trainer.train()


def inference(args):
    """Run inference"""
    from inference_modular import main as inference_main
    
    # Update config if custom paths provided
    if args.teacher_model:
        Config.TEACHER_MODEL_PATH = args.teacher_model
    if args.student_model:
        Config.STUDENT_CHECKPOINT_PATH = args.student_model
    
    inference_main()


def evaluate(args):
    """Run evaluation"""
    from test_metrics_modular import main as evaluate_main
    
    # Update config if custom paths provided
    if args.val_data:
        Config.VAL_HR_FOLDER = args.val_data
    if args.teacher_model:
        Config.TEACHER_MODEL_PATH = args.teacher_model
    if args.student_model:
        Config.STUDENT_CHECKPOINT_PATH = args.student_model
    
    evaluate_main()


def benchmark(args):
    """Run speed benchmark"""
    from test_speed_modular import main as benchmark_main
    
    # Update config if custom paths provided
    if args.teacher_model:
        Config.TEACHER_MODEL_PATH = args.teacher_model
    if args.student_model:
        Config.STUDENT_CHECKPOINT_PATH = args.student_model
    
    benchmark_main()


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Super Resolution with Knowledge Distillation')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train the student model')
    train_parser.add_argument('--train-data', type=str, help='Path to training data folder')
    train_parser.add_argument('--val-data', type=str, help='Path to validation data folder')
    train_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, help='Batch size for training')
    
    # Inference command
    inference_parser = subparsers.add_parser('inference', help='Run inference on test images')
    inference_parser.add_argument('--teacher-model', type=str, help='Path to teacher model')
    inference_parser.add_argument('--student-model', type=str, help='Path to student model checkpoint')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model performance')
    eval_parser.add_argument('--val-data', type=str, help='Path to validation data folder')
    eval_parser.add_argument('--teacher-model', type=str, help='Path to teacher model')
    eval_parser.add_argument('--student-model', type=str, help='Path to student model checkpoint')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark model speed')
    benchmark_parser.add_argument('--teacher-model', type=str, help='Path to teacher model')
    benchmark_parser.add_argument('--student-model', type=str, help='Path to student model checkpoint')
    
    # Configuration command
    config_parser = subparsers.add_parser('config', help='Show current configuration')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train(args)
    elif args.command == 'inference':
        inference(args)
    elif args.command == 'evaluate':
        evaluate(args)
    elif args.command == 'benchmark':
        benchmark(args)
    elif args.command == 'config':
        print("Current Configuration:")
        print("-" * 40)
        for key, value in Config.to_dict().items():
            print(f"{key}: {value}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
