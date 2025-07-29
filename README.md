# Super Resolution with Knowledge Distillation

This is a refactored and modularized version of the super resolution project that uses knowledge distillation to train a lightweight student model (SwinIR) from a teacher model (MicroSR).

## Project Structure

```
KD/
├── config.py                 # Centralized configuration
├── main.py                   # Main entry point with CLI
├── train_modular.py          # Modular training script
├── inference_modular.py      # Modular inference script
├── test_metrics_modular.py   # Modular evaluation script
├── test_speed_modular.py     # Modular benchmarking script
├── models/
│   ├── __init__.py
│   ├── model_loader.py       # Centralized model loading
│   └── team07_MicroSR/       # Teacher model components
└── utils/
    ├── __init__.py
    ├── dataset.py            # Dataset utilities
    ├── inference.py          # Inference processing
    ├── evaluation.py         # Model evaluation utilities
    └── benchmark.py          # Performance benchmarking
```

## Key Improvements

### 1. **Centralized Configuration** (`config.py`)
- All hyperparameters, paths, and settings in one place
- Easy to modify without touching multiple files
- Consistent configuration across all scripts

### 2. **Modular Model Loading** (`models/model_loader.py`)
- Single source of truth for model definitions
- Consistent model loading across training, inference, and testing
- Support for both wrapped and unwrapped teacher models

### 3. **Reusable Utilities**
- **Dataset utilities** (`utils/dataset.py`): Shared dataset class and dataloader creation
- **Inference processing** (`utils/inference.py`): Preprocessing, postprocessing, and patch-based inference
- **Evaluation utilities** (`utils/evaluation.py`): PSNR, LPIPS, and performance evaluation
- **Benchmarking utilities** (`utils/benchmark.py`): Speed and efficiency testing

### 4. **Unified CLI Interface** (`main.py`)
- Single entry point for all operations
- Command-line interface for training, inference, evaluation, and benchmarking
- Easy to integrate into automated workflows

## Usage

### Using the CLI (Recommended)

```powershell
# Train the model
python main.py train --epochs 100 --batch-size 8

# Run inference
python main.py inference

# Evaluate performance
python main.py evaluate

# Benchmark speed
python main.py benchmark

# Show current configuration
python main.py config
```

### Using Individual Scripts

```powershell
# Training
python train_modular.py

# Inference
python inference_modular.py

# Evaluation
python test_metrics_modular.py

# Speed testing
python test_speed_modular.py
```

## Configuration

Edit `config.py` to modify:

- **Model paths**: Teacher model location, student checkpoint path
- **Dataset paths**: Training and validation data folders
- **Training parameters**: Batch size, learning rate, epochs
- **Model parameters**: Window sizes, patch sizes
- **Loss weights**: MSE and LPIPS loss weights

Example configuration update:
```python
# In config.py
TRAIN_HR_FOLDER = "path/to/your/train/data"
VAL_HR_FOLDER = "path/to/your/val/data"
BATCH_SIZE = 16
NUM_EPOCHS = 200
```

## Dependencies

The modular version maintains the same dependencies as the original:

- PyTorch
- torchvision
- BasicSR (for SwinIR)
- lpips
- scikit-image
- opencv-python
- matplotlib
- tqdm
- fvcore (optional, for FLOPs analysis)

## Migration from Original Code

The original files (`train.py`, `inference.py`, `test_metrics.py`, `test_speed.py`) are preserved for reference. The new modular versions provide the same functionality with better organization:

- `train.py` → `train_modular.py`
- `inference.py` → `inference_modular.py`
- `test_metrics.py` → `test_metrics_modular.py`
- `test_speed.py` → `test_speed_modular.py`

## Benefits of Modular Design

1. **Reduced Code Duplication**: Common functionality is shared across scripts
2. **Easier Maintenance**: Changes to model definitions or processing logic only need to be made in one place
3. **Better Testing**: Individual components can be tested independently
4. **Improved Readability**: Each module has a clear, focused responsibility
5. **Scalability**: Easy to add new models, datasets, or evaluation metrics
6. **Configuration Management**: Centralized settings prevent inconsistencies

## Example Workflow

1. **Configure** your paths and parameters in `config.py`
2. **Train** the student model: `python main.py train`
3. **Evaluate** performance: `python main.py evaluate`
4. **Benchmark** speed: `python main.py benchmark`
5. **Run inference** on test images: `python main.py inference`

This modular structure makes the codebase much more maintainable and extensible while preserving all the original functionality.
