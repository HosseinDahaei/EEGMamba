# EEGMamba

Deep learning framework for EEG analysis using Mamba architecture.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Training](#training)
- [Preprocessing](#preprocessing)
- [Model Architectures](#model-architectures)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or use the full environment
conda env create -f environment.full.yml
```

## Quick Start

```python
from datasets.physio_dataset import LoadDataset
from models.model_for_physio import Model

# Load dataset and train
# See quick_example.py for a complete example
```

## Hyperparameter Optimization

The `optimize_weight_decay.py` script provides a comprehensive tool for optimizing the weight decay hyperparameter for EEGMamba models on the PhysioNet Motor Imagery dataset.

### Features

- 🔬 Systematic hyperparameter search across multiple weight decay values
- 📊 Comprehensive metrics tracking (Accuracy, Balanced Accuracy, F1, Cohen's Kappa)
- 📈 Automatic visualization of training curves and comparison charts
- 💾 JSON export of all experiment results
- ⚡ Quick test mode for rapid experimentation

### Usage

#### Basic Usage

Run optimization with default parameters (weight_decay: 1, 5, 10, 15, 20; epochs: 30):

```bash
python optimize_weight_decay.py
```

#### Custom Weight Decay Values

Test specific weight decay values:

```bash
python optimize_weight_decay.py --weight_decays 0.1 1 5 10 20 50
```

#### Custom Number of Epochs

Run experiments with a specific number of epochs (constant for all experiments):

```bash
python optimize_weight_decay.py --epochs 50
```

#### Adjust Learning Rate and Batch Size

```bash
python optimize_weight_decay.py --lr 5e-5 --batch_size 64
```

#### Quick Test Mode

For rapid testing with reduced configurations (2 weight decay values, 5 epochs):

```bash
python optimize_weight_decay.py --quick
```

#### Custom Save Path

Specify where to save results:

```bash
python optimize_weight_decay.py --save_path my_optimization
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--weight_decays` | float list | `[1, 5, 10, 15, 20]` | List of weight decay values to test |
| `--epochs` | int | `30` | Number of epochs to train (constant for all experiments) |
| `--batch_size` | int | `32` | Batch size for training |
| `--lr` | float | `1e-4` | Learning rate |
| `--save_path` | str | `optimization_results` | Base path to save results |
| `--quick` | flag | `False` | Enable quick test mode (2 configs, 5 epochs) |

### Output

The script generates the following outputs in `results/{save_path}_{timestamp}/`:

1. **`results.json`** - Complete experiment data including:
   - Training history (loss, accuracy per epoch)
   - Validation metrics per epoch
   - Final test set performance
   - Best validation accuracy and epoch

2. **`optimization_curves.png`** - Comprehensive 6-panel visualization:
   - Validation accuracy evolution
   - Training loss evolution
   - Validation F1 score evolution
   - Validation Cohen's Kappa evolution
   - Best validation accuracy vs weight decay
   - Final test accuracy comparison

3. **`optimization_comparison.png`** - Summary bar charts comparing:
   - Test Accuracy
   - Test Balanced Accuracy
   - Test F1 Score
   - Cohen's Kappa

4. **Console Summary Table** - Formatted results showing:
   - Weight decay and epochs for each experiment
   - Best validation accuracy
   - Test metrics (Accuracy, Balanced Accuracy, F1, Kappa)
   - Highlighted best configuration

### Example Output

```
================================================================================
🎯 EEGMamba Weight Decay Optimization
================================================================================
Weight Decay values: [1, 5, 10, 15, 20]
Epochs (constant): 30
Batch size: 32
Learning rate: 0.0001
Total experiments: 5

✅ CUDA available: NVIDIA GeForce RTX 3090

################################################################################
Experiment 1/5
################################################################################

🔬 Experiment: weight_decay=1, epochs=30
================================================================================
📂 Loading dataset...
🏗️ Initializing model...
🚀 Starting training for 30 epochs...
Epoch 1/30: Loss=1.2345, Train Acc=45.23%, Val Acc=42.15%, Val F1=41.80%
...

📊 OPTIMIZATION RESULTS SUMMARY
====================================================================================================
WD         Epochs     Best Val     Test Acc     Bal Acc      F1           Kappa     
----------------------------------------------------------------------------------------------------
15         30         85.67%       84.32%       83.45%       83.21%       0.7789
10         30         84.23%       83.12%       82.34%       82.10%       0.7654
...

🏆 BEST CONFIGURATION:
   Weight Decay: 15
   Epochs: 30
   Test Accuracy: 84.32%
   Test Balanced Accuracy: 83.45%
   Test F1 Score: 83.21%
   Cohen's Kappa: 0.7789
```

### Advanced Examples

#### Comprehensive Search

```bash
# Test wide range of weight decay values
python optimize_weight_decay.py \
    --weight_decays 0.01 0.1 1 5 10 15 20 25 30 50 \
    --epochs 50 \
    --lr 1e-4 \
    --batch_size 32
```

#### Fine-Grained Search Around Optimal Value

```bash
# If you found that weight_decay=15 works best, search around it
python optimize_weight_decay.py \
    --weight_decays 12 13 14 15 16 17 18 \
    --epochs 40
```

#### Resource-Constrained Environment

```bash
# Smaller batch size, fewer epochs
python optimize_weight_decay.py \
    --weight_decays 5 10 15 \
    --epochs 20 \
    --batch_size 16
```

### Implementation Details

The optimization script:
- Uses **AdamW optimizer** with configurable learning rate
- Employs **CrossEntropyLoss** for classification
- Tracks metrics on training, validation, and test sets
- Automatically saves best model based on validation accuracy
- Supports CUDA acceleration (automatically detected)
- Uses pretrained EEGMamba weights by default
- Applies the "all_patch_reps" classifier configuration

### Tips for Effective Optimization

1. **Start with default values** to establish a baseline
2. **Use --quick mode** for initial exploration
3. **Refine search** around promising values
4. **Consider computational budget** when setting epochs
5. **Check for overfitting** by comparing train vs validation curves
6. **Analyze the visualizations** to understand training dynamics
7. **Use multiple seeds** for statistical significance (modify `params.seed` in script)

## Training

For regular training without optimization:

```bash
# Fine-tuning on PhysioNet-MI
python finetune_main.py

# Pre-training
python pretrain_main.py
```

See `TRAINING_LOG_DOCUMENTATION.md` for detailed training documentation.

## Preprocessing

Dataset preprocessing scripts are located in `preprocessing/`:
- `preprocessing_for_pretraining/` - Scripts for pretraining data
- `preprocessing_for_finetuning/` - Scripts for downstream task datasets

See `preprocessing/README.md` for detailed preprocessing instructions.

## Model Architectures

The framework includes models for multiple EEG datasets:
- **PhysioNet Motor Imagery** (`model_for_physio.py`)
- **BCI Competition IV 2a** (`model_for_bciciv2a.py`)
- **SEED-V** (`model_for_seedv.py`)
- **SEED-VIG** (`model_for_seedvig.py`)
- **And more...**

Each model uses the EEGMamba backbone with dataset-specific classifiers.

## Project Structure

```
EEGMamba/
├── datasets/              # Dataset loaders
├── models/                # Model architectures
├── modules/               # Core Mamba modules
├── preprocessing/         # Data preprocessing scripts
├── pretrained_weights/    # Pre-trained model weights
├── results/               # Training and optimization results
├── utils/                 # Utility functions
├── optimize_weight_decay.py  # Hyperparameter optimization
├── finetune_main.py      # Fine-tuning script
├── pretrain_main.py      # Pre-training script
└── quick_example.py      # Quick start example
```

## Citation

If you use this code, please cite the original EEGMamba paper:
```bibtex
@article{eegmamba2024,
  title={EEGMamba: Efficient EEG Analysis with Mamba},
  author={...},
  journal={...},
  year={2024}
}
```

## License

[Include your license information here]

## Contact

For questions or issues, please open an issue on GitHub or contact [your contact information].
