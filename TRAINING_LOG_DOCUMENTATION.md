# Training Log and Visualization Documentation

## Overview
The EEGMamba finetuning process now automatically logs all training metrics to a JSON file and includes visualization capabilities to track training progress.

## What Was Added

### 1. Modified `finetune_trainer.py`

#### New Features:
- **Automatic Logging**: All training metrics are now logged to `training_log.json` in the model directory
- **Comprehensive Metrics**: Logs include:
  - Training loss per epoch
  - Validation accuracy, kappa, F1 score (for classification tasks)
  - Validation ROC-AUC, PR-AUC (for binary classification)
  - Validation R², correlation coefficient, RMSE (for regression tasks)
  - Learning rate schedule
  - Training time per epoch
  - Final test set results
  - Confusion matrices

#### Implementation Details:
```python
# Training history is initialized in __init__
self.training_history = {
    'train_loss': [],
    'val_acc': [],
    'val_kappa': [],
    'val_f1': [],
    'val_pr_auc': [],
    'val_roc_auc': [],
    'val_corrcoef': [],
    'val_r2': [],
    'val_rmse': [],
    'learning_rate': [],
    'epoch_time': [],
    'test_results': {}
}
```

Metrics are automatically saved after each epoch and a final log is saved after training completes.

### 2. New Notebook Cell: Training Progress Visualization

#### Location:
- **Notebook**: `PhysioNet_MI_Finetuning_Test.ipynb`
- **Section**: Step 5.5 - Visualize Training Progress
- **Cell Number**: After the training cell (cell 17)

#### Visualization Features:

The visualization cell creates a comprehensive 3x3 grid of plots showing:

1. **Training Loss**: Loss curve over all epochs
2. **Validation Accuracy**: Accuracy progression with best value marked
3. **Validation F1 Score**: F1 score progression (for classification)
4. **Validation Cohen's Kappa**: Kappa metric progression
5. **Learning Rate Schedule**: Shows the learning rate decay over time
6. **Training Time per Epoch**: Time analysis with average marked
7. **Combined Loss vs Accuracy**: Dual-axis plot showing both metrics
8. **AUC Metrics**: ROC-AUC and PR-AUC (for binary classification)
9. **Regression Metrics**: R² and correlation coefficient (for regression)

#### Text Summary Includes:
- Total training duration and time per epoch
- Loss reduction percentage
- Accuracy improvement
- Best epoch for each metric
- Final test set performance

## Usage

### During Training
No action needed! The logging happens automatically when you run:
```python
!python finetune_main.py --downstream_dataset PhysioNet-MI ...
```

### After Training
Simply run the visualization cell (Step 5.5):
```python
# The cell automatically searches for training logs
history = plot_training_progress("./results/physio_models_test/training_log.json")
```

### Custom Log Path
If your log is in a different location:
```python
history = plot_training_progress("/path/to/your/training_log.json")
```

## File Locations

### Training Log
- **Default Path**: `{MODEL_DIR}/training_log.json`
- **Example**: `./results/physio_models_test/training_log.json`
- **Format**: JSON file with all metrics

### Log File Structure
```json
{
  "train_loss": [0.523, 0.412, ...],
  "val_acc": [0.45, 0.52, ...],
  "val_kappa": [0.32, 0.41, ...],
  "val_f1": [0.44, 0.51, ...],
  "learning_rate": [0.0001, 0.00009, ...],
  "epoch_time": [2.3, 2.1, ...],
  "test_results": {
    "acc": 0.589,
    "kappa": 0.451,
    "f1": 0.588,
    "confusion_matrix": [[...], [...]],
    "best_epoch": 13
  }
}
```

## Benefits

1. **Track Progress**: Monitor training metrics in real-time
2. **Identify Issues**: Quickly spot overfitting, underfitting, or learning rate problems
3. **Compare Experiments**: Load and compare different training runs
4. **Reproducibility**: Complete log of training configuration and results
5. **Publication Ready**: Generate publication-quality training curves

## Examples

### Compare Multiple Training Runs
```python
# Load multiple logs
history1 = plot_training_progress("./results/run1/training_log.json")
history2 = plot_training_progress("./results/run2/training_log.json")

# Compare specific metrics
plt.figure(figsize=(10, 6))
plt.plot(history1['val_acc'], label='Run 1')
plt.plot(history2['val_acc'], label='Run 2')
plt.legend()
plt.title('Validation Accuracy Comparison')
plt.show()
```

### Export Metrics for External Analysis
```python
import pandas as pd

# Load log
with open('./results/physio_models_test/training_log.json', 'r') as f:
    history = json.load(f)

# Convert to DataFrame
df = pd.DataFrame({
    'epoch': range(1, len(history['train_loss']) + 1),
    'train_loss': history['train_loss'],
    'val_acc': history['val_acc'],
    'val_kappa': history['val_kappa'],
    'val_f1': history['val_f1']
})

# Export to CSV
df.to_csv('training_metrics.csv', index=False)
```

## Troubleshooting

### Log File Not Found
- Ensure training has completed successfully
- Check the model directory path in your training command
- Look for the log in `{MODEL_DIR}/training_log.json`

### Missing Metrics in Plots
- Some plots only appear for specific task types:
  - F1/Kappa: Classification tasks
  - ROC-AUC/PR-AUC: Binary classification
  - R²/Correlation: Regression tasks

### Visualization Cell Doesn't Work
- Make sure you have matplotlib installed: `pip install matplotlib`
- Ensure the JSON file is valid (not corrupted during training)

## Support for All Task Types

The logging system supports:
- ✅ Multi-class classification (PhysioNet-MI, BCIC-IV-2a, FACED, SEED-V, etc.)
- ✅ Binary classification (SHU-MI, CHB-MIT, Mumtaz2016, etc.)
- ✅ Regression (SEED-VIG)

All metrics relevant to each task type are automatically logged.
