#!/usr/bin/env python3
"""
Minimal script to optimize weight_decay parameter for EEGMamba on PhysioNet-MI
Runs multiple experiments with different weight_decay values and epochs
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import argparse

# Import EEGMamba modules
from datasets.physio_dataset import LoadDataset
from models.model_for_physio import Model
from sklearn.metrics import balanced_accuracy_score, f1_score, cohen_kappa_score


class OptimizationParams:
    """Parameters for optimization experiments"""
    def __init__(self, weight_decay, epochs, batch_size=32, lr=1e-4):
        self.datasets_dir = "/home/mahmood/HosseinDahaei/Codes/EEGMamba/data/raw_motor_movement_Imagery/processed_average"
        self.num_of_classes = 4
        self.downstream_dataset = 'PhysioNet-MI'
        self.classifier = 'all_patch_reps'
        self.use_pretrained_weights = True
        self.foundation_dir = "pretrained_weights/pretrained_weights.pth"
        self.dropout = 0.1
        self.seed = 3407
        self.cuda = 0
        
        # Experiment parameters
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.num_workers = 8


def train_one_epoch(model, data_loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred_y = torch.max(pred, dim=-1)[1]
        all_preds.extend(pred_y.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    return avg_loss, acc


def evaluate(model, data_loader, device):
    """Evaluate model"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            pred = model(x)
            pred_y = torch.max(pred, dim=-1)[1]
            
            all_preds.extend(pred_y.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    
    acc = np.mean(preds == labels)
    balanced_acc = balanced_accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    kappa = cohen_kappa_score(labels, preds)
    
    return {
        'acc': acc,
        'balanced_acc': balanced_acc,
        'f1': f1,
        'kappa': kappa
    }


def run_experiment(weight_decay, epochs, batch_size=32, lr=1e-4, verbose=True):
    """Run single experiment with given weight_decay and epochs"""
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"🔬 Experiment: weight_decay={weight_decay}, epochs={epochs}")
        print(f"{'='*80}")
    
    # Setup
    params = OptimizationParams(weight_decay, epochs, batch_size, lr)
    device = torch.device(f'cuda:{params.cuda}' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    if verbose:
        print("📂 Loading dataset...")
    load_dataset = LoadDataset(params)
    data_loaders = load_dataset.get_data_loader()
    
    # Initialize model
    if verbose:
        print("🏗️ Initializing model...")
    model = Model(params).to(device)
    
    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_balanced_acc': [],
        'val_f1': [],
        'val_kappa': []
    }
    
    best_val_acc = 0
    best_epoch = 0
    
    # Training loop
    if verbose:
        print(f"🚀 Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, data_loaders['train'], optimizer, criterion, device
        )
        
        # Evaluate on validation set
        val_metrics = evaluate(model, data_loaders['val'], device)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_metrics['acc'])
        history['val_balanced_acc'].append(val_metrics['balanced_acc'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_kappa'].append(val_metrics['kappa'])
        
        # Track best model
        if val_metrics['acc'] > best_val_acc:
            best_val_acc = val_metrics['acc']
            best_epoch = epoch + 1
        
        if verbose:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Loss={train_loss:.4f}, "
                  f"Train Acc={train_acc*100:.2f}%, "
                  f"Val Acc={val_metrics['acc']*100:.2f}%, "
                  f"Val F1={val_metrics['f1']*100:.2f}%")
    
    # Final test evaluation
    if verbose:
        print("\n📊 Evaluating on test set...")
    test_metrics = evaluate(model, data_loaders['test'], device)
    
    if verbose:
        print(f"✅ Test Results: Acc={test_metrics['acc']*100:.2f}%, "
              f"Balanced Acc={test_metrics['balanced_acc']*100:.2f}%, "
              f"F1={test_metrics['f1']*100:.2f}%, "
              f"Kappa={test_metrics['kappa']:.4f}")
    
    return {
        'weight_decay': weight_decay,
        'epochs': epochs,
        'history': history,
        'test_metrics': test_metrics,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch
    }


def plot_optimization_results(results, save_path='optimization_results'):
    """Generate comprehensive comparison charts"""
    
    os.makedirs(save_path, exist_ok=True)
    
    # Prepare data
    weight_decays = sorted(list(set([r['weight_decay'] for r in results])))
    constant_epochs = results[0]['epochs']  # All experiments use same epochs now
    
    # Create color map for weight_decay values
    colors = plt.cm.viridis(np.linspace(0, 1, len(weight_decays)))
    wd_color_map = dict(zip(weight_decays, colors))
    
    # Figure 1: Training curves for all experiments
    fig1 = plt.figure(figsize=(20, 12))
    
    # 1. Validation Accuracy over epochs
    ax1 = plt.subplot(2, 3, 1)
    for result in results:
        wd = result['weight_decay']
        epochs = result['epochs']
        label = f"WD={wd}, E={epochs}"
        x = range(1, len(result['history']['val_acc']) + 1)
        ax1.plot(x, [acc*100 for acc in result['history']['val_acc']], 
                label=label, linewidth=2, alpha=0.7, color=wd_color_map[wd])
    ax1.set_xlabel('Epoch', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Validation Accuracy (%)', fontweight='bold', fontsize=11)
    ax1.set_title('Validation Accuracy Evolution', fontweight='bold', fontsize=12)
    ax1.legend(fontsize=8, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 2. Training Loss over epochs
    ax2 = plt.subplot(2, 3, 2)
    for result in results:
        wd = result['weight_decay']
        epochs = result['epochs']
        label = f"WD={wd}, E={epochs}"
        x = range(1, len(result['history']['train_loss']) + 1)
        ax2.plot(x, result['history']['train_loss'], 
                label=label, linewidth=2, alpha=0.7, color=wd_color_map[wd])
    ax2.set_xlabel('Epoch', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Training Loss', fontweight='bold', fontsize=11)
    ax2.set_title('Training Loss Evolution', fontweight='bold', fontsize=12)
    ax2.legend(fontsize=8, loc='best')
    ax2.grid(True, alpha=0.3)
    
    # 3. Validation F1 Score
    ax3 = plt.subplot(2, 3, 3)
    for result in results:
        wd = result['weight_decay']
        epochs = result['epochs']
        label = f"WD={wd}, E={epochs}"
        x = range(1, len(result['history']['val_f1']) + 1)
        ax3.plot(x, [f1*100 for f1 in result['history']['val_f1']], 
                label=label, linewidth=2, alpha=0.7, color=wd_color_map[wd])
    ax3.set_xlabel('Epoch', fontweight='bold', fontsize=11)
    ax3.set_ylabel('Validation F1 Score (%)', fontweight='bold', fontsize=11)
    ax3.set_title('Validation F1 Score Evolution', fontweight='bold', fontsize=12)
    ax3.legend(fontsize=8, loc='best')
    ax3.grid(True, alpha=0.3)
    
    # 4. Validation Kappa
    ax4 = plt.subplot(2, 3, 4)
    for result in results:
        wd = result['weight_decay']
        epochs = result['epochs']
        label = f"WD={wd}, E={epochs}"
        x = range(1, len(result['history']['val_kappa']) + 1)
        ax4.plot(x, result['history']['val_kappa'], 
                label=label, linewidth=2, alpha=0.7, color=wd_color_map[wd])
    ax4.set_xlabel('Epoch', fontweight='bold', fontsize=11)
    ax4.set_ylabel('Cohen\'s Kappa', fontweight='bold', fontsize=11)
    ax4.set_title('Validation Kappa Evolution', fontweight='bold', fontsize=12)
    ax4.legend(fontsize=8, loc='best')
    ax4.grid(True, alpha=0.3)
    
    # 5. Best validation accuracy vs weight_decay
    ax5 = plt.subplot(2, 3, 5)
    wds = [r['weight_decay'] for r in results]
    best_vals = [r['best_val_acc']*100 for r in results]
    ax5.plot(wds, best_vals, 'o-', label=f'{constant_epochs} epochs', 
             linewidth=2, markersize=10, color='royalblue')
    
    # Highlight best
    best_idx = np.argmax(best_vals)
    ax5.scatter([wds[best_idx]], [best_vals[best_idx]], 
               s=200, color='gold', edgecolors='red', linewidths=2, 
               marker='*', zorder=5, label='Best')
    
    ax5.set_xlabel('Weight Decay', fontweight='bold', fontsize=11)
    ax5.set_ylabel('Best Validation Accuracy (%)', fontweight='bold', fontsize=11)
    ax5.set_title(f'Best Val Accuracy vs Weight Decay ({constant_epochs} epochs)', 
                  fontweight='bold', fontsize=12)
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.set_xscale('log')
    
    # 6. Test accuracy comparison
    ax6 = plt.subplot(2, 3, 6)
    x_pos = np.arange(len(results))
    test_accs = [r['test_metrics']['acc']*100 for r in results]
    colors_bar = [wd_color_map[r['weight_decay']] for r in results]
    bars = ax6.bar(x_pos, test_accs, color=colors_bar, alpha=0.7)
    
    # Add value labels
    for i, (bar, result) in enumerate(zip(bars, results)):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    ax6.set_xlabel('Experiment', fontweight='bold', fontsize=11)
    ax6.set_ylabel('Test Accuracy (%)', fontweight='bold', fontsize=11)
    ax6.set_title('Final Test Accuracy Comparison', fontweight='bold', fontsize=12)
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels([f"WD={r['weight_decay']}\nE={r['epochs']}" 
                         for r in results], fontsize=8, rotation=45, ha='right')
    ax6.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Weight Decay Optimization Results - PhysioNet-MI', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_path}/optimization_curves.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}/optimization_curves.png")
    plt.show()
    
    # Figure 2: Summary comparison bar chart
    fig2 = plt.figure(figsize=(16, 10))
    
    metrics_to_plot = [
        ('acc', 'Test Accuracy'),
        ('balanced_acc', 'Test Balanced Accuracy'),
        ('f1', 'Test F1 Score'),
        ('kappa', 'Cohen\'s Kappa')
    ]
    
    for idx, (metric_key, metric_title) in enumerate(metrics_to_plot, 1):
        ax = plt.subplot(2, 2, idx)
        
        wds = [r['weight_decay'] for r in results]
        values = [r['test_metrics'][metric_key] for r in results]
        
        # Convert to percentage for accuracy/f1
        if metric_key in ['acc', 'balanced_acc', 'f1']:
            values = [v * 100 for v in values]
            ylabel = 'Score (%)'
        else:
            ylabel = 'Score'
        
        # Create bar chart
        bars = ax.bar(range(len(wds)), values, color=colors, alpha=0.7, edgecolor='black')
        
        # Highlight best
        best_idx = np.argmax(values)
        bars[best_idx].set_color('gold')
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(2)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            if metric_key in ['acc', 'balanced_acc', 'f1']:
                label = f'{val:.1f}%'
            else:
                label = f'{val:.3f}'
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   label, ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Weight Decay', fontweight='bold', fontsize=11)
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=11)
        ax.set_title(f'{metric_title} ({constant_epochs} epochs)', fontweight='bold', fontsize=12)
        ax.set_xticks(range(len(wds)))
        ax.set_xticklabels([f'{wd}' for wd in wds], rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle(f'Weight Decay Optimization - Test Set Performance ({constant_epochs} epochs)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_path}/optimization_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}/optimization_comparison.png")
    plt.show()
    
    # Print summary table
    print("\n" + "="*100)
    print("📊 OPTIMIZATION RESULTS SUMMARY")
    print("="*100)
    print(f"{'WD':<10} {'Epochs':<10} {'Best Val':<12} {'Test Acc':<12} {'Bal Acc':<12} {'F1':<12} {'Kappa':<10}")
    print("-"*100)
    
    # Sort by test accuracy
    sorted_results = sorted(results, key=lambda x: x['test_metrics']['acc'], reverse=True)
    
    for result in sorted_results:
        print(f"{result['weight_decay']:<10} "
              f"{result['epochs']:<10} "
              f"{result['best_val_acc']*100:6.2f}%     "
              f"{result['test_metrics']['acc']*100:6.2f}%     "
              f"{result['test_metrics']['balanced_acc']*100:6.2f}%     "
              f"{result['test_metrics']['f1']*100:6.2f}%     "
              f"{result['test_metrics']['kappa']:6.4f}")
    
    print("="*100)
    
    # Find best configuration
    best_result = sorted_results[0]
    print(f"\n🏆 BEST CONFIGURATION:")
    print(f"   Weight Decay: {best_result['weight_decay']}")
    print(f"   Epochs: {best_result['epochs']}")
    print(f"   Test Accuracy: {best_result['test_metrics']['acc']*100:.2f}%")
    print(f"   Test Balanced Accuracy: {best_result['test_metrics']['balanced_acc']*100:.2f}%")
    print(f"   Test F1 Score: {best_result['test_metrics']['f1']*100:.2f}%")
    print(f"   Cohen's Kappa: {best_result['test_metrics']['kappa']:.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(description='Optimize weight_decay for EEGMamba')
    parser.add_argument('--weight_decays', nargs='+', type=float, default=[1, 5, 10, 15, 20],
                       help='List of weight_decay values to test (default: 1 5 10 15 20)')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs to train (constant for all experiments, default: 30)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--save_path', type=str, default='optimization_results',
                       help='Path to save results (default: optimization_results)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with fewer configurations')
    
    args = parser.parse_args()
    
    # Quick test mode
    if args.quick:
        print("⚡ Quick test mode enabled")
        args.weight_decays = [5, 15]
        args.epochs = 5
    
    print("="*80)
    print("🎯 EEGMamba Weight Decay Optimization")
    print("="*80)
    print(f"Weight Decay values: {args.weight_decays}")
    print(f"Epochs (constant): {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Total experiments: {len(args.weight_decays)}")
    print()
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ CUDA not available - training will be slow")
    print()
    
    # Run experiments
    results = []
    total_experiments = len(args.weight_decays)
    experiment_num = 0
    
    for weight_decay in args.weight_decays:
        experiment_num += 1
        print(f"\n{'#'*80}")
        print(f"Experiment {experiment_num}/{total_experiments}")
        print(f"{'#'*80}")
        
        result = run_experiment(
            weight_decay=weight_decay,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            verbose=True
        )
        results.append(result)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"results/{args.save_path}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    with open(f'{save_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to: {save_dir}/results.json")
    
    # Generate plots
    print("\n📊 Generating comparison charts...")
    plot_optimization_results(results, save_dir)
    
    print(f"\n✅ Optimization complete! Results saved in: {save_dir}/")


if __name__ == "__main__":
    main()
