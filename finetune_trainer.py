import copy
import os
from timeit import default_timer as timer
import glob
import json
from datetime import datetime

import numpy as np
import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
from tqdm import tqdm

from finetune_evaluator import Evaluator


def cleanup_old_checkpoints(model_dir, max_checkpoints=10):
    """
    Keep only the most recent checkpoints to save disk space.
    
    Args:
        model_dir: Directory containing model checkpoints
        max_checkpoints: Maximum number of checkpoints to keep (default: 10)
    """
    # Get all .pth files in the model directory
    checkpoint_files = glob.glob(os.path.join(model_dir, "*.pth"))
    
    if len(checkpoint_files) <= max_checkpoints:
        return  # No need to delete anything
    
    # Sort by modification time (oldest first)
    checkpoint_files.sort(key=os.path.getmtime)
    
    # Delete oldest checkpoints
    num_to_delete = len(checkpoint_files) - max_checkpoints
    for checkpoint_path in checkpoint_files[:num_to_delete]:
        try:
            os.remove(checkpoint_path)
            print(f"🗑️  Deleted old checkpoint: {os.path.basename(checkpoint_path)}")
        except Exception as e:
            print(f"⚠️  Could not delete {checkpoint_path}: {e}")


class Trainer(object):
    def __init__(self, params, data_loader, model):
        self.params = params
        self.data_loader = data_loader

        self.val_eval = Evaluator(params, self.data_loader['val'])
        self.test_eval = Evaluator(params, self.data_loader['test'])

        self.model = model.cuda()
        if self.params.downstream_dataset in ['FACED', 'SEED-V', 'PhysioNet-MI', 'ISRUC', 'BCIC2020-3', 'TUEV', 'BCIC-IV-2a']:
            self.criterion = CrossEntropyLoss(label_smoothing=self.params.label_smoothing).cuda()
        elif self.params.downstream_dataset in ['SHU-MI', 'CHB-MIT', 'Mumtaz2016', 'MentalArithmetic', 'TUAB']:
            self.criterion = BCEWithLogitsLoss().cuda()
        elif self.params.downstream_dataset == 'SEED-VIG':
            self.criterion = MSELoss().cuda()

        self.best_model_states = None
        
        # Initialize training history for logging
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
        
        # Create log file path
        self.log_file = os.path.join(self.params.model_dir, 'training_log.json')
        if not os.path.isdir(self.params.model_dir):
            os.makedirs(self.params.model_dir)

        backbone_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)

                if params.frozen:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                other_params.append(param)

        if self.params.optimizer == 'AdamW':
            if self.params.multi_lr: # set different learning rates for different modules
                self.optimizer = torch.optim.AdamW([
                    {'params': backbone_params, 'lr': self.params.lr},
                    {'params': other_params, 'lr': self.params.lr * 5}
                ], weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params.lr,
                                                   weight_decay=self.params.weight_decay)
        else:
            if self.params.multi_lr:
                self.optimizer = torch.optim.SGD([
                    {'params': backbone_params, 'lr': self.params.lr},
                    {'params': other_params, 'lr': self.params.lr * 5}
                ],  momentum=0.9, weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.params.lr, momentum=0.9,
                                                 weight_decay=self.params.weight_decay)

        self.data_length = len(self.data_loader['train'])
        self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.params.epochs * self.data_length, eta_min=1e-6
        )
        print(self.model)

    def save_training_log(self):
        """Save training history to JSON file"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.training_history, f, indent=2)
            print(f"📊 Training log saved to: {self.log_file}")
        except Exception as e:
            print(f"⚠️ Warning: Could not save training log: {e}")

    def train_for_multiclass(self):
        f1_best = 0
        kappa_best = 0
        acc_best = 0
        cm_best = None
        best_f1_epoch = 0
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()
                pred = self.model(x)
                if self.params.downstream_dataset == 'ISRUC':
                    loss = self.criterion(pred.transpose(1, 2), y)
                else:
                    loss = self.criterion(pred, y)

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                acc, kappa, f1, cm = self.val_eval.get_metrics_for_multiclass(self.model)
                
                # Log metrics
                self.training_history['train_loss'].append(float(np.mean(losses)))
                self.training_history['val_acc'].append(float(acc))
                self.training_history['val_kappa'].append(float(kappa))
                self.training_history['val_f1'].append(float(f1))
                self.training_history['learning_rate'].append(float(optim_state['param_groups'][0]['lr']))
                self.training_history['epoch_time'].append(float((timer() - start_time) / 60))
                
                print(
                    "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        acc,
                        kappa,
                        f1,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                # print(cm)
                if kappa > kappa_best:
                    print("kappa increasing....saving weights !! ")
                    print("Val Evaluation: acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(
                        acc,
                        kappa,
                        f1,
                    ))
                    best_f1_epoch = epoch + 1
                    acc_best = acc
                    kappa_best = kappa
                    f1_best = f1
                    cm_best = cm
                    self.best_model_states = copy.deepcopy(self.model.state_dict())
        
        # Handle case where model never improved
        if self.best_model_states is None:
            print("⚠️ Warning: Model never improved on validation set. Using final epoch weights.")
            self.best_model_states = self.model.state_dict()
            best_f1_epoch = self.params.epochs
        
        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            acc, kappa, f1, cm = self.test_eval.get_metrics_for_multiclass(self.model)
            
            # Log test results
            self.training_history['test_results'] = {
                'acc': float(acc),
                'kappa': float(kappa),
                'f1': float(f1),
                'confusion_matrix': cm.tolist() if hasattr(cm, 'tolist') else cm,
                'best_epoch': int(best_f1_epoch)
            }
            
            # Save training log
            self.save_training_log()
            
            print("***************************Test results************************")
            print(
                "Test Evaluation: acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(
                    acc,
                    kappa,
                    f1,
                )
            )
            print(cm)
            
            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)

            model_path = self.params.model_dir + "/epoch{}_acc_{:.5f}_kappa_{:.5f}_f1_{:.5f}.pth".format(best_f1_epoch, acc, kappa, f1)
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)
            
            # Clean up old checkpoints to save disk space
            cleanup_old_checkpoints(self.params.model_dir, max_checkpoints=10)


    def train_for_binaryclass(self):
        acc_best = 0
        roc_auc_best = 0
        pr_auc_best = 0
        cm_best = None
        best_f1_epoch = 0
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()
                pred = self.model(x)

                loss = self.criterion(pred, y)

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                acc, pr_auc, roc_auc, cm = self.val_eval.get_metrics_for_binaryclass(self.model)
                
                # Log metrics
                self.training_history['train_loss'].append(float(np.mean(losses)))
                self.training_history['val_acc'].append(float(acc))
                self.training_history['val_pr_auc'].append(float(pr_auc))
                self.training_history['val_roc_auc'].append(float(roc_auc))
                self.training_history['learning_rate'].append(float(optim_state['param_groups'][0]['lr']))
                self.training_history['epoch_time'].append(float((timer() - start_time) / 60))
                
                print(
                    "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        acc,
                        pr_auc,
                        roc_auc,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                print(cm)
                if roc_auc > roc_auc_best:
                    print("roc_auc increasing....saving weights !! ")
                    print("Val Evaluation: acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}".format(
                        acc,
                        pr_auc,
                        roc_auc,
                    ))
                    best_f1_epoch = epoch + 1
                    acc_best = acc
                    pr_auc_best = pr_auc
                    roc_auc_best = roc_auc
                    cm_best = cm
                    self.best_model_states = copy.deepcopy(self.model.state_dict())
        
        # Handle case where model never improved
        if self.best_model_states is None:
            print("⚠️ Warning: Model never improved on validation set. Using final epoch weights.")
            self.best_model_states = self.model.state_dict()
            best_f1_epoch = self.params.epochs
        
        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            acc, pr_auc, roc_auc, cm = self.test_eval.get_metrics_for_binaryclass(self.model)
            
            # Log test results
            self.training_history['test_results'] = {
                'acc': float(acc),
                'pr_auc': float(pr_auc),
                'roc_auc': float(roc_auc),
                'confusion_matrix': cm.tolist() if hasattr(cm, 'tolist') else cm,
                'best_epoch': int(best_f1_epoch)
            }
            
            # Save training log
            self.save_training_log()
            
            print("***************************Test results************************")
            print(
                "Test Evaluation: acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}".format(
                    acc,
                    pr_auc,
                    roc_auc,
                )
            )
            print(cm)
            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            model_path = self.params.model_dir + "/epoch{}_acc_{:.5f}_pr_{:.5f}_roc_{:.5f}.pth".format(best_f1_epoch, acc, pr_auc, roc_auc)
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)
            
            # Clean up old checkpoints to save disk space
            cleanup_old_checkpoints(self.params.model_dir, max_checkpoints=10)

    def train_for_regression(self):
        corrcoef_best = 0
        r2_best = 0
        rmse_best = 0
        best_r2_epoch = 0
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()
                pred = self.model(x)
                loss = self.criterion(pred, y)

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                corrcoef, r2, rmse = self.val_eval.get_metrics_for_regression(self.model)
                
                # Log metrics
                self.training_history['train_loss'].append(float(np.mean(losses)))
                self.training_history['val_corrcoef'].append(float(corrcoef))
                self.training_history['val_r2'].append(float(r2))
                self.training_history['val_rmse'].append(float(rmse))
                self.training_history['learning_rate'].append(float(optim_state['param_groups'][0]['lr']))
                self.training_history['epoch_time'].append(float((timer() - start_time) / 60))
                
                print(
                    "Epoch {} : Training Loss: {:.5f}, corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        corrcoef,
                        r2,
                        rmse,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                if r2 > r2_best:
                    print("r2 increasing....saving weights !! ")
                    print("Val Evaluation: corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}".format(
                        corrcoef,
                        r2,
                        rmse,
                    ))
                    best_r2_epoch = epoch + 1
                    corrcoef_best = corrcoef
                    r2_best = r2
                    rmse_best = rmse
                    self.best_model_states = copy.deepcopy(self.model.state_dict())

        # Handle case where model never improved
        if self.best_model_states is None:
            print("⚠️ Warning: Model never improved on validation set. Using final epoch weights.")
            self.best_model_states = self.model.state_dict()
            best_r2_epoch = self.params.epochs

        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            corrcoef, r2, rmse = self.test_eval.get_metrics_for_regression(self.model)
            
            # Log test results
            self.training_history['test_results'] = {
                'corrcoef': float(corrcoef),
                'r2': float(r2),
                'rmse': float(rmse),
                'best_epoch': int(best_r2_epoch)
            }
            
            # Save training log
            self.save_training_log()
            
            print("***************************Test results************************")
            print(
                "Test Evaluation: corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}".format(
                    corrcoef,
                    r2,
                    rmse,
                )
            )

            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            model_path = self.params.model_dir + "/epoch{}_corrcoef_{:.5f}_r2_{:.5f}_rmse_{:.5f}.pth".format(best_r2_epoch, corrcoef, r2, rmse)
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)
            
            # Clean up old checkpoints to save disk space
            cleanup_old_checkpoints(self.params.model_dir, max_checkpoints=10)