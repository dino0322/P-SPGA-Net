import os
import numpy as np
import torch
import torch.nn as nn 
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

import datetime
from zoneinfo import ZoneInfo

from utils.data_utils import save_csv, plot_confusion_matrix, roc_plot

# =========================
# Single-epoch training
# =========================
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0

    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        preds = torch.argmax(outputs, dim=1)
        total_correct += (preds == y_batch).sum().item()
        total_samples += y_batch.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


# =========================
# Shared validation/test evaluation
# =========================
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    all_labels, all_preds, all_probs, all_losses = [], [], [], []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            preds = torch.argmax(outputs, dim=1)
            probs = torch.softmax(outputs, dim=1)[:, 1] if outputs.size(1) > 1 else torch.sigmoid(outputs)

            total_correct += (preds == y_batch).sum().item()
            total_samples += y_batch.size(0)

            all_labels.extend(y_batch.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_losses.extend([loss.item()] * X_batch.size(0))  # Store batch loss per sample.

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    print("val unique labels:", np.unique(all_labels, return_counts=True))
    print("val unique preds:", np.unique(all_preds, return_counts=True))
    return avg_loss, avg_acc, np.array(all_labels), np.array(all_preds), np.array(all_probs), np.array(all_losses)


# =========================
# Bootstrap confidence intervals
# =========================
def bootstrap_ci(metric_fn, y_true, y_pred, probs=None, n_bootstrap=1000, alpha=0.95):
    rng = np.random.default_rng()
    n = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        y_true_sample = y_true[idx]
        y_pred_sample = y_pred[idx]
        try:
            if probs is not None:
                if len(np.unique(y_true_sample)) < 2:
                    continue
                probs_sample = probs[idx]
                score = metric_fn(y_true_sample, probs_sample)
            else:
                score = metric_fn(y_true_sample, y_pred_sample)
        except Exception:
            continue
        if np.isfinite(score):
            scores.append(score)

    if not scores:
        return [np.nan, np.nan, np.nan]

    lower = np.percentile(scores, ((1 - alpha) / 2) * 100)
    upper = np.percentile(scores, (1 - (1 - alpha) / 2) * 100)
    return [np.mean(scores), lower, upper]


def bootstrap_ci_loss(values, n_bootstrap=1000, alpha=0.95):
    rng = np.random.default_rng()
    n = len(values)
    means = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        sample = values[idx]
        means.append(sample.mean())
    lower = np.percentile(means, ((1 - alpha) / 2) * 100)
    upper = np.percentile(means, (1 - (1 - alpha) / 2) * 100)
    return [np.mean(means), lower, upper]

# =========================
# Device selection
# =========================
def get_device(prefer='auto'):
    if prefer == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    elif prefer == 'mps' and torch.backends.mps.is_available():
        return torch.device('mps')
    elif prefer == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
    return torch.device('cpu')


# =========================
# Full training/evaluation loop
# =========================
def train_full_loop(train_loader, val_loader, test_loader, model_dicts, class_name, fold_num,
                    lr=1e-4, epochs=50, patience=10, device='auto'):

    device = get_device(device)
    print(f"Current Using Device: {device}")

    for model_name, model in model_dicts.items():
        print(f"\n=== Training {model_name} on Fold {fold_num} ===")

        save_path = f'./model_save/{class_name}/fold{fold_num}/{model_name}.pt'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_acc = 0
        patience_counter = 0
        
        try:
            model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
            print(f"Loaded existing model for {model_name}")
            training_needed = False
        except Exception as e:
            print(f"No saved model found, training from scratch: {e}")
            training_needed = True

        best_val_acc = 0
        patience_counter = 0

        # -----------------------------
        # Training
        # -----------------------------
        if training_needed:
            for epoch in range(epochs):
                train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
                val_loss, val_acc, _, _, _, _ = evaluate_model(model, val_loader, criterion, device)

                scheduler.step()

                print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    torch.save(model.state_dict(), save_path)
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        # -----------------------------
        # Test evaluation
        # -----------------------------
        if os.path.exists(save_path):
            model.load_state_dict(torch.load(save_path, map_location=device))
        else: 
            print(f"No Checkpoint saved for {model_name}; using current model weights")
        test_loss, test_acc, y_true, y_pred, y_prob, all_losses = evaluate_model(model, test_loader, criterion, device)



        # Bootstrap metrics
        test_loss_ci = bootstrap_ci_loss(all_losses)
        test_acc_ci = bootstrap_ci(balanced_accuracy_score, y_true, y_pred)
        test_recall_ci = bootstrap_ci(lambda y, yhat: recall_score(y, yhat, average='macro', zero_division=0), y_true, y_pred)
        test_prec_ci = bootstrap_ci(lambda y, yhat: precision_score(y, yhat, average='macro', zero_division=0), y_true, y_pred)
        test_f1_ci = bootstrap_ci(lambda y, yhat: f1_score(y, yhat, average='macro', zero_division=0), y_true, y_pred)

        if len(np.unique(y_true)) < 2:
            print("ROC AUC calculation skipped: only one class in y_true")
            test_auc_ci = [np.nan, np.nan, np.nan]
        else:
            test_auc_ci = bootstrap_ci(lambda y, p: roc_auc_score(y, p), y_true, y_pred, probs=y_prob)

        # -----------------------------
        # Save results
        # -----------------------------
        now = datetime.datetime.now(ZoneInfo("Asia/Seoul"))
        save_csv(f'{model_name}_{fold_num}', test_acc_ci, test_loss_ci, test_recall_ci, 
                 test_prec_ci, test_f1_ci, test_auc_ci, class_name, now)
 
        plot_confusion_matrix(y_true, y_pred, model_name, class_name, fold_num)
        roc_plot(y_true, y_prob, model_name, class_name, fold_num)
