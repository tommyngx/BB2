"""
Testing utilities for classification models
"""

import torch
from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def evaluate_classification_model(model, test_loader, device):
    """Evaluate model and return predictions"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total

    return {
        "accuracy": accuracy,
        "preds": all_preds,
        "labels": all_labels,
        "probs": all_probs,
    }


def compute_classification_metrics(all_preds, all_labels, all_probs, class_names):
    """Compute detailed classification metrics"""
    metrics = {}

    # AUC
    try:
        if len(class_names) == 2:
            probs_class1 = [p[1] for p in all_probs]
            auc = roc_auc_score(all_labels, probs_class1)
            metrics["auc"] = auc
        else:
            auc = roc_auc_score(
                all_labels, all_probs, multi_class="ovo", average="macro"
            )
            metrics["auc"] = auc
    except:
        metrics["auc"] = None

    # Precision & Recall
    try:
        precision = precision_score(
            all_labels,
            all_preds,
            average="macro" if len(class_names) > 2 else "binary",
            zero_division=0,
        )
        recall = recall_score(
            all_labels,
            all_preds,
            average="macro" if len(class_names) > 2 else "binary",
            zero_division=0,
        )
        metrics["precision"] = precision * 100
        metrics["recall"] = recall * 100
    except:
        metrics["precision"] = 0.0
        metrics["recall"] = 0.0

    # Sensitivity & Specificity (binary only)
    cm = confusion_matrix(all_labels, all_preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics["sensitivity"] = sensitivity * 100
        metrics["specificity"] = specificity * 100

    return metrics


def print_test_metrics(metrics, class_names, all_labels, all_preds):
    """Print test metrics"""
    print("\n" + "=" * 60)
    print("Test Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.2f}%")
    if metrics.get("auc") is not None:
        print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  Precision: {metrics.get('precision', 0.0):.2f}%")
    print(f"  Recall: {metrics.get('recall', 0.0):.2f}%")
    if "sensitivity" in metrics:
        print(f"  Sensitivity: {metrics['sensitivity']:.2f}%")
        print(f"  Specificity: {metrics['specificity']:.2f}%")
    print("=" * 60)

    print("\nClassification Report:")
    print(
        classification_report(
            all_labels, all_preds, target_names=class_names, digits=4, zero_division=0
        )
    )
