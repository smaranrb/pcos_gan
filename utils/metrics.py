"""
Evaluation metrics for PCOS classification
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


class MetricsCalculator:
    """Calculate comprehensive metrics for binary classification"""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred, y_prob=None):
        """
        Calculate all metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional, for AUC)
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='binary')
        metrics['recall'] = recall_score(y_true, y_pred, average='binary')
        metrics['f1'] = f1_score(y_true, y_pred, average='binary')
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Medical metrics
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0  # Same as recall
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        
        # Confusion matrix components
        metrics['tp'] = int(tp)
        metrics['tn'] = int(tn)
        metrics['fp'] = int(fp)
        metrics['fn'] = int(fn)
        
        # AUC if probabilities provided
        if y_prob is not None:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_prob)
            except:
                metrics['auc'] = 0.0
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics, title="Metrics"):
        """Pretty print metrics"""
        print(f"\n{title}")
        print("=" * 70)
        print(f"Accuracy:    {metrics['accuracy']:.4f}")
        print(f"Precision:   {metrics['precision']:.4f}")
        print(f"Recall:      {metrics['recall']:.4f}")
        print(f"F1-Score:    {metrics['f1']:.4f}")
        print(f"\nMedical Metrics:")
        print(f"Sensitivity: {metrics['sensitivity']:.4f}  (True Positive Rate)")
        print(f"Specificity: {metrics['specificity']:.4f}  (True Negative Rate)")
        print(f"PPV:         {metrics['ppv']:.4f}  (Positive Predictive Value)")
        print(f"NPV:         {metrics['npv']:.4f}  (Negative Predictive Value)")
        
        if 'auc' in metrics:
            print(f"\nAUC-ROC:     {metrics['auc']:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"              Predicted")
        print(f"              Neg    Pos")
        print(f"Actual  Neg   {metrics['tn']:<6} {metrics['fp']:<6}")
        print(f"        Pos   {metrics['fn']:<6} {metrics['tp']:<6}")
        print("=" * 70)


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    maxk = max(topk)
    batch_size = target.size(0)
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    
    return res


if __name__ == '__main__':
    # Test metrics
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.2, 0.85, 0.6])
    
    calc = MetricsCalculator()
    metrics = calc.calculate_metrics(y_true, y_pred, y_prob)
    calc.print_metrics(metrics, "Test Metrics")
