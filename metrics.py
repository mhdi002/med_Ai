import torch
import numpy as np

def dice_score(predictions, targets, smooth=1.0):
    """
    Calculate Dice score (F1 score) between predictions and targets
    
    Args:
        predictions: Tensor of predicted segmentation map (B, C, H, W)
        targets: Tensor of ground truth segmentation map (B, C, H, W)
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        dice: The Dice coefficient (F1 score)
    """
    # Flatten predictions and targets
    batch_size = predictions.size(0)
    predictions = predictions.view(batch_size, -1)
    targets = targets.view(batch_size, -1)
    
    # Calculate intersection and union
    intersection = (predictions * targets).sum(dim=1)
    predictions_sum = predictions.sum(dim=1)
    targets_sum = targets.sum(dim=1)
    
    # Calculate Dice coefficient
    dice = (2. * intersection + smooth) / (predictions_sum + targets_sum + smooth)
    
    return dice.mean().item()

def iou_score(predictions, targets, smooth=1.0):
    """
    Calculate IoU (Intersection over Union) score between predictions and targets
    
    Args:
        predictions: Tensor of predicted segmentation map (B, C, H, W)
        targets: Tensor of ground truth segmentation map (B, C, H, W)
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        iou: The IoU score (Jaccard Index)
    """
    # Flatten predictions and targets
    batch_size = predictions.size(0)
    predictions = predictions.view(batch_size, -1)
    targets = targets.view(batch_size, -1)
    
    # Calculate intersection and union
    intersection = (predictions * targets).sum(dim=1)
    union = predictions_sum = predictions.sum(dim=1) + targets.sum(dim=1) - intersection
    
    # Calculate IoU
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.mean().item()

def accuracy(predictions, targets, threshold=0.5):
    """
    Calculate pixel-wise accuracy between predictions and targets
    
    Args:
        predictions: Tensor of predicted segmentation map (B, C, H, W)
        targets: Tensor of ground truth segmentation map (B, C, H, W)
        threshold: Threshold to binarize predictions
        
    Returns:
        accuracy: The pixel-wise accuracy
    """
    # Binarize predictions
    binary_preds = (predictions > threshold).float()
    
    # Calculate accuracy
    correct = (binary_preds == targets).float().sum()
    total = torch.numel(targets)
    
    return (correct / total).item()

def sensitivity(predictions, targets, threshold=0.5, smooth=1.0):
    """
    Calculate sensitivity (recall) between predictions and targets
    
    Args:
        predictions: Tensor of predicted segmentation map (B, C, H, W)
        targets: Tensor of ground truth segmentation map (B, C, H, W)
        threshold: Threshold to binarize predictions
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        sensitivity: The sensitivity (recall)
    """
    # Binarize predictions
    binary_preds = (predictions > threshold).float()
    
    # Flatten predictions and targets
    batch_size = binary_preds.size(0)
    binary_preds = binary_preds.view(batch_size, -1)
    targets = targets.view(batch_size, -1)
    
    # Calculate true positives and false negatives
    tp = (binary_preds * targets).sum(dim=1)
    fn = ((1 - binary_preds) * targets).sum(dim=1)
    
    # Calculate sensitivity
    sens = (tp + smooth) / (tp + fn + smooth)
    
    return sens.mean().item()

def specificity(predictions, targets, threshold=0.5, smooth=1.0):
    """
    Calculate specificity between predictions and targets
    
    Args:
        predictions: Tensor of predicted segmentation map (B, C, H, W)
        targets: Tensor of ground truth segmentation map (B, C, H, W)
        threshold: Threshold to binarize predictions
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        specificity: The specificity
    """
    # Binarize predictions
    binary_preds = (predictions > threshold).float()
    
    # Flatten predictions and targets
    batch_size = binary_preds.size(0)
    binary_preds = binary_preds.view(batch_size, -1)
    targets = targets.view(batch_size, -1)
    
    # Calculate true negatives and false positives
    tn = ((1 - binary_preds) * (1 - targets)).sum(dim=1)
    fp = (binary_preds * (1 - targets)).sum(dim=1)
    
    # Calculate specificity
    spec = (tn + smooth) / (tn + fp + smooth)
    
    return spec.mean().item()

def evaluate_all_metrics(predictions, targets, threshold=0.5):
    """
    Calculate all metrics between predictions and targets
    
    Args:
        predictions: Tensor of predicted segmentation map (B, C, H, W)
        targets: Tensor of ground truth segmentation map (B, C, H, W)
        threshold: Threshold to binarize predictions
        
    Returns:
        metrics: Dictionary containing all metrics
    """
    metrics = {
        'dice': dice_score(predictions, targets),
        'iou': iou_score(predictions, targets),
        'accuracy': accuracy(predictions, targets, threshold),
        'sensitivity': sensitivity(predictions, targets, threshold),
        'specificity': specificity(predictions, targets, threshold)
    }
    
    return metrics