import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, predictions, targets):
        # Debugging: Print shapes
        print(f"Predictions shape: {predictions.shape}")
        print(f"Targets shape: {targets.shape}")

        # Flatten tensors
        batch_size = predictions.size(0)
        predictions = predictions.view(batch_size, -1)
        targets = targets.view(batch_size, -1)

        # Compute intersection and union
        intersection = (predictions * targets).sum(dim=1)
        predictions_sum = predictions.sum(dim=1)
        targets_sum = targets.sum(dim=1)

        # Calculate Dice coefficient
        dice = (2. * intersection + self.smooth) / (predictions_sum + targets_sum + self.smooth)

        # Return Dice loss (1 - dice coefficient)
        return 1 - dice.mean()

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, smooth=1.0):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.dice_loss = DiceLoss(smooth=smooth)
        
    def forward(self, predictions, targets):
        """
        Combines Binary Cross Entropy and Dice Loss for better segmentation results
        
        Args:
            predictions: Tensor of predicted segmentation map (B, C, H, W)
            targets: Tensor of ground truth segmentation map (B, C, H, W)
            
        Returns:
            combined_loss: The weighted sum of BCE and Dice loss
        """
        # Compute BCE Loss
        bce_loss = F.binary_cross_entropy(predictions, targets, reduction='mean')
        
        # Compute Dice Loss
        dice_loss = self.dice_loss(predictions, targets)
        
        # Combine the losses
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss
