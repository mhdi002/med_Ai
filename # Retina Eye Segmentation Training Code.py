import os
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast  # Import for mixed precision
# Import the modules we've just created
from r2unet import R2UNet
from loss import DiceLoss, CombinedLoss
from metrics import dice_score, iou_score, evaluate_all_metrics

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Enable benchmark mode in cudnn
torch.backends.cudnn.benchmark = True

# Custom dataset for segmentation
class RetinalSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        if self.transform:
            image = self.transform(image)
            mask = transforms.Resize((256, 256))(mask)
            mask = transforms.ToTensor()(mask)
        
        mask = (mask > 0.5).float()
        
        return image, mask

# Dataset paths
data_dir = r"C:\Users\mahdi\Downloads\FIVES A Fundus Image Dataset for AI-based Vessel Segmentation\FIVES A Fundus Image Dataset for AI-based Vessel Segmentation"
train_images_dir = os.path.join(data_dir, 'train', 'Original')
train_masks_dir = os.path.join(data_dir, 'train', 'Ground truth')
test_images_dir = os.path.join(data_dir, 'test', 'Original')
test_masks_dir = os.path.join(data_dir, 'test', 'Ground truth')

# Create directories if they don't exist
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_masks_dir, exist_ok=True)
os.makedirs(test_images_dir, exist_ok=True)
os.makedirs(test_masks_dir, exist_ok=True)

# Data transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Datasets
train_dataset = RetinalSegmentationDataset(
    images_dir=train_images_dir,
    masks_dir=train_masks_dir,
    transform=transform
)

test_dataset = RetinalSegmentationDataset(
    images_dir=test_images_dir,
    masks_dir=test_masks_dir,
    transform=transform
)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)

# Model initialization
model = R2UNet(in_channels=3, out_channels=1).to(device)
criterion = DiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)

# Mixed precision scaler
scaler = GradScaler()

# Define checkpoints directory
checkpoint_dir = os.path.join(data_dir, 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)

# Training loop
num_epochs = 4
train_loss_history = []
best_train_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    train_dice = 0.0
    
    for i, (images, masks) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast():
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            loss = criterion(probs, masks)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        
        batch_dice = dice_score(probs, masks)
        train_dice += batch_dice
    
    avg_train_loss = running_loss / len(train_loader)
    avg_train_dice = train_dice / len(train_loader)
    train_loss_history.append(avg_train_loss)
    
    if avg_train_loss < best_train_loss:
        best_train_loss = avg_train_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'train_dice': avg_train_dice,
        }, os.path.join(checkpoint_dir, 'best_model.pth'))
        print(f'Model saved at epoch {epoch+1}')
    
    if (epoch+1) % 5 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'train_dice': avg_train_dice,
        }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}')
    print(f'Train Dice: {avg_train_dice:.4f}')

# Remove validation loss plotting
plt.figure(figsize=(10, 5))
plt.plot(train_loss_history, label='Train Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(checkpoint_dir, 'loss_curves.png'))
plt.show()

# Load best model for evaluation
checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pth'))
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded best model from epoch {checkpoint['epoch']+1} with validation Dice score: {checkpoint['val_dice']:.4f}")

# Evaluate on test set
model.eval()
test_loss = 0.0
test_dice = 0.0

with torch.no_grad():
    for images, masks in test_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs)
        
        loss = criterion(probs, masks)
        test_loss += loss.item()
        
        # Calculate Dice score
        batch_dice = dice_score(probs, masks)
        test_dice += batch_dice

avg_test_loss = test_loss / len(test_loader)
avg_test_dice = test_dice / len(test_loader)
print(f'Test Loss: {avg_test_loss:.4f}, Test Dice Score: {avg_test_dice:.4f}')

# Visualization of predictions
def denormalize(tensor):
    """Denormalize image tensor to show actual image"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

# Create directory for saving test results
results_dir = os.path.join(data_dir, 'test_results')
os.makedirs(results_dir, exist_ok=True)

model.eval()
with torch.no_grad():
    for i, (images, masks) in enumerate(test_loader):
        if i >= 5:  # Only visualize first 5 batches
            break
            
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        
        # Visualize results for each sample in the batch
        for j in range(min(4, images.size(0))):  # Show up to 4 images per batch
            image = denormalize(images[j]).cpu()
            mask = masks[j].cpu()
            pred = preds[j].cpu()
            
            # Calculate dice score for this sample
            sample_dice = dice_score(pred.unsqueeze(0), mask.unsqueeze(0))
            
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(image.numpy().transpose(1, 2, 0))  # Convert to HWC for matplotlib
            plt.title('Input Image')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(mask.numpy().squeeze(), cmap='gray')
            plt.title('Ground Truth')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(pred.numpy().squeeze(), cmap='gray')
            plt.title(f'Prediction (Dice: {sample_dice:.4f})')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f'batch_{i}_sample_{j}.png'))
            plt.show()

# Optional: Generate and save overlay views (prediction boundaries on original)
for i, (images, masks) in enumerate(test_loader):
    if i >= 3:  # Only process first 3 batches for overlay
        break
        
    images, masks = images.to(device), masks.to(device)
    outputs = model(images)
    probs = torch.sigmoid(outputs)
    preds = (probs > 0.5).float()
    
    for j in range(min(2, images.size(0))):  # Show up to 2 images per batch
        image = denormalize(images[j]).cpu().numpy().transpose(1, 2, 0)  # Convert to HWC
        mask = masks[j].cpu().numpy().squeeze()
        pred = preds[j].cpu().numpy().squeeze()
        
        # Create overlay image
        plt.figure(figsize=(10, 5))
        
        # Original image with ground truth contour
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.contour(mask, colors='r', levels=[0.5])
        plt.title('Ground Truth Contour')
        plt.axis('off')
        
        # Original image with prediction contour
        plt.subplot(1, 2, 2)
        plt.imshow(image)
        plt.contour(pred, colors='b', levels=[0.5])
        plt.title('Prediction Contour')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'overlay_batch_{i}_sample_{j}.png'))
        plt.show()

print(f"Results saved to {results_dir}")