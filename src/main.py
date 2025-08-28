import os
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from custom_metrics import FoundBallMetric
from scale import unscale_x, unscale_y
from utils import get_flops

# Set device
device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

import models
from csv_label_reader import load_csv_collection
from dataset_handling import create_dataset
from config import patch_width, patch_height, image_dir, testset_csv_collection, trainingset_csv_collection

batch_size_train: int = 64
batch_size_test: int = 128

png_files: dict[str, str] = {f.name: str(f) for f in image_dir.glob("**/*.png")}

train_img: list[str]
train_labels: list[tuple[int, int, int, int]]
skipped_trainingset: int
train_img, train_labels, skipped_trainingset = load_csv_collection(trainingset_csv_collection, png_files)

test_img: list[str]
test_labels: list[tuple[int, int, int, int]]
skipped_testset: int
test_img, test_labels, skipped_testset = load_csv_collection(testset_csv_collection, png_files)

print(f"Trainingset contains {len(train_img)} and we removed {skipped_trainingset} balls.")
print(f"Testset contains {len(test_img)} and we removed {skipped_testset} balls.")

train_ds: torch.utils.data.DataLoader = create_dataset(train_img, train_labels, batch_size_train)
test_ds: torch.utils.data.DataLoader = create_dataset(test_img, test_labels, batch_size_test, trainset=False)

train_steps_per_epoch: int = (int(len(train_img) / batch_size_train) + 1)
test_steps_per_epoch: int = (int(len(test_img) / batch_size_test))

print(f"Train steps {train_steps_per_epoch}, test steps {test_steps_per_epoch}")

#
# According to https://stackoverflow.com/questions/46784648/mean-euclidean-distance-in-tensorflow the sqrt is unstable
# near 0 and to avoid. Better to use the squared distance.
#
#@tf.function
class DistanceLoss(nn.Module):
    """Custom distance loss for ball detection."""
    
    def __init__(self) -> None:
        super().__init__()
    
    def _logcosh(self, x: torch.Tensor) -> torch.Tensor:
        """Stable log-cosh function."""
        return x + torch.nn.functional.softplus(-2. * x) - torch.log(torch.tensor(2.))
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_t = torch.stack([unscale_x(y_true[:,0]), unscale_y(y_true[:,1])], dim=1)
        y_p = torch.stack([unscale_x(y_pred[:,0]), unscale_y(y_pred[:,1])], dim=1)
        
        r = self._logcosh(y_p - y_t)
        e = torch.exp(3. / y_true[:,2])
        r = r * e.unsqueeze(1)
        return torch.mean(r)


def calculate_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Calculate accuracy for position prediction."""
    # Simple accuracy based on how close the predictions are
    diff = torch.abs(y_pred - y_true[:, :2])
    threshold = 0.1  # Adjust threshold as needed
    accuracy = torch.mean((diff < threshold).float())
    return accuracy


def create_model() -> tuple[torch.nn.Module, str, str]:
    """Create and initialize the model."""
    model = models.create_network_v2(patch_height, patch_width)
    model = model.to(device)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--test':
            exit(1)
        else:
            model_name = sys.argv[1]
    else:
        model_name = "yuv_" + time.strftime("%Y-%m-%d-%H-%M-%S")
    
    model_dir = "model/" + model_name
    os.makedirs(model_dir)

    for f in Path(os.path.realpath(__file__)).parent.glob("*.py"):
        shutil.copy2(str(f.absolute()), model_dir)
    
    log_dir = './logs/' + model_name
    os.makedirs(log_dir)
    
    # Save model architecture
    torch.save(model.state_dict(), os.path.join(model_dir, 'model_weights.pth'))
    
    # Calculate FLOPs
    try:
        flops = get_flops(model, (3, patch_height, patch_width))
        print(f"Model has {flops / 1e6:.2f} MFlops")
    except Exception as e:
        print(f"Could not calculate FLOPs: {e}")

    return model, model_dir, log_dir


def create_training_components(model: torch.nn.Module, model_dir: str, log_dir: str) -> tuple[torch.optim.Optimizer, torch.nn.Module, torch.optim.lr_scheduler.ReduceLROnPlateau, SummaryWriter, object]:
    """Create optimizer, loss function, and logging components."""
    optimizer = optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
    criterion = DistanceLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=15, 
        verbose=True, min_lr=1e-7
    )
    
    writer = SummaryWriter(log_dir)
    
    # Create CSV logger manually
    csv_file = open(os.path.join(model_dir, 'training.csv'), 'w')
    csv_file.write('epoch,loss,val_loss,accuracy,val_accuracy,found_balls,val_found_balls\n')
    
    return optimizer, criterion, scheduler, writer, csv_file


def train_model(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau, writer: SummaryWriter, csv_file: object, model_dir: str, epochs: int = 10000) -> None:
    """Training loop for the model."""
    best_val_loss = float('inf')
    best_val_found_balls = 0.0
    patience_counter = 0
    early_stopping_patience = 60
    
    train_metric = FoundBallMetric()
    val_metric = FoundBallMetric()
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_metric.reset_states()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += calculate_accuracy(output, target).item()
            train_metric.update_state(target, output)
            
            if batch_idx >= train_steps_per_epoch:
                break
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_metric.reset_states()
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                val_acc += calculate_accuracy(output, target).item()
                val_metric.update_state(target, output)
                
                if batch_idx >= test_steps_per_epoch:
                    break
        
        # Calculate averages
        train_loss /= train_steps_per_epoch
        train_acc /= train_steps_per_epoch
        val_loss /= test_steps_per_epoch
        val_acc /= test_steps_per_epoch
        
        train_found_balls = train_metric.result()
        val_found_balls = val_metric.result()
        
        # Logging
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.6f}, Train Found Balls: {train_found_balls:.6f}')
        print(f'Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.6f}, Val Found Balls: {val_found_balls:.6f}')
        
        # TensorBoard logging
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        writer.add_scalar('FoundBalls/Train', train_found_balls, epoch)
        writer.add_scalar('FoundBalls/Validation', val_found_balls, epoch)
        
        # CSV logging
        csv_file.write(f'{epoch+1},{train_loss},{val_loss},{train_acc},{val_acc},{train_found_balls},{val_found_balls}\n')
        csv_file.flush()
        
        # Model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 
                      os.path.join(model_dir, f'weights.loss.{epoch+1:03d}-{val_loss:.6f}-{val_found_balls:.6f}.pth'))
            patience_counter = 0
        else:
            patience_counter += 1
        
        if val_found_balls > best_val_found_balls:
            best_val_found_balls = val_found_balls
            torch.save(model.state_dict(), 
                      os.path.join(model_dir, f'weights.balls.{epoch+1:03d}-{val_found_balls:.6f}-{val_loss:.6f}.pth'))
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    writer.close()
    csv_file.close()


# Main execution
model, model_dir, log_dir = create_model()
optimizer, criterion, scheduler, writer, csv_file = create_training_components(model, model_dir, log_dir)

train_model(model, train_ds, test_ds, optimizer, criterion, scheduler, writer, csv_file, model_dir)

