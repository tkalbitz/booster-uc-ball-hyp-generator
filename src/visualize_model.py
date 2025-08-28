import os
import sys

import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

from scale import unscale_x, unscale_y
from utils import get_flops

if __name__ != '__main__':
    exit(1)

# Set device
device = torch.device('cpu')  # Force CPU for visualization
print(f"Using device: {device}")

import models
from config import patch_height, patch_width, image_dir, testset_csv_collection
from csv_label_reader import load_csv_collection
from dataset_handling import create_dataset, yuv2rgb

png_files = {f.name: str(f) for f in image_dir.glob("**/*.png")}

test_img, test_labels, skipped_testset = load_csv_collection(testset_csv_collection, png_files)

print(f"Testset contains {len(test_img)} and we removed {skipped_testset} balls.")

ds = create_dataset(test_img, test_labels, 10, trainset=False)

model_file_loss = "/home/tkalbitz/naoTeamRepo/firmware_5.0/deploy/data/tflite/lc-object-hyp-gen-ball.h5"
model_file_acc = "/home/tkalbitz/naoTeamRepo/firmware_5.0/deploy/data/tflite/lc-object-hyp-gen-ball.h5"

model_loss = models.create_network_v2(patch_height, patch_width)
model_loss.load_state_dict(torch.load(model_file_loss, map_location=device))
model_loss.to(device)
model_loss.eval()

model_acc = models.create_network_v2(patch_height, patch_width)
model_acc.load_state_dict(torch.load(model_file_acc, map_location=device))
model_acc.to(device)
model_acc.eval()

try:
    flops = get_flops(model_loss, (3, patch_height, patch_width))
    print(f"Model has {flops / 1e6:.2f} MFlops")
except Exception as e:
    print(f"Could not calculate FLOPs: {e}")

it = iter(ds)

while True:
    image_batch, label_batch = next(it)
    
    # Convert to tensors and move to device
    image_batch = image_batch.to(device)
    
    with torch.no_grad():
        pred_loss = model_loss(image_batch)
        pred_acc = model_acc(image_batch)

    plt.figure(figsize=(30, 30))
    for i in range(min(9, len(image_batch))):
        ax = plt.subplot(3, 3, i + 1)
        
        # Convert image back to HWC format and to numpy
        image_np = image_batch[i].permute(1, 2, 0).cpu().numpy()
        image_rgb = yuv2rgb(image_np * 255.).clamp(0, 255).int().numpy()
        plt.imshow(image_rgb)
        
        label_true = label_batch[i]
        label_pred_loss = pred_loss[i].cpu()
        label_pred_acc = pred_acc[i].cpu()

        x_t = unscale_x(label_true[0]) + patch_width / 2
        y_t = unscale_y(label_true[1]) + patch_height / 2

        x_pl = unscale_x(label_pred_loss[0]) + patch_width / 2
        y_pl = unscale_y(label_pred_loss[1]) + patch_height / 2

        x_pa = unscale_x(label_pred_acc[0]) + patch_width / 2
        y_pa = unscale_y(label_pred_acc[1]) + patch_height / 2

        plt.gca().add_patch(Ellipse((x_t, y_t), 3, 3, linewidth=1, edgecolor='b', facecolor='none'))
        plt.gca().add_patch(Ellipse((x_pl, y_pl), 3, 3, linewidth=1, edgecolor='r', facecolor='none'))
        plt.gca().add_patch(Ellipse((x_pa, y_pa), 3, 3, linewidth=1, edgecolor='g', facecolor='none'))

        plt.axis("off")
    plt.waitforbuttonpress()
    plt.close()
