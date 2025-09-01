import torch

import models
from config import patch_height, patch_width, image_dir, testset_csv_collection
from csv_label_reader import load_csv_collection
from dataset_handling import create_dataset_image_based
from scale import unscale_x, unscale_y

if __name__ != '__main__':
    exit(1)

# Set device
device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

png_files: dict[str, str] = {f.name: str(f) for f in image_dir.glob("**/*.png")}

test_img, test_labels, skipped_testset = load_csv_collection(testset_csv_collection, png_files)

print(f"Testset contains {len(test_img)} and we removed {skipped_testset} balls.")

batch_size: int = 1
steps: int = len(test_img) // batch_size + 1
ds = create_dataset_image_based(test_img, test_labels, batch_size)

# YUV 3830 model_file = "/home/tkalbitz/PycharmProjects/uc-ball-hyp-generator/model/yuv_2021-05-17-20-31-42/weights.acc.208-0.957764.h5"
model_file: str = "/home/tkalbitz/PycharmProjects/uc-ball-hyp-generator/model/yuv_2021-05-22-09-23-23/weights.balls.284-0.955393-1.368752.h5"

model = models.create_network_v2(patch_height, patch_width)
model.load_state_dict(torch.load(model_file, map_location=device))
model.to(device)
model.eval()

ball_sum: int = 0
it = iter(ds)

for s in range(steps):
    image_batch, label_batch = next(it)
    
    # Convert to tensors and move to device
    input_batch = image_batch[0].to(device)
    label_batch = label_batch.to(device)

    with torch.no_grad():
        pred = model(input_batch)

    x_t = torch.tensor(unscale_x(label_batch[:,0])) + patch_width / 2
    y_t = torch.tensor(unscale_y(label_batch[:,1])) + patch_height / 2

    x_p = torch.tensor(unscale_x(pred[:,0])) + patch_width / 2
    y_p = torch.tensor(unscale_y(pred[:,1])) + patch_height / 2

    d = torch.sqrt((x_t - x_p)**2 + (y_t - y_p)**2)
    r = d < label_batch[:,2]
    found = torch.sum(r.int()) > 0
    ball_sum += int(found.item())


print(f"Found in total {ball_sum} / {steps * batch_size}")
