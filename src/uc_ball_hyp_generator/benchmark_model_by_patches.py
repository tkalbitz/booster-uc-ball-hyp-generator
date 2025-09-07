import torch

import uc_ball_hyp_generator.models as models
from uc_ball_hyp_generator.config import image_dir, patch_height, patch_width, testset_csv_collection
from uc_ball_hyp_generator.csv_label_reader import load_csv_collection
from uc_ball_hyp_generator.dataset_handling import create_dataset
from uc_ball_hyp_generator.scale import unscale_x, unscale_y

if __name__ != "__main__":
    exit(1)

# Set device
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

png_files: dict[str, str] = {f.name: str(f) for f in image_dir.glob("**/*.png")}

test_img, test_labels, skipped_testset = load_csv_collection(testset_csv_collection, png_files)

print(f"Testset contains {len(test_img)} and we removed {skipped_testset} balls.")

batch_size: int = 200
steps: int = len(test_img) // batch_size
ds = create_dataset(test_img, test_labels, batch_size, trainset=False)

model_file = (
    "/home/tkalbitz/PycharmProjects/uc-ball-hyp-generator/model/yuv_2021-05-21-17-24-53/weights.acc.268-0.964466.h5"
)

model = models.create_network_v2(patch_height, patch_width)
model.load_state_dict(torch.load(model_file, map_location=device))
model.to(device)
model.eval()

ball_sum = 0
it = iter(ds)

for s in range(steps):
    image_batch, label_batch = next(it)

    # Convert to tensors and move to device
    image_batch = image_batch.to(device)
    label_batch = label_batch.to(device)

    with torch.no_grad():
        pred = model(image_batch)

    x_t = torch.tensor(unscale_x(label_batch[:, 0])) + patch_width / 2
    y_t = torch.tensor(unscale_y(label_batch[:, 1])) + patch_height / 2

    x_p = torch.tensor(unscale_x(pred[:, 0])) + patch_width / 2
    y_p = torch.tensor(unscale_y(pred[:, 1])) + patch_height / 2

    d = torch.sqrt((x_t - x_p) ** 2 + (y_t - y_p) ** 2)
    r = d < label_batch[:, 2]
    cur_sum = torch.sum(r.int())
    ball_sum += int(cur_sum.item())


print(f"Found in total {ball_sum} / {steps * batch_size}")
