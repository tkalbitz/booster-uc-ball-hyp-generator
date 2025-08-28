import torch
import os
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

from scale import unscale_x, unscale_y

if __name__ != '__main__':
    exit(1)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

import models
from config import patch_height, patch_width, image_dir, testset_csv_collection
from csv_label_reader import load_csv_collection
from dataset_handling import create_dataset

png_files = {f.name: str(f) for f in image_dir.glob("**/*.png")}

test_img, test_labels, skipped_testset = load_csv_collection(testset_csv_collection, png_files)

print(f"Testset contains {len(test_img)} and we removed {skipped_testset} balls.")

batch_size = 200
steps = len(test_img) // batch_size
ds = create_dataset(test_img, test_labels, batch_size, trainset=False)

# RGB 3804 model_file = "/home/tkalbitz/PycharmProjects/uc-ball-hyp-generator/model/2021-05-17-17-57-51/weights.loss.152-1.657603.h5"
# RGB 3806 model_file = "/home/tkalbitz/PycharmProjects/uc-ball-hyp-generator/model/2021-05-17-17-57-51/weights.acc.149-0.958740.h5"
# YUV 3824 model_file = "/home/tkalbitz/PycharmProjects/uc-ball-hyp-generator/model/yuv_2021-05-17-21-08-39/weights.loss.192-1.614760.h5"
# YUV 3819 model_file = "/home/tkalbitz/PycharmProjects/uc-ball-hyp-generator/model/yuv_2021-05-17-21-08-39/weights.loss.228-1.578639.h5"
# YUV 3821 model_file = "/home/tkalbitz/PycharmProjects/uc-ball-hyp-generator/model/yuv_2021-05-17-21-08-39/weights.acc.227-0.959717.h5"
# YUV 3801 model_file = "/home/tkalbitz/PycharmProjects/uc-ball-hyp-generator/model/yuv_2021-05-17-20-52-43/weights.loss.125-1.722393.h5"
# YUV 3830 model_file = "/home/tkalbitz/PycharmProjects/uc-ball-hyp-generator/model/yuv_2021-05-17-20-31-42/weights.acc.208-0.957764.h5"
model_file = "/home/tkalbitz/PycharmProjects/uc-ball-hyp-generator/model/yuv_2021-05-21-17-24-53/weights.acc.268-0.964466.h5"

model = models.create_network_v2(patch_height, patch_width)
model.load_weights(model_file)

ball_sum = 0
it = iter(ds)

for s in range(steps):
    image_batch, label_batch = next(it)

    pred = model.predict_on_batch(image_batch)

    x_t = unscale_x(label_batch[:,0]) + patch_width / 2
    y_t = unscale_y(label_batch[:,1]) + patch_height / 2

    x_p = unscale_x(pred[:,0]) + patch_width / 2
    y_p = unscale_y(pred[:,1]) + patch_height / 2

    d = torch.sqrt((x_t - x_p)**2 + (y_t - y_p)**2)
    r = d < label_batch[:,2]
    cur_sum = torch.sum(r.int())
    ball_sum += cur_sum

    #print(f"{cur_sum} / {batch_size}")

print(f"Found in total {ball_sum} / {steps * batch_size}")
