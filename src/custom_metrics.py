import torch

from config import patch_width, patch_height
from scale import unscale_x, unscale_y


class FoundBallMetric:
    def __init__(self, name='found_balls'):
        self.name = name
        self.reset_states()

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update the metric state with new predictions."""
        x_t = unscale_x(y_true[:, 0]) + patch_width / 2
        y_t = unscale_y(y_true[:, 1]) + patch_height / 2

        x_p = unscale_x(y_pred[:, 0]) + patch_width / 2
        y_p = unscale_y(y_pred[:, 1]) + patch_height / 2

        d = torch.sqrt((x_t - x_p)**2 + (y_t - y_p)**2)
        r = d < y_true[:, 2]

        self.found_balls += torch.sum(r.float())
        self.totals_balls += float(len(y_true))

    def result(self):
        """Compute the current metric result."""
        if self.totals_balls == 0:
            return 0.0
        return self.found_balls / self.totals_balls

    def reset_states(self):
        """Reset the metric state."""
        self.found_balls = 0.0
        self.totals_balls = 0.0
