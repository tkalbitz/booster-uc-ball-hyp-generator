import torch

from uc_ball_hyp_generator.hyp_generator.config import patch_height, patch_width, scale_factor_f
from uc_ball_hyp_generator.hyp_generator.scale_patch import unscale_patch_x, unscale_patch_y, unscale_radius


class PatchFoundBallMetric:
    def __init__(self, name: str = "found_balls") -> None:
        self.name = name
        self.reset_states()

    def update_state(
        self, y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: torch.Tensor | None = None
    ) -> None:
        """Update the metric state with new predictions."""
        x_t = unscale_patch_x(torch.tanh(y_true[:, 0])) + patch_width / 2
        y_t = unscale_patch_y(torch.tanh(y_true[:, 1])) + patch_height / 2

        x_p = unscale_patch_x(torch.tanh(y_pred[:, 0])) + patch_width / 2
        y_p = unscale_patch_y(torch.tanh(y_pred[:, 1])) + patch_height / 2

        # Ensure tensors are used for torch.sqrt
        x_diff = torch.as_tensor(x_t - x_p)
        y_diff = torch.as_tensor(y_t - y_p)
        d = torch.sqrt(x_diff * x_diff + y_diff * y_diff)
        r = d < y_true[:, 3]

        self.found_balls += torch.sum(r.float()).item()
        self.totals_balls += float(len(y_true))

    def result(self) -> float:
        """Compute the current metric result."""
        if self.totals_balls == 0:
            return 0.0
        return self.found_balls / self.totals_balls

    def reset_states(self) -> None:
        """Reset the metric state."""
        self.found_balls: float = 0.0
        self.totals_balls: float = 0.0


class PatchBallRadiusMetric:
    def __init__(self, name: str = "found_balls") -> None:
        self.name = name
        self.reset_states()

    def update_state(
        self, y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: torch.Tensor | None = None
    ) -> None:
        """Update the metric state with new predictions."""
        t_r = unscale_radius(torch.tanh(y_true[:, 2])) * scale_factor_f * 2
        p_r = unscale_radius(torch.tanh(y_pred[:, 2])) * scale_factor_f * 2

        # Ensure tensors are used for torch.sqrt
        # r_diff = torch.abs(t_r - p_r)
        max_delta = 0.2
        lower_bound = t_r * (1 - max_delta)
        upper_bound = t_r * (1 + max_delta)

        r = (lower_bound <= p_r) & (p_r <= upper_bound)

        self.found_balls += torch.sum(r.float()).item()
        self.totals_balls += float(len(y_true))

    def result(self) -> float:
        """Compute the current metric result."""
        if self.totals_balls == 0:
            return 0.0
        return self.found_balls / self.totals_balls

    def reset_states(self) -> None:
        """Reset the metric state."""
        self.found_balls: float = 0.0
        self.totals_balls: float = 0.0
