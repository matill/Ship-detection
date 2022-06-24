import torch
from torch import nn
from yolo_lib.cfg import SAFE_MODE, DEVICE


class SoftBinaryFocalLoss(nn.Module):
    def __init__(self, gamma: int, pos_loss_weight: float, neg_loss_weight: float):
        super().__init__()
        self.gamma = gamma
        self.pos_loss_weight = pos_loss_weight
        self.neg_loss_weight = neg_loss_weight

    def forward(
        self,
        logits: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> torch.Tensor:
        assert logits.shape == ground_truth.shape
        assert ground_truth.dtype in [torch.float32, torch.float64], f"Expected float, got {ground_truth.dtype}"
        assert logits.dtype in [torch.float32, torch.float64], f"Expected float, got {logits.dtype}"

        pos_exp = 1 + torch.exp(logits)
        neg_exp = 1 + torch.exp(-logits)
        pos_loss = torch.log(neg_exp) * (pos_exp ** -self.gamma) * self.pos_loss_weight
        neg_loss = torch.log(pos_exp) * (neg_exp ** -self.gamma) * self.neg_loss_weight

        pos_assignment = ground_truth
        neg_assignment = -ground_truth
        loss = pos_loss * pos_assignment + neg_loss * neg_assignment
        assert loss.shape == logits.shape
        return loss


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma, pos_loss_weight, neg_loss_weight):
        super().__init__()

        self.gamma = torch.tensor(gamma, device=DEVICE, requires_grad=False)
        self.pos_loss_weight = torch.tensor(pos_loss_weight, device=DEVICE, requires_grad=False)
        self.neg_loss_weight = torch.tensor(neg_loss_weight, device=DEVICE, requires_grad=False)

    def forward(
        self,
        logits: torch.Tensor,
        ground_truth: torch.Tensor,
        get_grid=False
    ) -> torch.Tensor:
        assert ground_truth.dtype == torch.bool, f"Exoected bool, got {ground_truth.dtype}"

        # TODO: Use _focal_loss_core
        if SAFE_MODE:
            assert logits.shape == ground_truth.shape

        logits_true = torch.where(ground_truth, logits, -logits)
        weights = torch.where(ground_truth, self.pos_loss_weight, self.neg_loss_weight)
        loss_pre_w = StableFocalLossAutograd.apply(logits_true, self.gamma)
        loss = loss_pre_w * weights
        if not get_grid:
            return loss.sum()
        else:
            return loss


class StableFocalLossAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits_true: torch.Tensor, gamma: int):
        # Cast to float64, to reduce the probability of numerical errors
        if logits_true.dtype != torch.float64:
            logits_true = logits_true.type(torch.float64)

        pos_exp = torch.exp(logits_true)
        pos_exp_1 = 1 + pos_exp
        neg_exp_1 = 1 + torch.exp(-logits_true)

        # ln(1 + exp(-x)): If exp(-x) is inf, the trivially computed ln is also inf, but
        # the real value is very close to -x.
        ln_neg_exp_1 = torch.log(neg_exp_1)
        isinf = ln_neg_exp_1.isinf()
        if isinf.any():
            print("ln_neg_exp_1.isinf()", isinf.sum())

        ln_neg_exp_1 = torch.where(isinf, -logits_true, ln_neg_exp_1)
        pos_exp_1_gamma = pos_exp_1 ** -gamma
        loss_pre_w = ln_neg_exp_1 * pos_exp_1_gamma
        ctx.save_for_backward(pos_exp_1, ln_neg_exp_1, pos_exp, gamma)

        if loss_pre_w.isnan().any():
            print("pos_exp_1_gamma", pos_exp_1_gamma)
            print("ln_neg_exp_1", ln_neg_exp_1)
        return loss_pre_w

    @staticmethod
    def backward(ctx, grad_loss_pre_w: torch.Tensor):
        (pos_exp_1, ln_neg_exp_1, pos_exp, gamma) = ctx.saved_tensors
        # If ln_times_pos_exp is nan, assume ln_times_pos_exp = 0. Why:
        # pos_exp = exp(x), ln_neg_exp_1 = ln(1 + exp(-x))
        # ln_times_pos_exp = ln(1 + exp(-x)) * exp(x)
        # We know ln(1 + exp(-x)) does not overflow, but it can overflow and become 0.
        # If exp(x) underflows, ln(1 + exp(-x)) * 0 = 0
        # If exp(x) overflows, ln(1 + exp(-x)) * inf = inf or nan
        # Note: ln_times_pos_exp cannot be inf, since pos_exp and ln_neg_exp_1 cannot be inf at the same time
        ln_times_pos_exp = ln_neg_exp_1 * pos_exp
        ln_times_pos_exp = torch.where(ln_neg_exp_1 < 0.0001, 1., ln_times_pos_exp)
        grad_logits_true = -(pos_exp_1 ** -(gamma + 1)) * (1 + gamma * ln_times_pos_exp) * grad_loss_pre_w
        assert grad_logits_true.shape == grad_loss_pre_w.shape

        if grad_logits_true.isnan().any():
            print("grad_logits_true", grad_logits_true)
            print("ln_times_pos_exp", ln_times_pos_exp)

        return grad_logits_true, None

