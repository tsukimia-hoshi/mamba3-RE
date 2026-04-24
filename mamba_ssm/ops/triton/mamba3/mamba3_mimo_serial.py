"""Serial single-token MIMO Mamba-3 implementation with Triton rotary kernel.

This path intentionally supports only serial recurrence (token-by-token), which
matches the "one chunk == one token" requirement.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from mamba_ssm.ops.triton.mamba3.mamba3_mimo_rotary_step import (
    apply_rotary_qk_inference_fwd,
    apply_rotary_qk_inference_reference,
)


def _rotate_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    angle_state: torch.Tensor,
    angle_proj: torch.Tensor,
    dt: torch.Tensor,
    q_bias: torch.Tensor,
    k_bias: torch.Tensor,
    use_triton: bool,
    rotate_pairwise: bool,
):
    if use_triton:
        return apply_rotary_qk_inference_fwd(
            q=q,
            k=k,
            angle_state=angle_state,
            angle_proj=angle_proj,
            dt=dt,
            bias_q=q_bias,
            bias_k=k_bias,
            inplace=False,
            conjugate=False,
            rotate_pairwise=rotate_pairwise,
        )
    return apply_rotary_qk_inference_reference(
        q=q,
        k=k,
        angle_state=angle_state,
        angle_proj=angle_proj,
        dt=dt,
        bias_q=q_bias,
        bias_k=k_bias,
        conjugate=False,
        rotate_pairwise=rotate_pairwise,
    )


def mamba3_mimo_serial_token(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    adt: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
    angles: torch.Tensor,
    q_bias: torch.Tensor,
    k_bias: torch.Tensor,
    mimo_v: torch.Tensor,
    mimo_o: torch.Tensor,
    d_skip: torch.Tensor,
    z: Optional[torch.Tensor],
    angle_state: torch.Tensor,
    ssm_state: torch.Tensor,
    k_state: torch.Tensor,
    v_state: torch.Tensor,
    rotate_pairwise: bool = False,
):
    """Single-token MIMO recurrence.

    Shapes:
      q,k: (B, R, H, S)
      v,z: (B, H, P)
      adt,dt,trap: (B, H)
      angles: (B, H, S/2)
      q_bias,k_bias,mimo_v,mimo_o: (R, H, S/P)
      angle_state: (B, H, S/2)
      ssm_state: (B, R, H, P, S)
      k_state: (B, R, H, S)
      v_state: (B, H, P)
    """
    need_grad = any(
        t.requires_grad
        for t in (q, k, v, adt, dt, trap, angles, q_bias, k_bias, mimo_v, mimo_o, d_skip)
        if t is not None
    )
    use_triton_rotary = (not need_grad) and q.is_cuda
    q_rot, k_rot, next_angle = _rotate_qk(
        q=q,
        k=k,
        angle_state=angle_state,
        angle_proj=angles,
        dt=dt,
        q_bias=q_bias,
        k_bias=k_bias,
        use_triton=use_triton_rotary,
        rotate_pairwise=rotate_pairwise,
    )

    # Up-project V to R branches: (B,R,H,P)
    v_up = torch.einsum("bhp,rhp->brhp", v, mimo_v)
    alpha = torch.exp(adt).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    beta = (alpha.squeeze(-1).squeeze(-1) * dt.unsqueeze(1) * (1.0 - torch.sigmoid(trap).unsqueeze(1))).unsqueeze(-1)
    gamma = (dt.unsqueeze(1) * torch.sigmoid(trap).unsqueeze(1)).unsqueeze(-1)

    ssm_delta = beta.unsqueeze(-1) * v_state.unsqueeze(1).unsqueeze(-1) * k_state.unsqueeze(-2)
    ssm_delta = ssm_delta + gamma.unsqueeze(-1) * v_up.unsqueeze(-1) * k_rot.unsqueeze(-2)
    next_ssm = alpha * ssm_state + ssm_delta

    y_branch = torch.einsum("brhps,brhs->brhp", next_ssm.to(v.dtype), q_rot.to(v.dtype))
    y_branch = y_branch + d_skip.view(1, 1, -1, 1) * v_up
    y = torch.einsum("brhp,rhp->bhp", y_branch, mimo_o)
    if z is not None:
        y = y * F.silu(z)

    return y, next_angle, next_ssm, k_rot, v


def mamba3_mimo_serial(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    adt: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
    angles: torch.Tensor,
    q_bias: torch.Tensor,
    k_bias: torch.Tensor,
    mimo_v: torch.Tensor,
    mimo_o: torch.Tensor,
    d_skip: torch.Tensor,
    z: Optional[torch.Tensor] = None,
    input_states: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
):
    """Serial MIMO for training/inference. Supports only token-by-token recurrence."""
    b, l, r, h, s = q.shape
    p = v.shape[-1]
    assert l >= 1
    if input_states is None:
        angle_state = torch.zeros((b, h, angles.shape[-1]), device=q.device, dtype=torch.float32)
        ssm_state = torch.zeros((b, r, h, p, s), device=q.device, dtype=torch.float32)
        k_state = torch.zeros((b, r, h, s), device=q.device, dtype=q.dtype)
        v_state = torch.zeros((b, h, p), device=q.device, dtype=v.dtype)
    else:
        angle_state, ssm_state, k_state, v_state = input_states

    outs = []
    for t in range(l):
        y, angle_state, ssm_state, k_state, v_state = mamba3_mimo_serial_token(
            q=q[:, t],
            k=k[:, t],
            v=v[:, t],
            adt=adt[:, :, t],
            dt=dt[:, :, t],
            trap=trap[:, :, t],
            angles=angles[:, t],
            q_bias=q_bias,
            k_bias=k_bias,
            mimo_v=mimo_v,
            mimo_o=mimo_o,
            d_skip=d_skip,
            z=None if z is None else z[:, t],
            angle_state=angle_state,
            ssm_state=ssm_state,
            k_state=k_state,
            v_state=v_state,
        )
        outs.append(y)

    out = torch.stack(outs, dim=1)
    return out, (angle_state, ssm_state, k_state, v_state)
