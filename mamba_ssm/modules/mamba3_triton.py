# Copyright (c) 2026, Dao AI Lab, Goombalab.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from mamba_ssm.ops.triton.mamba3.mamba3_mimo_serial import mamba3_mimo_serial


class Mamba3Triton(nn.Module):
    """Pure Triton-oriented Mamba-3 variant.

    This implementation focuses on serial token-by-token MIMO recurrence,
    for both training forward/backward and inference step.
    """

    def __init__(
        self,
        d_model,
        d_state=128,
        expand=2,
        headdim=64,
        ngroups=1,
        rope_fraction=0.5,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        A_floor=1e-4,
        mimo_rank=4,
        is_outproj_norm=False,
        layer_idx=None,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.headdim = headdim
        self.mimo_rank = mimo_rank
        self.layer_idx = layer_idx
        self.A_floor = A_floor
        self.is_outproj_norm = is_outproj_norm

        self.d_inner = int(expand * d_model)
        self.nheads = self.d_inner // headdim
        self.num_bc_heads = ngroups

        assert rope_fraction in [0.5, 1.0]
        self.split_tensor_size = int(d_state * rope_fraction)
        if self.split_tensor_size % 2 != 0:
            self.split_tensor_size -= 1
        self.num_rope_angles = self.split_tensor_size // 2

        d_in_proj = 2 * self.d_inner + 2 * self.d_state * self.num_bc_heads * self.mimo_rank + 3 * self.nheads + self.num_rope_angles
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=False, **factory_kwargs)

        _dt = torch.exp(
            torch.rand(self.nheads, device=device, dtype=torch.float32) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        _dt = torch.clamp(_dt, min=dt_init_floor)
        _dt_bias = _dt + torch.log(-torch.expm1(-_dt))
        self.dt_bias = nn.Parameter(_dt_bias, requires_grad=True)

        self.B_bias = nn.Parameter(torch.ones((self.nheads, self.mimo_rank, self.d_state), dtype=torch.float32, device=device))
        self.C_bias = nn.Parameter(torch.ones((self.nheads, self.mimo_rank, self.d_state), dtype=torch.float32, device=device))
        self.B_norm = RMSNormGated(self.d_state, eps=1e-5, **factory_kwargs)
        self.C_norm = RMSNormGated(self.d_state, eps=1e-5, **factory_kwargs)

        self.mimo_x = nn.Parameter(torch.ones(self.nheads, self.mimo_rank, self.headdim, device=device) / self.mimo_rank)
        self.mimo_z = nn.Parameter(torch.ones(self.nheads, self.mimo_rank, self.headdim, device=device))
        self.mimo_o = nn.Parameter(torch.ones(self.nheads, self.mimo_rank, self.headdim, device=device) / self.mimo_rank)

        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        if is_outproj_norm:
            self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=True, group_size=self.headdim, **factory_kwargs)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False, **factory_kwargs)

    def _project_inputs(self, u):
        zxBCdtAtrap = self.in_proj(u)
        z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
            zxBCdtAtrap,
            [
                self.d_inner,
                self.d_inner,
                self.d_state * self.num_bc_heads * self.mimo_rank,
                self.d_state * self.num_bc_heads * self.mimo_rank,
                self.nheads,
                self.nheads,
                self.nheads,
                self.num_rope_angles,
            ],
            dim=-1,
        )
        z = rearrange(z, "b l (h p) -> b l h p", p=self.headdim)
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        B = rearrange(B, "b l (r g n) -> b l r g n", r=self.mimo_rank, g=self.num_bc_heads)
        C = rearrange(C, "b l (r g n) -> b l r g n", r=self.mimo_rank, g=self.num_bc_heads)
        trap = rearrange(trap, "b l h -> b h l")

        _A = -F.softplus(dd_A.float())
        _A = torch.clamp(_A, max=-self.A_floor)
        DT = F.softplus(dd_dt + self.dt_bias)
        ADT = rearrange(_A * DT, "b l h -> b h l")
        DT = rearrange(DT, "b l h -> b h l")

        angles = angles.unsqueeze(-2).expand(-1, -1, self.nheads, -1).float()
        B = self.B_norm(B)
        C = self.C_norm(C)
        return z, x, B, C, ADT, DT, trap, angles

    def forward(self, u, inference_params=None, cu_seqlens=None, seq_idx=None):
        del cu_seqlens, seq_idx
        z, x, B, C, ADT, DT, trap, angles = self._project_inputs(u)

        angle_state = ssm_state = k_state = v_state = None
        if inference_params is not None:
            angle_state, ssm_state, k_state, v_state = self._get_states_from_cache(inference_params, u.shape[0])

        y, states = mamba3_mimo_serial(
            q=C,
            k=B,
            v=x,
            adt=ADT,
            dt=DT,
            trap=trap,
            angles=angles,
            q_bias=rearrange(self.C_bias, "h r s -> r h s"),
            k_bias=rearrange(self.B_bias, "h r s -> r h s"),
            mimo_v=rearrange(self.mimo_x, "h r p -> r h p"),
            mimo_o=rearrange(self.mimo_o, "h r p -> r h p"),
            d_skip=self.D,
            z=None if self.is_outproj_norm else z,
            input_states=(angle_state, ssm_state, k_state, v_state) if angle_state is not None else None,
        )

        if self.is_outproj_norm:
            z_flat = rearrange(z, "b l h p -> b l (h p)")
            y_flat = rearrange(y, "b l h p -> b l (h p)")
            y = rearrange(self.norm(y_flat, z_flat), "b l (h p) -> b l h p", p=self.headdim)

        if inference_params is not None:
            angle_state.copy_(states[0])
            ssm_state.copy_(states[1])
            k_state.copy_(states[2])
            v_state.copy_(states[3])

        out = self.out_proj(rearrange(y, "b l h p -> b l (h p)").to(u.dtype))
        return out

    def allocate_inference_cache(self, batch_size, max_seqlen, device=None, dtype=None, **kwargs):
        del max_seqlen, kwargs
        device = self.in_proj.weight.device if device is None else device
        dtype = self.in_proj.weight.dtype if dtype is None else dtype
        angle_dt_state = torch.zeros((batch_size, self.nheads, self.num_rope_angles), device=device, dtype=torch.float32)
        ssm_state = torch.zeros((batch_size, self.mimo_rank, self.nheads, self.headdim, self.d_state), device=device, dtype=torch.float32)
        k_state = torch.zeros((batch_size, self.mimo_rank, self.nheads, self.d_state), device=device, dtype=dtype)
        v_state = torch.zeros((batch_size, self.nheads, self.headdim), device=device, dtype=dtype)
        return angle_dt_state, ssm_state, k_state, v_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            inference_params.key_value_memory_dict[self.layer_idx] = self.allocate_inference_cache(batch_size, 1)
        states = inference_params.key_value_memory_dict[self.layer_idx]
        if initialize_states:
            for s in states:
                s.zero_()
        return states
