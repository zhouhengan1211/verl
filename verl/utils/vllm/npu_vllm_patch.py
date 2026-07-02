# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from functools import wraps

from verl.utils.device import is_torch_npu_available


def vllm_ascend_v011_select_moe_comm_method_wrapper(fn):
    @wraps(fn)
    def wrapper(self, num_tokens, with_prefill):
        moe_comm_method = fn(self, num_tokens, with_prefill)
        from vllm_ascend.ascend_forward_context import MoECommType
        from vllm_ascend.utils import AscendSocVersion, enable_sp, get_ascend_soc_version

        soc_version = get_ascend_soc_version()

        # AscendSocVersion.A2 is not support MC2 in Single-card multi-process scenario now.
        if soc_version in {AscendSocVersion.A2} and moe_comm_method == MoECommType.MC2:
            quant_type = getattr(self.vllm_config.model_config.hf_config, "moe_quantize", None)
            # Currently, w4a8_dynamic does not support allgatherep
            if quant_type == "w4a8_dynamic":
                moe_comm_method = MoECommType.ALLTOALL
            else:
                moe_comm_method = MoECommType.ALLGATHER

        if with_prefill:
            if enable_sp():
                moe_comm_method = MoECommType.ALLGATHER
            else:
                moe_comm_method = MoECommType.NAIVE_MULTICAST

        return moe_comm_method

    return wrapper


def vllm_ascend_v011_matmul_and_reduce_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        from vllm_ascend.utils import AscendSocVersion, get_ascend_soc_version

        soc_version = get_ascend_soc_version()
        # AscendSocVersion.A2 is not support MC2 in Single-card multi-process scenario now.
        if soc_version in {AscendSocVersion.A2}:
            from vllm.forward_context import get_forward_context

            try:
                forward_context = get_forward_context()
                forward_context.mmrs_fusion = False
            except AssertionError:
                # forward_context.mmrs_fusion will be false in matmul_and_reduce func.
                pass
        return fn(self, *args, **kwargs)

    return wrapper


def check_vllm_ascend_before_server_launch():
    import torch_npu
    import vllm

    def _is_ascend_soc_version_A2_v011_local():
        from vllm_ascend.utils import AscendSocVersion

        soc_version = torch_npu.npu.get_soc_version()
        if 220 <= soc_version <= 225:
            _ascend_soc_version = AscendSocVersion.A2
        elif 250 <= soc_version <= 255:
            _ascend_soc_version = AscendSocVersion.A3
        else:
            _ascend_soc_version = AscendSocVersion.UNDEFINED

        return _ascend_soc_version == AscendSocVersion.A2

    def _is_ascend_soc_version_A2_v013_local():
        from vllm_ascend.utils import AscendDeviceType

        soc_version = torch_npu.npu.get_soc_version()
        if 220 <= soc_version <= 225:
            cur_device_type = AscendDeviceType.A2
        elif 250 <= soc_version <= 255:
            cur_device_type = AscendDeviceType.A3
        elif 200 <= soc_version <= 205:
            cur_device_type = AscendDeviceType._310P
        elif soc_version == 260:
            cur_device_type = AscendDeviceType.A5
        else:
            raise RuntimeError(f"Can not support soc_version: {soc_version}.")

        return cur_device_type == AscendDeviceType.A2

    if vllm.__version__ == "0.11.0":
        is_A2 = _is_ascend_soc_version_A2_v011_local()
    elif vllm.__version__ == "0.13.0":
        is_A2 = _is_ascend_soc_version_A2_v013_local()
    else:
        is_A2 = False

    if is_A2:
        VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE = bool(int(os.getenv("VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE", "0")))
        if VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE:
            raise AssertionError(
                "AscendSocVersion.A2 is not support VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE \
                in Single-card multi-process scenario now. "
            )


def vllm_ascend_v013_select_moe_comm_method_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        moe_comm_method = fn(*args, **kwargs)
        from vllm_ascend.ascend_forward_context import MoECommType
        from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

        ascend_device_type = get_ascend_device_type()

        # AscendSocVersion.A2 is not support MC2 in Single-card multi-process scenario now.
        if ascend_device_type in {AscendDeviceType.A2} and moe_comm_method == MoECommType.MC2:
            moe_comm_method = MoECommType.ALLGATHER

        return moe_comm_method

    return wrapper


def vllm_ascend_v013_matmul_and_reduce_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

        ascend_device_type = get_ascend_device_type()
        # AscendSocVersion.A2 is not support MC2 in Single-card multi-process scenario now.
        if ascend_device_type in {AscendDeviceType.A2}:
            from vllm.forward_context import get_forward_context

            try:
                forward_context = get_forward_context()
                forward_context.mmrs_fusion = False
            except AssertionError:
                # forward_context.mmrs_fusion will be false in matmul_and_reduce func.
                pass
        return fn(self, *args, **kwargs)

    return wrapper


def vllm_v013_weight_loader_method_wrapper(fn):
    @wraps(fn)
    def wrapper(self, param, loaded_weight, weight_name, shard_id, expert_id, return_success=False):
        if (shard_id in ("w1", "w3") and param.shape[1] == self.hidden_size) or (
            shard_id == "w2" and param.shape[2] == self.hidden_size
        ):
            param.data = param.data.transpose(1, 2)
        return fn(self, param, loaded_weight, weight_name, shard_id, expert_id, return_success)

    return wrapper


def patch_vllm013_rotary_emb():
    from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

    def vllm013_npu_rotary_embedding_init_impl(
        self,
        enforce_enable: bool = False,
        is_neox_style: bool = True,
        enable_fp32_compute: bool = False,
    ) -> None:
        super(ApplyRotaryEmb, self).__init__()
        self.is_neox_style = is_neox_style
        self.enable_fp32_compute = enable_fp32_compute
        self.apply_rotary_emb_flash_attn = None

    ApplyRotaryEmb.__init__ = vllm013_npu_rotary_embedding_init_impl


def _patch_qwen3_vl_moe_load_weights_for_async_sync():
    """Fix Qwen3MoeLLMModel.load_weights for verl async weight sync.

    Bug: The original code in qwen3_vl_moe.py:load_weights (line 247) does
         loaded_weight.transpose(-1, -2)  # no bias
    on all fused expert weights. For 3D weights from Megatron async sync
    (gate_up_proj shape [E, 2I, H], down_proj shape [E, H, I]), this transpose
    swaps hidden and intermediate dimensions. Then:

      gate_up_proj [E, 2I, H]:
        after transpose → [E, H, 2I]
        chunk(2, dim=-2) → splits dim=1 (H) instead of dim=2 (2I)
        result: [E, H/2, 2I] = [32, 1024, 1536] per-expert — WRONG

      down_proj [E, H, I]:
        after transpose → [E, I, H]
        passed to _load_w2 which expects [E, H, I] — WRONG layout

    Fix 1: Wrap load_weights to pre-transpose 3D expert weights, so the
    original transpose(-1, -2) cancels out as a no-op.

    Fix 2: Wrap load_fused_expert_weights to iterate loaded_weight.shape[0]
    instead of the config's num_experts. Megatron with EP=4 sends only
    num_experts_per_rank=32 entries, but the model config has num_experts=128.
    """
    import typing
    from functools import wraps

    from vllm.model_executor.models.qwen3_vl_moe import Qwen3MoeLLMModel

    # ---- Fix 1: load_weights - cancel out incorrect transpose ----
    original_load_weights = Qwen3MoeLLMModel.load_weights

    @wraps(original_load_weights)
    def patched_load_weights(self, weights):
        processed_weights = []
        for name, tensor in weights:
            if tensor.dim() == 3 and (
                "experts.gate_up_proj" in name or "experts.down_proj" in name
            ):
                # Pre-transpose: the original method will call transpose(-1, -2)
                # again, reverting to the correct layout. This makes the
                # original transpose a no-op for 3D Megatron-originated weights.
                tensor = tensor.transpose(-1, -2)
            processed_weights.append((name, tensor))
        return original_load_weights(self, processed_weights)

    Qwen3MoeLLMModel.load_weights = patched_load_weights

    # ---- Fix 2: load_fused_expert_weights - use actual expert count ----
    original_load_fused = Qwen3MoeLLMModel.load_fused_expert_weights

    @wraps(original_load_fused)
    def patched_load_fused_expert_weights(
        self, name, params_dict, loaded_weight, shard_id, num_experts
    ):
        param = params_dict[name]
        weight_loader = typing.cast(typing.Callable[..., bool], param.weight_loader)
        loaded_local_expert = False
        # Use actual expert count from tensor dim 0, not config num_experts.
        # When Megatron uses EP=4, each rank sends only 32 experts instead of
        # the full 128, avoiding IndexError on loaded_weight[32].
        for expert_id in range(loaded_weight.shape[0]):
            curr_expert_weight = loaded_weight[expert_id]
            success = weight_loader(
                param,
                curr_expert_weight,
                name,
                shard_id,
                expert_id,
                return_success=True,
            )
            if success:
                loaded_local_expert = True
        return loaded_local_expert

    Qwen3MoeLLMModel.load_fused_expert_weights = patched_load_fused_expert_weights


if is_torch_npu_available(check_device=False):
    import vllm
    from packaging import version

    _VLLM_VERSION = version.parse(vllm.__version__)
    if _VLLM_VERSION >= version.parse("0.13.0") and _VLLM_VERSION <= version.parse("0.14.0"):
        # Disable flash_attn in RotaryEmbedding (NPU) when VLLM >= 0.13
        from vllm.model_executor.layers.fused_moe import FusedMoE

        patch_vllm013_rotary_emb()
        FusedMoE.weight_loader = vllm_v013_weight_loader_method_wrapper(FusedMoE.weight_loader)
    elif _VLLM_VERSION >= version.parse("0.18.0") and _VLLM_VERSION < version.parse("0.19.0"):
        # vLLM 0.18.x: Rotary + FusedMoE weight_loader + Qwen3-VL-MoE async sync fix
        from vllm.model_executor.layers.fused_moe import FusedMoE

        patch_vllm013_rotary_emb()
        FusedMoE.weight_loader = vllm_v013_weight_loader_method_wrapper(FusedMoE.weight_loader)
        _patch_qwen3_vl_moe_load_weights_for_async_sync()
    elif _VLLM_VERSION >= version.parse("0.19.0"):
        # Disable flash_attn in RotaryEmbedding (NPU) when VLLM >= 0.19
        from vllm.model_executor.layers.fused_moe import FusedMoE

        patch_vllm013_rotary_emb()
        FusedMoE.weight_loader = vllm_v013_weight_loader_method_wrapper(FusedMoE.weight_loader)
        _patch_qwen3_vl_moe_load_weights_for_async_sync()

    VERL_NPU_ENABLE_A2_PATCH_VLLM_ASCEND_MC2 = bool(int(os.getenv("VERL_NPU_ENABLE_A2_PATCH_VLLM_ASCEND_MC2", "1")))
    if VERL_NPU_ENABLE_A2_PATCH_VLLM_ASCEND_MC2:
        # only support vllm 0.13 and 0.11 now.
        if _VLLM_VERSION >= version.parse("0.13.0") and _VLLM_VERSION <= version.parse("0.14.0"):
            from vllm_ascend import ascend_forward_context
            from vllm_ascend.ops.linear_op import SequenceRowParallelOp

            ascend_forward_context.select_moe_comm_method = vllm_ascend_v013_select_moe_comm_method_wrapper(
                ascend_forward_context.select_moe_comm_method
            )
            SequenceRowParallelOp.matmul_and_reduce = vllm_ascend_v013_matmul_and_reduce_wrapper(
                SequenceRowParallelOp.matmul_and_reduce
            )

        elif _VLLM_VERSION >= version.parse("0.11.0") and _VLLM_VERSION < version.parse("0.13.0"):
            from vllm_ascend.ops.linear_op import SequenceRowParallelOp
            from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

            NPUModelRunner._select_moe_comm_method = vllm_ascend_v011_select_moe_comm_method_wrapper(
                NPUModelRunner._select_moe_comm_method
            )
            SequenceRowParallelOp.matmul_and_reduce = vllm_ascend_v011_matmul_and_reduce_wrapper(
                SequenceRowParallelOp.matmul_and_reduce
            )
