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


def _patch_fused_moe_load_w13_for_npu():
    """Patch FusedMoE._load_w13 and _load_w2 to handle weights that are already
    TP-sharded during verl IPC weight sync on NPU.

    During verl colocate IPC weight sync, fused MoE weights (w13, w2) may arrive
    already sharded per TP rank. The original _load_w13/_load_w2 unconditionally
    applies a TP narrow on the loaded_weight, which causes an IndexError when the
    loaded weight's shard_dim is already at the per-TP size (start offset exceeds
    dimension).

    This patch adds a guard: the TP narrow is only applied when the loaded weight
    has MORE elements along shard_dim than the target shard_size, indicating it is
    the full unsharded weight. If the weight is already at the correct shard size,
    the narrow is skipped.
    """
    import torch

    from vllm.model_executor.layers.fused_moe import FusedMoE

    _original_load_w13 = FusedMoE._load_w13

    @wraps(_original_load_w13)
    def _patched_load_w13(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        load_full: bool = False,
    ):
        if self.moe_config.is_act_and_mul:
            shard_size = expert_data.shape[shard_dim] // 2
        else:
            shard_size = expert_data.shape[shard_dim]

        # Only apply TP narrow if loaded_weight is larger than per-TP shard size.
        # When the weight is already TP-sharded along shard_dim (e.g. from IPC
        # colocate sync), loaded_weight.shape[shard_dim] == shard_size, so the
        # narrow on shard_dim is skipped. However, the OTHER dimension may still
        # carry the full (unsharded) intermediate_size and need TP narrowing.
        if not load_full and loaded_weight.ndim > 0:
            if loaded_weight.shape[shard_dim] > shard_size:
                loaded_weight = loaded_weight.narrow(
                    shard_dim, shard_size * tp_rank, shard_size
                )
            else:
                # shard_dim is already at per-TP size; check if the other
                # dimension still needs TP sharding (common for IPC weights
                # that are TP-partitioned along a different axis).
                other_dim = 1 - shard_dim
                if loaded_weight.shape[other_dim] > expert_data.shape[other_dim]:
                    tp_shard_size = expert_data.shape[other_dim]
                    loaded_weight = loaded_weight.narrow(
                        other_dim, tp_shard_size * tp_rank, tp_shard_size
                    )

        # Narrow parameter and load.
        if shard_id == "w1":
            expert_data = expert_data.narrow(shard_dim, 0, shard_size)
        else:
            assert shard_id == "w3"
            expert_data = expert_data.narrow(shard_dim, shard_size, shard_size)

        # On NPU, torch.narrow / torch.t return views whose storage shapes
        # may not be correctly recognized by aclnnInplaceCopy. Try to match
        # shapes and materialize before copying.
        if expert_data.shape != loaded_weight.shape:
            # The intermediate and hidden dimensions may be swapped relative
            # to the FusedMoE parameter layout — try transposing.
            if tuple(expert_data.shape) == tuple(loaded_weight.t().shape):
                loaded_weight = loaded_weight.t().contiguous()
        if expert_data.shape != loaded_weight.shape:
            raise RuntimeError(
                f"Shape mismatch in _load_w13: "
                f"expert_data={expert_data.shape}, "
                f"loaded_weight={loaded_weight.shape}, "
                f"shard_id={shard_id}, tp_rank={tp_rank}"
            )
        expert_data.copy_(loaded_weight)

    FusedMoE._load_w13 = _patched_load_w13

    _original_load_w2 = FusedMoE._load_w2

    @wraps(_original_load_w2)
    def _patched_load_w2(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        load_full: bool = False,
    ):
        shard_size = expert_data.shape[shard_dim]
        # Same guard as _load_w13: only narrow when the weight is unsharded
        # along shard_dim, otherwise try TP-narrow along the other dimension.
        if not load_full and loaded_weight.ndim > 0:
            if loaded_weight.shape[shard_dim] > shard_size:
                loaded_weight = loaded_weight.narrow(
                    shard_dim, shard_size * tp_rank, shard_size
                )
            else:
                other_dim = 1 - shard_dim
                if loaded_weight.shape[other_dim] > expert_data.shape[other_dim]:
                    tp_shard_size = expert_data.shape[other_dim]
                    loaded_weight = loaded_weight.narrow(
                        other_dim, tp_shard_size * tp_rank, tp_shard_size
                    )
        # On NPU, torch.narrow / torch.t return views whose storage shapes
        # may not be correctly recognized by aclnnInplaceCopy.
        if expert_data.shape != loaded_weight.shape:
            if tuple(expert_data.shape) == tuple(loaded_weight.t().shape):
                loaded_weight = loaded_weight.t().contiguous()
        if expert_data.shape != loaded_weight.shape:
            raise RuntimeError(
                f"Shape mismatch in _load_w2: "
                f"expert_data={expert_data.shape}, "
                f"loaded_weight={loaded_weight.shape}, "
                f"shard_id={shard_id}, tp_rank={tp_rank}"
            )
        expert_data.copy_(loaded_weight)

    FusedMoE._load_w2 = _patched_load_w2


def _patch_vllm_ascend_process_weights_for_npu():
    """Patch AscendUnquantizedFusedMoEMethod.process_weights_after_loading to
    reclaim memory from old weight tensors before allocating new padded tensors.

    During verl IPC weight sync, the vllm-ascend process_weights_after_loading
    method creates new padded+transposed+contiguous tensors for w13_weight and
    w2_weight. The original implementation orphans the old weight tensors while
    allocating new ones, which can OOM on NPU when memory is tight.

    This patch explicitly deletes the old tensor references and triggers
    NPU cache reclaim between allocations.
    """
    try:
        import torch_npu
        from vllm_ascend.ops.fused_moe.fused_moe import AscendUnquantizedFusedMoEMethod

        _original_process = AscendUnquantizedFusedMoEMethod.process_weights_after_loading

        @wraps(_original_process)
        def _patched_process_weights_after_loading(self, layer):
            from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
                UnquantizedFusedMoEMethod,
            )

            UnquantizedFusedMoEMethod.process_weights_after_loading(self, layer)

            # ---- w13_weight ----
            old_w13 = layer.w13_weight.data
            w13_padded = self._maybe_pad_weight(old_w13)
            del old_w13  # free original unpadded data
            w13_data = w13_padded.transpose(1, 2).contiguous()
            del w13_padded  # free intermediate padded tensor
            layer.w13_weight = torch.nn.Parameter(w13_data, requires_grad=False)
            torch_npu.npu.synchronize()
            torch_npu.npu.empty_cache()

            # ---- w2_weight ----
            old_w2 = layer.w2_weight.data
            w2_padded = self._maybe_pad_weight(old_w2)
            del old_w2
            w2_data = w2_padded.transpose(1, 2).contiguous()
            del w2_padded
            layer.w2_weight = torch.nn.Parameter(w2_data, requires_grad=False)
            torch_npu.npu.synchronize()
            torch_npu.npu.empty_cache()

            # ---- NPU format casting (preserved from original) ----
            from vllm_ascend.ascend_config import get_ascend_config
            from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, maybe_trans_nz

            if get_ascend_config().enable_fused_mc2:
                layer.w13_weight.data = torch_npu.npu_format_cast(
                    layer.w13_weight.data, ACL_FORMAT_FRACTAL_NZ
                )
                layer.w2_weight.data = torch_npu.npu_format_cast(
                    layer.w2_weight.data, ACL_FORMAT_FRACTAL_NZ
                )
            else:
                layer.w13_weight.data = maybe_trans_nz(layer.w13_weight.data)
                layer.w2_weight.data = maybe_trans_nz(layer.w2_weight.data)

        AscendUnquantizedFusedMoEMethod.process_weights_after_loading = (
            _patched_process_weights_after_loading
        )
    except ImportError:
        pass


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


if is_torch_npu_available(check_device=False):
    import vllm
    from packaging import version

    # Patch FusedMoE._load_w13 / _load_w2 to handle TP-sharded weights
    # during verl IPC weight sync. This is needed for all vllm versions
    # because the IPC weight may already be TP-partitioned.
    _patch_fused_moe_load_w13_for_npu()

    # Patch vllm-ascend's process_weights_after_loading to avoid OOM during
    # IPC weight sync. The original code creates new padded+contiguous tensors
    # while the old weight tensors are still resident in NPU memory.
    _patch_vllm_ascend_process_weights_for_npu()

    _VLLM_VERSION = version.parse(vllm.__version__)
    if _VLLM_VERSION >= version.parse("0.13.0") and _VLLM_VERSION <= version.parse("0.14.0"):
        # Disable flash_attn in RotaryEmbedding (NPU) when VLLM >= 0.13
        from vllm.model_executor.layers.fused_moe import FusedMoE

        patch_vllm013_rotary_emb()
        FusedMoE.weight_loader = vllm_v013_weight_loader_method_wrapper(FusedMoE.weight_loader)
    elif _VLLM_VERSION >= version.parse("0.19.0"):
        # Disable flash_attn in RotaryEmbedding (NPU) when VLLM >= 0.19
        from vllm.model_executor.layers.fused_moe import FusedMoE

        patch_vllm013_rotary_emb()
        FusedMoE.weight_loader = vllm_v013_weight_loader_method_wrapper(FusedMoE.weight_loader)

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
