"""LLaDA-Image backend: FP8 dequantisation, model resolution, scheduler flag, status. No GPU, no weights."""
import json
import math

import pytest
import torch

from server import llada, server_config


def _ref(w, s, block=128):
    n, k = w.shape
    grid = s.repeat_interleave(block, 0)[:n].repeat_interleave(block, 1)[:, :k]
    return w.float() * grid


@pytest.mark.parametrize("shape", [(300, 200), (11520, 3840), (3840, 3840), (3840, 10240)])
def test_dequant_block_fp8_matches_reference(shape):
    n, k = shape
    w = torch.randn(n, k).to(torch.float8_e4m3fn)
    s = torch.rand(math.ceil(n / 128), math.ceil(k / 128)) + 0.5
    out = llada._dequant_block_fp8(w, s, dtype=torch.float32)
    assert out.shape == (n, k) and torch.allclose(out, _ref(w, s))
    if n != k:
        out_t = llada._dequant_block_fp8(w, s.t().contiguous(), dtype=torch.float32, transposed=True)
        assert torch.allclose(out_t, _ref(w, s))
        with pytest.raises(RuntimeError, match="does not match weight"):
            llada._dequant_block_fp8(w, s.t().contiguous(), dtype=torch.float32)


def test_detect_scale_orientation():
    w = torch.zeros(256, 128).to(torch.float8_e4m3fn)
    assert llada._detect_scale_orientation({"a.weight": w, "a.weight_scale_inv": torch.ones(2, 1)}) is False
    assert llada._detect_scale_orientation({"a.weight": w, "a.weight_scale_inv": torch.ones(1, 2)}) is True
    sq = torch.zeros(128, 128).to(torch.float8_e4m3fn)
    assert llada._detect_scale_orientation({"a.weight": sq, "a.weight_scale_inv": torch.ones(1, 1)}) is False
    with pytest.raises(RuntimeError, match="neither orientation"):
        llada._detect_scale_orientation({"a.weight": w, "a.weight_scale_inv": torch.ones(3, 3)})


def test_load_fp8_state_dict_dequantises_pairs(tmp_path):
    from safetensors.torch import save_file

    w = torch.randn(256, 128).to(torch.float8_e4m3fn)
    s = torch.rand(2, 1) + 0.5
    bias = torch.randn(256)
    save_file({"blk.weight": w, "blk.weight_scale_inv": s, "blk.bias": bias, "emb.weight": torch.randn(4, 4).bfloat16()},
              str(tmp_path / "diffusion_pytorch_model.safetensors"))
    state, n_dq = llada._load_fp8_state_dict(tmp_path, torch.bfloat16)
    assert n_dq == 1 and set(state) == {"blk.weight", "blk.bias", "emb.weight"}
    assert state["blk.weight"].dtype == torch.bfloat16
    assert torch.allclose(state["blk.weight"].float(), _ref(w, s).bfloat16().float())


def test_resolve_model_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(server_config, "_config", {**server_config.get_all(), "llada_model_dir": str(tmp_path)})
    (tmp_path / "LLaDA-Image-Turbo-FP8").mkdir()
    (tmp_path / "LLaDA-Image-Turbo-FP8" / "model_index.json").write_text("{}")
    (tmp_path / "custom").mkdir()
    (tmp_path / "custom" / "model_index.json").write_text("{}")
    assert llada.resolve_model_dir("turbo") == ("turbo", tmp_path / "LLaDA-Image-Turbo-FP8")
    assert llada.resolve_model_dir("LLaDA-Image-Turbo-FP8")[0] == "turbo"
    assert llada.resolve_model_dir("custom") == ("custom", tmp_path / "custom")
    with pytest.raises(ValueError, match=r"Available in .*: \['LLaDA-Image-Turbo-FP8', 'custom'\]"):
        llada.resolve_model_dir("base")
    for bad in ("../x", "C:\\x", "sub/dir", ".hidden", ".."):
        with pytest.raises(ValueError, match="folder name inside"):
            llada.resolve_model_dir(bad)


def test_load_scheduler_reregisters_dropped_flag(tmp_path):
    cfg = {"_class_name": "FlowMatchEulerDiscreteScheduler", "num_train_timesteps": 1000, "shift": 3.0,
           "use_uniform_sigmas": True, "stochastic_sampling": True}
    (tmp_path / "scheduler").mkdir()
    (tmp_path / "scheduler" / "scheduler_config.json").write_text(json.dumps(cfg))
    sched = llada._load_scheduler(tmp_path)
    assert sched.config.get("use_uniform_sigmas") is True


def test_status_when_nothing_loaded():
    st = llada.status()
    assert st["available"] is True and st["loaded"] is False and st["variants"]["turbo"]["default_steps"] == 4
    assert st["variants"]["base"]["default_guidance"] == 5.0 and "model_root" in st


def test_resolve_defaults_per_variant(monkeypatch):
    monkeypatch.setattr(llada, "_variant", "turbo")
    assert llada._resolve_defaults(None, None) == (4, 1.0)
    assert llada._resolve_defaults(8, None) == (8, 1.0)
    monkeypatch.setattr(llada, "_variant", "base")
    assert llada._resolve_defaults(None, 3.0) == (50, 3.0)


def test_split_fused_keys():
    q, k, v = torch.arange(6.).view(2, 3), torch.arange(6., 12.).view(2, 3), torch.arange(12., 18.).view(2, 3)
    w1, w3 = torch.ones(4, 3), torch.zeros(4, 3)
    state = {"layers.0.attention.to_qkv.weight": torch.cat([q, k, v]), "layers.0.feed_forward.w13.weight": torch.cat([w1, w3]),
             "layers.0.feed_forward.w2.weight": torch.ones(3, 4)}
    assert llada._split_fused_keys(state) == 2
    assert torch.equal(state["layers.0.attention.to_q.weight"], q) and torch.equal(state["layers.0.attention.to_k.weight"], k)
    assert torch.equal(state["layers.0.attention.to_v.weight"], v)
    assert torch.equal(state["layers.0.feed_forward.w1.weight"], w1) and torch.equal(state["layers.0.feed_forward.w3.weight"], w3)
    assert "layers.0.attention.to_qkv.weight" not in state and "layers.0.feed_forward.w13.weight" not in state
    assert "layers.0.feed_forward.w2.weight" in state
