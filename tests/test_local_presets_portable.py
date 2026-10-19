"""Backend-aware presets keep 64K per slot without loading a model."""

import configparser

import pytest

from local.presets import FILES, model_presets, write_presets


def test_cpu_subset_disables_gpu_flash_attention_and_mtp(tmp_path):
    values = model_presets(tmp_path, backend="cpu", aliases=["bonsai-2b", "bonsai-9b"])
    assert set(values) == {"bonsai-2b", "bonsai-9b"}
    for preset in values.values():
        assert preset["n-gpu-layers"] == "0"
        assert preset["flash-attn"] == "off"
        assert preset["spec-type"] == "none"
        assert preset["cache-type-v"] == "f16"
        assert int(preset["ctx-size"]) // int(preset["parallel"]) == 65536
    assert "spec-draft-n-max" not in values["bonsai-9b"]


def test_default_cuda_catalog_and_tuning_are_preserved(tmp_path):
    default = model_presets(tmp_path)
    assert set(default) == set(FILES)
    assert default == model_presets(tmp_path, backend="cuda")
    assert default["bonsai-9b"]["spec-type"] == "draft-mtp"
    for preset in default.values():
        assert preset["n-gpu-layers"] == "99"
        assert preset["flash-attn"] == "on"
        assert int(preset["ctx-size"]) // int(preset["parallel"]) == 65536


@pytest.mark.parametrize("kwargs", [{"backend": "metal"}, {"aliases": ["../../unknown"]}])
def test_unknown_backend_or_alias_is_rejected_without_write(tmp_path, kwargs):
    with pytest.raises(ValueError):
        write_presets(tmp_path, **kwargs)
    assert not (tmp_path / "local/models.ini").exists()


def test_write_only_requested_presets_in_existing_app_root(tmp_path):
    (tmp_path / "local").mkdir()
    path = write_presets(tmp_path, backend="cpu", aliases=["bonsai-2b", "bonsai-2b"])
    parser = configparser.ConfigParser()
    parser.read_string("[global]\n" + path.read_text(encoding="utf-8"))
    assert parser.sections() == ["global", "bonsai-2b"]
    assert parser["bonsai-2b"]["n-gpu-layers"] == "0"
    assert model_presets(tmp_path, aliases=[]) == {}
