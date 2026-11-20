"""Route checks and export settings for a local Gemma 4 E2B folder (Linux x86-64)."""
import importlib.util
import json
import os
import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT / "recipes" / "gemma4_v2"
QUANTIZATION = PROJECT / "int4_block128.json"
NOTEBOOK = ROOT / "notebooks" / "unsloth_gemma4_e2b" / "gemma4_to_litertlm.py"
CACHE_LENGTH = 2048
# The weights-in-graph E2B form peaked at 21.48 GiB in Docker stats on 2026-11-20.
MIN_RAM_GIB = 22
RAM_FLOOR_SOURCE = "weights-in-graph form, Docker stats max 21.48 GiB on 2026-11-20"
# The tested containers had a 24 GiB limit. A 22 GiB limit leaves less than MIN_RAM_GIB available and is refused.
RECOMMENDED_RAM_GIB = 24
RAM_HINT = ("close other programs, or run on a machine or container with at least "
            f"{RECOMMENDED_RAM_GIB} GiB of memory")
PLATFORM_REFUSAL = "a local Gemma 4 E2B folder converts on Linux x86-64 only"
PLATFORM_HINT = ("run the same command on Linux x86-64, for example in a linux/amd64 "
                 f"container with at least {RECOMMENDED_RAM_GIB} GiB of memory")
E2B_ONLY = "only Gemma 4 E2B (hidden_size 1536, 35 layers, per-layer inputs) has been tested"
CONFIG_HINT = "pass the folder that holds the model's config.json"
ADAPTER_BASE_HINT = "set base_model_name_or_path in adapter_config.json to the base model's Hugging Face id"
HUB_BASE_HINT = "check base_model_name_or_path in adapter_config.json and your network"


class ProbeError(ValueError):
    def __init__(self, reason, hint=None):
        super().__init__(reason)
        self.hint = hint


def supported_platform():
    return sys.platform == "linux" and platform.machine() == "x86_64"


def is_gemma4_ple(config):
    if config.get("model_type") != "gemma4":
        return False
    text = config.get("text_config") or config
    width = text.get("hidden_size_per_layer_input") if isinstance(text, dict) else None
    return type(width) is int and width > 0


def is_e2b(config):
    if not is_gemma4_ple(config):
        return False
    text = config.get("text_config") or config
    return text.get("hidden_size") == 1536 and text.get("num_hidden_layers") == 35


def e2b_route(config):
    model_type = config.get("model_type")
    if not isinstance(model_type, str) or not model_type.startswith("gemma4"):
        return False
    if not is_e2b(config):
        raise ProbeError(E2B_ONLY)
    return True


def source_json(source, filename, revision=None, *, optional=False):
    """Read local metadata or the Hub config of a local adapter's base."""
    if isinstance(source, Path):
        path = source / filename
        hint = CONFIG_HINT if filename == "config.json" else None
    else:
        from huggingface_hub import hf_hub_download

        hint = HUB_BASE_HINT
        try:
            cached = hf_hub_download(source, filename, revision=revision)
        except Exception as e:  # noqa: BLE001 — both Hub HTTP clients use this preflight hint
            raise ProbeError(str(e), hint) from e
        path = Path(cached)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as e:
        if optional and isinstance(e, FileNotFoundError):
            return None
        raise ProbeError(str(e), hint) from e
    if not isinstance(data, dict):
        raise ValueError(f"{source}/{filename} must contain a JSON object")
    return data


def adapter_base(source, adapter):
    base = adapter.get("base_model_name_or_path")
    if not isinstance(base, str) or not base:
        raise ProbeError("adapter_config.json is missing base_model_name_or_path", ADAPTER_BASE_HINT)
    relative = source / base
    if relative.is_dir():
        return relative.resolve()
    return base


def probe(source):
    """Return the V2 route and merge target of a local folder using only config files."""
    adapter = source_json(source, "adapter_config.json", optional=True)
    if adapter is not None:
        base = adapter_base(source, adapter)
        base_revision = adapter.get("revision")
    else:
        base = base_revision = None
    config = source_json(base if base is not None else source, "config.json", base_revision)
    uses_v2 = e2b_route(config)
    return uses_v2, base, base_revision


def export_kwargs(template):
    """The tested export options, with the folder's own chat template."""
    return {
        "task": "text_generation",
        "quantization_recipe": str(QUANTIZATION),
        "prefill_lengths": [1024, 128],
        "cache_length": CACHE_LENGTH,
        "externalize_embedder": True,
        "externalize_weights": False,
        "use_jinja_template": True,
        "use_v2": True,
        "delete_in_memory_params": True,
        "model_dtype": "bfloat16",
        "apply_gpu_composites": True,
        "use_qkv_norm_rope_composite": False,
        "use_sdpa_composite": False,
        "fuse_gate_up": False,
        "fuse_qkv": False,
        "use_rope_composite": False,
        "use_swiglu_composite": False,
        "use_short_conv_composite": False,
        "use_bool_mask": False,
        "enable_gpu_dynamic_cache": False,
        "jinja_chat_template_override": str(template),
    }


def format_export(source, output_dir):
    args = {"model": str(source), "output_dir": str(output_dir),
            **export_kwargs(output_dir / "chat_template.bundle.jinja")}
    return "litert_torch.generative.export_hf.export.export(" + ", ".join(
        f"{key}={value!r}" for key, value in args.items()
    ) + ")"


def available_ram_bytes(proc=Path("/proc"), cgroup=Path("/sys/fs/cgroup")):
    meminfo = (proc / "meminfo").read_text()
    values = {key: value.strip() for key, value in (line.split(":", 1)
                                                    for line in meminfo.splitlines())}
    available = int(values["MemAvailable"].split()[0]) * 1024
    for entry in (proc / "self" / "cgroup").read_text().splitlines():
        _, controllers, group = entry.split(":", 2)
        if controllers == "":
            root = cgroup
            limit_name, usage_name, reclaim_key = "memory.max", "memory.current", "inactive_file"
        elif "memory" in controllers.split(","):
            root = cgroup / "memory"
            limit_name = "memory.limit_in_bytes"
            usage_name, reclaim_key = "memory.usage_in_bytes", "total_inactive_file"
        else:
            continue
        # The visible mount root can be the process cgroup in a container.
        relative = Path(group.lstrip("/"))
        if ".." in relative.parts:
            relative = Path()
        member = root / relative
        for directory in (member, *member.parents):
            if directory == root.parent:
                break
            limit = directory / limit_name
            if not limit.is_file():
                continue
            ceiling = limit.read_text().strip()
            if ceiling == "max":
                continue
            usage = int((directory / usage_name).read_text())
            stat = directory / "memory.stat"
            stats = dict(line.split() for line in stat.read_text().splitlines()) if stat.is_file() else {}
            working_set = max(0, usage - int(stats.get(reclaim_key, 0)))
            available = min(available, max(0, int(ceiling) - working_set))
    return available


def ram_refusal(available):
    if available < MIN_RAM_GIB * 2**30:
        return (f"Gemma 4 E2B needs {MIN_RAM_GIB} GiB of available RAM; "
                f"found {available / 2**30:.1f} GiB")
    return None


def prepare_template(source, destination):
    spec = importlib.util.spec_from_file_location("unsloth_notebook_recipe", NOTEBOOK)
    notebook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(notebook)
    template = notebook.read_template(source)
    if not template:
        raise ValueError(f"{source} has no chat template (chat_template.jinja or tokenizer_config.json)")
    proof, rendered_template = notebook.prove_template(template)
    if proof["pass"]:
        destination.write_text(rendered_template, encoding="utf-8")
    return proof


def fix_command(python, source, destination):
    return [python, str(NOTEBOOK), "fix", str(source), str(destination)]


def validate_adapter(folder):
    """Check safetensors metadata against the file length before loading the base."""
    from safetensors import SafetensorError, safe_open

    weights = folder / "adapter_model.safetensors"
    if weights.is_file():
        try:
            with safe_open(weights, framework="np"):
                pass
        except (SafetensorError, OSError) as e:
            raise ValueError(f"{weights.name}: {e}") from e


def merge_lora_layers(model, layer_type, fp32):
    """Use PEFT's safe merge in FP32, holding only one adapted layer in FP32."""
    for layer in model.modules():
        if isinstance(layer, layer_type):
            base_layer = layer.get_base_layer()
            dtype = base_layer.weight.dtype
            base_layer.to(dtype=fp32)
            layer.merge(safe_merge=True)
            base_layer.to(dtype=dtype)
    return model.unload()


def merge_adapter(base, adapter, merged):
    """Save a merged checkpoint while avoiding a model-wide FP32 LoRA copy."""
    import torch
    from peft import PeftModel
    from peft.tuners.lora.layer import LoraLayer
    from transformers import Gemma4ForConditionalGeneration

    config = source_json(base, "config.json")
    dtype = getattr(torch, config.get("dtype") or config.get("torch_dtype") or "bfloat16")
    lora = source_json(adapter, "adapter_config.json").get("peft_type") == "LORA"
    model = Gemma4ForConditionalGeneration.from_pretrained(
        base, dtype=dtype if lora else torch.float32, device_map={"": "cpu"}, local_files_only=True,
        trust_remote_code=False,
    ).eval()
    with torch.inference_mode():
        model = PeftModel.from_pretrained(model, adapter, local_files_only=True)
        if lora:
            model = merge_lora_layers(model, LoraLayer, torch.float32)
        else:
            model = model.merge_and_unload(safe_merge=True).to(dtype=dtype)
        model.eval().save_pretrained(merged, safe_serialization=True, max_shard_size="5GB")

    # Copy tokenizer bytes verbatim. An adapter template, if supplied, supersedes the base's.
    tokenizer = adapter if (adapter / "tokenizer.json").is_file() else base
    for path in tokenizer.iterdir():
        if path.is_file() and (path.name.startswith(("tokenizer", "vocab"))
                              or path.name in {"special_tokens_map.json", "added_tokens.json",
                                               "merges.txt"}):
            shutil.copyfile(path, merged / path.name)
    template = None
    for folder in (adapter, base):
        path = folder / "chat_template.jinja"
        if path.is_file():
            template = path.read_bytes()
        else:
            config = source_json(folder, "tokenizer_config.json", optional=True) or {}
            value = config.get("chat_template")
            if isinstance(value, str) and value.strip():
                template = value.encode("utf-8")
        if template is not None:
            (merged / "chat_template.jinja").write_bytes(template)
            break
    for filename in ("processor_config.json", "generation_config.json"):
        for folder in (adapter, base):
            if (folder / filename).is_file():
                shutil.copyfile(folder / filename, merged / filename)
                break


def export(source, output_dir, uuid_value="", timestamp=""):
    # Save and load locally so the builder cannot fetch the official checkpoint's template.
    os.environ["HF_HUB_OFFLINE"] = "1"
    if uuid_value:
        from convert import pin_system_metadata
        pin_system_metadata(uuid_value, timestamp)
    from litert_torch.generative.export_hf.export import export as export_hf

    template = Path(output_dir) / "chat_template.bundle.jinja"
    export_hf(model=str(source), output_dir=str(output_dir), **export_kwargs(template))


if __name__ == "__main__":
    import resource

    action, *args = sys.argv[1:]
    work = {"merge": merge_adapter, "export": export}.get(action)
    if work is None:
        sys.exit(f"unknown Gemma 4 recipe action: {action}")
    try:
        work(*(map(Path, args) if action == "merge" else args))
    finally:
        # Linux reports KiB, macOS bytes. Only this action's child is counted.
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(f"{action}_peak_rss_bytes={peak * (1024 if sys.platform == 'linux' else 1)}", flush=True)
