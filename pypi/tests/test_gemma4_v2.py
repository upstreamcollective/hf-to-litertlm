"""Recipe and launcher tests. Exporter, PEFT and device runtime stay outside the test process."""
import argparse
import io
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import convert
import failure_line
import gemma4_v2
from hf_to_litertlm import cli

GEMMA4 = {"model_type": "gemma4", "text_config": {
    "hidden_size_per_layer_input": 256, "hidden_size": 1536, "num_hidden_layers": 35,
}}
GIB = 2**30
RAM_FAILURE = (
    "FAIL route: Gemma 4 E2B needs 22 GiB of available RAM; found {available} GiB -> next: "
    "close other programs, or run on a machine or container with at least 24 GiB of memory\n"
)
SCOPE_FAILURE = (
    "FAIL route: only Gemma 4 E2B (hidden_size 1536, 35 layers, per-layer inputs) has been tested "
    "-> next: open a model request with this line: "
    "https://github.com/john-rocky/hf-to-litertlm/issues/new?template=model-request.yml\n"
)
PLATFORM_FAILURE = (
    "FAIL route: a local Gemma 4 E2B folder converts on Linux x86-64 only "
    "-> next: run the same command on Linux x86-64, for example in a linux/amd64 container "
    "with at least 24 GiB of memory\n"
)
LOCAL_ONLY_FAILURE = (
    "FAIL route: local folders convert only for Gemma 4 E2B -> next: "
    "pass the model's Hugging Face id instead\n"
)
FALLBACK_HINT = ("open a model request with this line: "
                 "https://github.com/john-rocky/hf-to-litertlm/issues/new?template=model-request.yml")


class LauncherTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.merged = self.root / "merged checkpoint"
        self.merged.mkdir()
        (self.merged / "config.json").write_text(json.dumps(GEMMA4))
        self.adapter = self.root / "adapter"
        self.adapter.mkdir()
        (self.adapter / "adapter_config.json").write_text(json.dumps({
            "base_model_name_or_path": "org/gemma", "revision": "trained-base",
        }))
        for name, value in (("home", gemma4_v2.ROOT), ("unpack", gemma4_v2.ROOT), ("uv_bin", "uv")):
            replace = patch.object(cli, name, return_value=value)
            replace.start()
            self.addCleanup(replace.stop)
        modules = {"gemma4_v2": gemma4_v2, "failure_line": failure_line}
        loader = patch.object(cli, "load_payload_script", side_effect=lambda _, name: modules[name])
        loader.start()
        self.addCleanup(loader.stop)

    def test_gemma_is_selected_by_config_and_positive_ple(self):
        self.assertTrue(gemma4_v2.is_gemma4_ple(GEMMA4))
        self.assertTrue(gemma4_v2.is_gemma4_ple({"model_type": "gemma4",
                                               "hidden_size_per_layer_input": 256}))
        self.assertFalse(gemma4_v2.is_gemma4_ple({"model_type": "gemma4",
                                                "hidden_size_per_layer_input": 0}))
        self.assertFalse(gemma4_v2.is_gemma4_ple({"model_type": "gemma4_text",
                                                "hidden_size_per_layer_input": 256}))

    def test_merged_dry_run_on_mac_prints_linux_call(self):
        out = self.root / "output"
        stdout = io.StringIO()
        with patch.object(gemma4_v2, "supported_platform", return_value=False), redirect_stdout(stdout):
            self.assertEqual(cli.main(["--out", str(out), str(self.merged), "--dry-run"]), 0)
        text = stdout.getvalue()
        self.assertIn("route: gemma4_v2 (Linux x86-64)", text)
        self.assertIn("execution requires Linux x86-64\n", text)
        self.assertIn(gemma4_v2.format_export(self.merged, out / ".gemma4-export-<temporary>"), text)

    def test_adapter_dry_run_uses_pinned_base_and_merged_directory(self):
        with patch("huggingface_hub.hf_hub_download",
                   return_value=str(self.merged / "config.json")) as download, redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(cli.main([str(self.adapter), "--dry-run"]), 0)
        download.assert_called_once_with("org/gemma", "config.json", revision="trained-base")
        self.assertIn("into org/gemma @ trained-base", stdout.getvalue())
        self.assertIn("adapter/.gemma4-merged-<temporary>", stdout.getvalue())

    def test_unsupported_platform_stops_before_uv(self):
        for source in (self.merged, self.adapter):
            with self.subTest(source=source), patch.object(gemma4_v2, "supported_platform", return_value=False), \
                    patch("huggingface_hub.hf_hub_download", return_value=str(self.merged / "config.json")), \
                    patch.object(cli, "uv_bin") as uv, redirect_stderr(io.StringIO()) as stderr:
                self.assertEqual(cli.main([str(source)]), 2)
                uv.assert_not_called()
                self.assertEqual(stderr.getvalue(), PLATFORM_FAILURE)

    def test_other_gemma4_sizes_refuse_before_unpack(self):
        variants = [{"hidden_size": 2560, "num_hidden_layers": 42, "hidden_size_per_layer_input": 256},
                    {"hidden_size": 1536, "num_hidden_layers": 36, "hidden_size_per_layer_input": 256},
                    {"hidden_size": 1536, "num_hidden_layers": 35, "hidden_size_per_layer_input": 0}]
        for config in variants:
            self.assert_local_scope_failure({"model_type": "gemma4", "text_config": config}, SCOPE_FAILURE)
        self.assert_local_scope_failure({"model_type": "gemma4_text", "text_config": GEMMA4["text_config"]},
                                        SCOPE_FAILURE)

    def test_non_gemma_local_folders_refuse_on_either_platform(self):
        self.assert_local_scope_failure({"model_type": "qwen3"}, LOCAL_ONLY_FAILURE)

    def assert_local_scope_failure(self, config, expected):
        path = self.merged / "config.json"
        path.write_text(json.dumps(config))
        for source in (self.merged, self.adapter):
            for linux in (True, False):
                with self.subTest(source=source, linux=linux, config=config), \
                        patch.object(gemma4_v2, "supported_platform", return_value=linux), \
                        patch.object(gemma4_v2, "available_ram_bytes", return_value=16 * GIB), \
                        patch("huggingface_hub.hf_hub_download", return_value=str(path)), \
                        patch.object(cli, "unpack") as unpack, redirect_stderr(io.StringIO()) as stderr:
                    self.assertEqual(cli.main([str(source)]), 2)
                    unpack.assert_not_called()
                    self.assertEqual(stderr.getvalue(), expected)

    def test_live_ram_stop_precedes_download_unpack_and_environment_setup(self):
        with patch.object(gemma4_v2, "supported_platform", return_value=True), \
                patch.object(gemma4_v2, "available_ram_bytes", return_value=16 * GIB), \
                patch("huggingface_hub.hf_hub_download") as download, \
                patch.object(cli, "unpack") as unpack, patch.object(cli, "uv_bin") as uv, \
                redirect_stderr(io.StringIO()) as stderr:
            self.assertEqual(cli.main([str(self.merged)]), 2)
        download.assert_not_called()
        unpack.assert_not_called()
        uv.assert_not_called()
        self.assertEqual(stderr.getvalue(), RAM_FAILURE.format(available="16.0"))

    def test_live_ram_sample_precedes_the_adapter_base_probe(self):
        calls = []

        def memory():
            calls.append("memory")
            return 16 * GIB

        def probe(*args):
            calls.append("metadata")
            return True, None, None

        with patch.object(gemma4_v2, "supported_platform", return_value=True), \
                patch.object(gemma4_v2, "available_ram_bytes", side_effect=memory), \
                patch.object(gemma4_v2, "probe", side_effect=probe), redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main([str(self.adapter)]), 2)
        self.assertEqual(calls, ["memory", "metadata"])

    def test_dry_run_below_ram_floor_on_mac_stops_before_showing_export(self):
        with patch.object(gemma4_v2, "supported_platform", return_value=False), \
                redirect_stderr(io.StringIO()) as stderr, redirect_stdout(io.StringIO()) as stdout:
            rc = cli.main([str(self.merged), "--dry-run", "--available-ram-gib", "20"])
        self.assertEqual(rc, 2)
        self.assertIn("route: gemma4_v2 (Linux x86-64)", stdout.getvalue())
        self.assertNotIn("export:", stdout.getvalue())
        self.assertEqual(stderr.getvalue(), RAM_FAILURE.format(available="20.0"))

    def test_ram_simulation_cannot_override_a_live_conversion(self):
        with patch.object(cli, "uv_bin") as uv, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as stopped:
                cli.main([str(self.merged), "--available-ram-gib", "1000"])
        self.assertEqual(stopped.exception.code, 2)
        uv.assert_not_called()

    @unittest.skipIf(os.name == "nt", "Unix launcher uses execvpe")
    def test_linux_dispatches_both_local_e2b_forms(self):
        for source in (self.merged, self.adapter):
            with self.subTest(source=source), \
                    patch.object(gemma4_v2, "supported_platform", return_value=True), \
                    patch.object(gemma4_v2, "available_ram_bytes", return_value=24 * GIB), \
                    patch("huggingface_hub.hf_hub_download", return_value=str(self.merged / "config.json")), \
                    patch.object(cli.os, "execvpe") as execute:
                cli.main([str(source)])
            cmd = execute.call_args.args[1]
            self.assertEqual(cmd, ["uv", "run", "--frozen", "--project", str(gemma4_v2.PROJECT), "python",
                                   str(gemma4_v2.ROOT / "scripts/convert.py"), str(source)])

    @unittest.skipIf(os.name == "nt", "Unix launcher uses execvpe")
    def test_hub_ids_always_forward_legacy_command_without_loading_recipe(self):
        ids = ("google/gemma-4-E2B-it", "google/gemma-4-E4B-it", "org/gemma-lora", "org/llama")
        for model in ids:
            for linux in (True, False):
                argv = ["--revision", "abc", model, "--recipe", "int4"]
                with self.subTest(model=model, linux=linux), \
                        patch.object(gemma4_v2, "supported_platform", return_value=linux), \
                        patch.object(cli, "load_payload_script") as load, \
                        patch("huggingface_hub.hf_hub_download") as download, \
                        patch.object(cli.os, "execvpe") as execute:
                    cli.main(argv)
                load.assert_not_called()
                download.assert_not_called()
                self.assertEqual(execute.call_args.args[1],
                                 ["uv", "run", "--frozen", "--project", str(gemma4_v2.ROOT), "python",
                                  str(gemma4_v2.ROOT / "scripts/convert.py"), *argv])

    @unittest.skipIf(os.name == "nt", "Unix launcher uses execvpe")
    def test_verify_forwards_to_main_with_options_before_lock(self):
        lock = self.root / "source.lock.json"
        lock.write_text('{"route": "gemma4_v2"}')
        with patch.object(cli, "load_payload_script") as load, patch.object(cli.os, "execvpe") as execute:
            cli.main(["verify", "--json", "compare.json", str(lock)])
        load.assert_not_called()
        self.assertIn(str(gemma4_v2.ROOT), execute.call_args.args[1])
        self.assertIn(str(gemma4_v2.ROOT / "scripts/verify_lock.py"), execute.call_args.args[1])

    def test_unreadable_local_config_has_folder_hint(self):
        path = self.merged / "config.json"
        path.unlink()
        with redirect_stderr(io.StringIO()) as stderr:
            self.assertEqual(cli.main([str(self.merged)]), 2)
        self.assertEqual(stderr.getvalue(), f"FAIL route: [Errno 2] No such file or directory: '{path}' "
                         "-> next: pass the folder that holds the model's config.json\n")

    def test_missing_adapter_base_has_field_hint(self):
        (self.adapter / "adapter_config.json").write_text("{}")
        with redirect_stderr(io.StringIO()) as stderr:
            self.assertEqual(cli.main([str(self.adapter)]), 2)
        self.assertEqual(stderr.getvalue(), "FAIL route: adapter_config.json is missing base_model_name_or_path "
                         "-> next: set base_model_name_or_path in adapter_config.json to the base model's "
                         "Hugging Face id\n")

    def test_unreadable_adapter_hub_base_has_config_and_network_hint(self):
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
        from httpx import ConnectError, Request, Response

        request = Request("HEAD", "https://huggingface.co/org/gemma/resolve/main/config.json")
        errors = (GatedRepoError("base unavailable", response=Response(403, request=request)),
                  RepositoryNotFoundError("base unavailable", response=Response(404, request=request)),
                  ConnectError("base unavailable"))
        for error in errors:
            with self.subTest(exc=type(error)), \
                    patch("huggingface_hub.hf_hub_download", side_effect=error), \
                    redirect_stderr(io.StringIO()) as stderr:
                self.assertEqual(cli.main([str(self.adapter)]), 2)
                self.assertEqual(stderr.getvalue(), "FAIL route: base unavailable -> next: "
                                 "check base_model_name_or_path in adapter_config.json and your network\n")

    def test_non_object_json_has_fallback_hint(self):
        (self.merged / "config.json").write_text("[]")
        with redirect_stderr(io.StringIO()) as stderr:
            self.assertEqual(cli.main([str(self.merged)]), 2)
        self.assertEqual(stderr.getvalue(), f"FAIL route: {self.merged}/config.json must contain a JSON object "
                         f"-> next: {FALLBACK_HINT}\n")

    def test_unreadable_ram_has_fallback_hint(self):
        with patch.object(gemma4_v2, "supported_platform", return_value=True), \
                patch.object(gemma4_v2, "available_ram_bytes", side_effect=OSError("meminfo unavailable")), \
                redirect_stderr(io.StringIO()) as stderr:
            self.assertEqual(cli.main([str(self.merged)]), 2)
        self.assertEqual(stderr.getvalue(), "FAIL route: could not read available RAM (meminfo unavailable) "
                         f"-> next: {FALLBACK_HINT}\n")


class RecipeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.out = self.root / "out"
        self.out.mkdir()
        self.base = self.root / "base"
        self.base.mkdir()
        (self.base / "config.json").write_text(json.dumps(GEMMA4))
        (self.base / "model.safetensors").touch()
        self.args = argparse.Namespace(model=str(self.base), revision=None, task=None,
                                       recipe="int4", check=False, curated=False, keep_merged=False)
        self.report = {"model": self.args.model, "output_dir": str(self.out), "decisions": {}}
        no_render = patch.object(gemma4_v2, "prepare_template", return_value={"pass": True})
        no_render.start()
        self.addCleanup(no_render.stop)

    def test_memory_counts_headroom_on_each_cgroup_parent(self):
        proc = self.root / "proc"
        (proc / "self").mkdir(parents=True)
        (proc / "meminfo").write_text(f"MemAvailable: {40 * GIB // 1024} kB\n")
        (proc / "self" / "cgroup").write_text("0::/parent/child\n")
        cgroup = self.root / "cgroup"
        parent = cgroup / "parent"
        child = parent / "child"
        child.mkdir(parents=True)
        for directory, limit, current, inactive in ((parent, 24, 7, 3), (child, 30, 5, 2)):
            (directory / "memory.max").write_text(str(limit * GIB))
            (directory / "memory.current").write_text(str(current * GIB))
            (directory / "memory.stat").write_text(f"inactive_file {inactive * GIB}\n")
        self.assertEqual(gemma4_v2.available_ram_bytes(proc, cgroup), 20 * GIB)

    def test_memory_uses_memavailable_when_below_the_cgroup_headroom(self):
        proc = self.root / "proc"
        (proc / "self").mkdir(parents=True)
        (proc / "meminfo").write_text(f"MemAvailable: {12 * GIB // 1024} kB\n")
        (proc / "self" / "cgroup").write_text("0::/\n")
        cgroup = self.root / "cgroup"
        cgroup.mkdir()
        (cgroup / "memory.max").write_text(str(24 * GIB))
        (cgroup / "memory.current").write_text(str(2 * GIB))
        self.assertEqual(gemma4_v2.available_ram_bytes(proc, cgroup), 12 * GIB)
        (cgroup / "memory.max").write_text("max")
        self.assertEqual(gemma4_v2.available_ram_bytes(proc, cgroup), 12 * GIB)

    def test_small_working_set_requires_more_than_a_22_gib_container_limit(self):
        proc = self.root / "proc"
        (proc / "self").mkdir(parents=True)
        (proc / "meminfo").write_text(f"MemAvailable: {40 * GIB // 1024} kB\n")
        (proc / "self" / "cgroup").write_text("0::/\n")
        cgroup = self.root / "cgroup"
        cgroup.mkdir()
        (cgroup / "memory.current").write_text(str(GIB // 2))
        limit = cgroup / "memory.max"
        limit.write_text(str(22 * GIB))
        self.assertEqual(gemma4_v2.ram_refusal(gemma4_v2.available_ram_bytes(proc, cgroup)),
                         "Gemma 4 E2B needs 22 GiB of available RAM; found 21.5 GiB")
        limit.write_text(str(24 * GIB))
        self.assertIsNone(gemma4_v2.ram_refusal(gemma4_v2.available_ram_bytes(proc, cgroup)))

    def test_memory_respects_cgroup_v1_limit_and_usage(self):
        proc = self.root / "proc"
        (proc / "self").mkdir(parents=True)
        (proc / "meminfo").write_text(f"MemAvailable: {40 * GIB // 1024} kB\n")
        (proc / "self" / "cgroup").write_text("3:cpu,cpuacct:/test\n4:memory:/test\n")
        cgroup = self.root / "cgroup"
        group = cgroup / "memory" / "test"
        group.mkdir(parents=True)
        (group / "memory.limit_in_bytes").write_text(str(16 * GIB))
        (group / "memory.usage_in_bytes").write_text(str(2 * GIB))
        (group / "memory.stat").write_text(f"total_inactive_file {GIB}\n")
        self.assertEqual(gemma4_v2.available_ram_bytes(proc, cgroup), 15 * GIB)

    def test_nightly_export_keeps_weights_in_graph_for_notebook_fix(self):
        kwargs = gemma4_v2.export_kwargs("template.jinja")
        self.assertIs(kwargs["externalize_weights"], False)
        self.assertIs(kwargs["externalize_embedder"], True)

    def test_low_memory_stops_before_merge_or_export(self):
        with patch.object(gemma4_v2, "supported_platform", return_value=True), \
                patch.object(gemma4_v2, "available_ram_bytes", return_value=20 * GIB), \
                patch.object(convert.subprocess, "run") as run, patch.object(convert, "_REPORT", self.report), \
                redirect_stderr(io.StringIO()) as stderr:
            with self.assertRaises(SystemExit) as stopped:
                convert.run_gemma4(self.args, self.report, self.out, "base")
        self.assertEqual(stopped.exception.code, 2)
        self.assertEqual(stderr.getvalue(), RAM_FAILURE.format(available="20.0"))
        self.assertEqual(self.report["memory"]["available_bytes"], 20 * GIB)
        run.assert_not_called()

    def test_converter_platform_and_scope_refusals_match_launcher(self):
        cfg = {"model_type": "gemma4", "text_config": {
            **GEMMA4["text_config"], "hidden_size": 2560,
        }}
        (self.base / "config.json").write_text(json.dumps(cfg))
        self.args.check = True
        for supported, expected in ((False, PLATFORM_FAILURE), (True, SCOPE_FAILURE)):
            with self.subTest(supported=supported), \
                    patch.object(gemma4_v2, "supported_platform", return_value=supported), \
                    patch.object(convert, "_REPORT", self.report), redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as stopped:
                    convert.run_gemma4(self.args, self.report, self.out, "base")
                self.assertEqual(stopped.exception.code, 2)
                self.assertEqual(stderr.getvalue(), expected)

    def make_tiny_bundle(self, *, mixed_types=False, external_weights=False, system_metadata=None):
        import flatbuffers
        from ai_edge_litert import schema_py_generated as schema
        from litert_lm_builder import litertlm_builder as lb

        tensor = schema.TensorT()
        tensor.name = "bf16_weight"
        tensor.shape = [2]
        tensor.type = schema.TensorType.BFLOAT16
        tensor.buffer = 1
        weights = schema.BufferT()
        weights.data = list(struct.pack("<HH", 0x3f80, 0xc020))
        graph = schema.SubGraphT()
        graph.name = "test"
        graph.tensors = [tensor]
        graph.inputs = []
        graph.outputs = [0]
        graph.operators = []
        if mixed_types:
            shared = schema.TensorT()
            shared.name = "shared_float_weight"
            shared.shape = [1]
            shared.type = schema.TensorType.FLOAT32
            shared.buffer = 1
            graph.tensors.append(shared)
            graph.outputs.append(1)
        model = schema.ModelT()
        model.version = 3
        model.subgraphs = [graph]
        model.buffers = [schema.BufferT(), weights]
        model.operatorCodes = []
        if external_weights:
            model.externalBuffers = [schema.ExternalBufferT()]
        buffer = flatbuffers.Builder(1024)
        buffer.Finish(model.Pack(buffer), file_identifier=b"TFL3")
        tflite = self.root / "tiny.tflite"
        tflite.write_bytes(buffer.Output())
        builder = lb.LitertLmFileBuilder()
        builder.add_tflite_model(str(tflite), "prefill_decode")
        tiny = self.root / "tiny.litertlm"
        with patch.object(lb, "populate_system_metadata", lb.populate_system_metadata), \
                tiny.open("wb") as output:
            if system_metadata:
                convert.pin_system_metadata(*system_metadata)
            builder.build(output, weights_in_model_graph=True)
        self.assertLess(tflite.stat().st_size, 1024)
        self.assertLess(tiny.stat().st_size, 32 * 1024)
        return tiny

    def run_with_tiny_export(self, fixture, name, *, expected_exit=None):
        real_run = subprocess.run
        calls = []

        def child(cmd, **kwargs):
            script, action, *args = cmd[1:]
            calls.append((Path(script), action, args))
            if action == "export":
                self.assertEqual(kwargs["env"]["HF_HUB_OFFLINE"], "1")
                source = fixture or self.make_tiny_bundle(system_metadata=args[2:])
                shutil.copyfile(source, Path(args[1]) / "model.litertlm")
            if action == "merge":
                shutil.copytree(Path(args[0]), Path(args[2]), dirs_exist_ok=True)
            if action == "fix":
                return real_run(cmd, **kwargs)
            return SimpleNamespace(returncode=0)

        with patch.object(gemma4_v2, "supported_platform", return_value=True), \
                patch.object(gemma4_v2, "available_ram_bytes", return_value=22 * GIB), \
                patch.object(convert.subprocess, "run", side_effect=child), \
                patch.object(convert, "finish_conversion") as finish, \
                patch.object(convert, "_REPORT", self.report), patch.dict(os.environ), \
                redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()) as stderr:
            if expected_exit:
                with self.assertRaises(SystemExit) as stopped:
                    convert.run_gemma4(self.args, self.report, self.out, name)
                self.assertEqual(stopped.exception.code, expected_exit)
                finish.assert_not_called()
            else:
                convert.run_gemma4(self.args, self.report, self.out, name)
                self.assertEqual(os.environ["LITERT_LM_CLI"], str(Path(sys.executable).parent / "litert-lm"))
                self.assertEqual(finish.call_args.kwargs["gate_backend"], "cpu")
        return calls, stdout.getvalue(), stderr.getvalue()

    def test_local_metadata_depends_on_contents_and_toolchain(self):
        weights = self.base / "model.safetensors"
        weights.write_bytes(b"first")
        original = convert.gemma4_local_system_metadata(self.base, "toolchain-a")
        moved = self.root / "moved"
        moved.mkdir()
        # Create the same files in reverse order and use different mtimes.
        for source in (weights, self.base / "config.json"):
            target = moved / source.name
            target.write_bytes(source.read_bytes())
            os.utime(target, (1, 1))
        self.assertEqual(original, convert.gemma4_local_system_metadata(moved, "toolchain-a"))
        self.assertNotEqual(original["uuid"],
                            convert.gemma4_local_system_metadata(moved, "toolchain-b")["uuid"])
        (moved / weights.name).write_bytes(b"other")
        self.assertNotEqual(original["uuid"],
                            convert.gemma4_local_system_metadata(moved, "toolchain-a")["uuid"])
        self.assertEqual(original["creation_timestamp"], "1970-01-01T00:00:00.000000+00:00")

    def test_local_metadata_survives_builder_and_rewrite(self):
        first, _, _ = self.run_with_tiny_export(None, "first")
        second, _, _ = self.run_with_tiny_export(None, "second")
        self.assertEqual(first[0][2][2:], second[0][2][2:])
        a = self.out / "first_int4.litertlm"
        b = self.out / "second_int4.litertlm"
        self.assertEqual(a.read_bytes(), b.read_bytes())
        header = convert.read_system_metadata(a)
        self.assertEqual({key: header[key] for key in ("uuid", "creation_timestamp")},
                         self.report["decisions"]["system_metadata"])
        self.assertFalse(self.report["decisions"]["lock"])

    def test_lora_merges_exports_and_checks_fp32_bundle_before_existing_gate(self):
        import numpy as np
        from safetensors.numpy import save_file

        adapter = self.root / "lora"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text(json.dumps({
            "base_model_name_or_path": "../base",
        }))
        save_file({"weight": np.zeros(1, dtype=np.float32)}, adapter / "adapter_model.safetensors")
        self.args.model = str(adapter)
        calls, stdout, _ = self.run_with_tiny_export(self.make_tiny_bundle(), "lora")
        self.assertEqual([action for _, action, _ in calls], ["merge", "export", "fix"])
        merge, export, fix = (args for _, _, args in calls)
        self.assertEqual(merge[:2], [str(self.base), str(adapter)])
        self.assertEqual(export[0], merge[-1])
        expected = convert.gemma4_local_system_metadata(self.base, self.report["toolchain"]["id"])
        self.assertEqual(export[2:], [expected["uuid"], expected["creation_timestamp"]])
        self.assertEqual(calls[2][0], gemma4_v2.NOTEBOOK)
        staging = Path(export[1])
        self.assertEqual([Path(p) for p in fix],
                         [staging / "model.litertlm", staging / "model.fp32.litertlm"])
        self.assertFalse(Path(merge[-1]).exists())
        self.assertFalse(staging.exists())
        fixed = self.out / "lora_int4.litertlm"
        self.assertIn(struct.pack("<ff", 1.0, -2.5), fixed.read_bytes())
        fix_report = json.loads(fixed.with_suffix(".bf16to32.json").read_text())
        self.assertEqual(fix_report["bf16_tensors"], 1)
        self.assertIn("BFLOAT16 tensors 0", stdout)
        self.assertIn("references naming another tensor than the input's none", stdout)
        self.assertFalse(self.report["decisions"]["lock"])

    def test_truncated_adapter_stops_before_loading_base_or_export(self):
        import numpy as np
        from safetensors.numpy import save_file

        adapter = self.root / "lora"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "org/base", "revision": "trained-base"}))
        weights = adapter / "adapter_model.safetensors"
        save_file({"weight": np.zeros(1024, dtype=np.float32)}, weights)
        with weights.open("r+b") as stream:
            stream.truncate(weights.stat().st_size // 2)
        self.args.model = str(adapter)
        with patch.object(gemma4_v2, "supported_platform", return_value=True), \
                patch.object(gemma4_v2, "available_ram_bytes", return_value=24 * GIB), \
                patch.object(convert, "gemma4_base_files", return_value=("sha", {"model.safetensors"})), \
                patch("huggingface_hub.hf_hub_download", return_value=str(self.base / "config.json")), \
                patch.object(convert, "download_source") as download, \
                patch.object(convert.subprocess, "run") as run, patch.object(convert, "_REPORT", self.report), \
                redirect_stderr(io.StringIO()) as stderr:
            with self.assertRaises(SystemExit) as stopped:
                convert.run_gemma4(self.args, self.report, self.out, "lora")
        self.assertEqual(stopped.exception.code, 3)
        self.assertIn("adapter_model.safetensors", stderr.getvalue())
        self.assertIn("save or download the adapter again", stderr.getvalue())
        self.assertNotIn("open a model request", stderr.getvalue())
        download.assert_not_called()
        run.assert_not_called()

    def test_local_adapter_downloads_and_verifies_only_its_hub_base(self):
        import numpy as np
        from safetensors.numpy import save_file

        adapter = self.root / "lora"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "org/base", "revision": "trained-base"}))
        save_file({"weight": np.zeros(1, dtype=np.float32)}, adapter / "adapter_model.safetensors")
        self.args.model = str(adapter)
        names = ["config.json", "model.safetensors"]
        info = SimpleNamespace(sha="base-commit", gated=False,
                               siblings=[SimpleNamespace(rfilename=name) for name in names])
        with patch("huggingface_hub.HfApi.model_info", return_value=info) as model_info, \
                patch("huggingface_hub.hf_hub_download", return_value=str(self.base / "config.json")) as metadata, \
                patch.object(convert, "download_source", return_value=(self.base, set())) as download, \
                patch.object(convert, "verify_source") as verify:
            calls, _, _ = self.run_with_tiny_export(self.make_tiny_bundle(), "lora")
        model_info.assert_called_once_with("org/base", revision="trained-base")
        metadata.assert_called_once_with("org/base", "config.json", revision="base-commit")
        download.assert_called_once_with(self.report, "org/base", "base-commit", names)
        self.assertEqual(verify.call_args.args[2:], ("org/base", "base-commit", self.base, names, set()))
        self.assertEqual([action for _, action, _ in calls], ["merge", "export", "fix"])
        self.assertEqual(calls[0][2][:2], [str(self.base), str(adapter)])
        self.assertFalse(self.report["decisions"]["lock"])

    def test_ram_floor_uses_weights_in_graph_measurement(self):
        self.assertIn("needs 22 GiB", gemma4_v2.ram_refusal(22 * GIB - 1))
        self.assertIsNone(gemma4_v2.ram_refusal(22 * GIB))

    def test_gate_missing_command_propagates_exit_127_into_actionable_failure(self):
        bundle = self.out / "test.litertlm"
        bundle.touch()
        self.report["bundle_bytes"] = 0
        args = argparse.Namespace(no_gate=False, smoke=False, gate_script=None,
                                  min_correct=6, keep_merged=False)
        with patch.object(convert, "resolve_litert_lm_cli", return_value=str(self.root / "missing")), \
                redirect_stderr(io.StringIO()) as stderr, redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as stopped:
                convert.finish_conversion(args, self.report, self.out, bundle, None, None)
        self.assertEqual(stopped.exception.code, 1)
        self.assertEqual(self.report["status"], "converted_gate_failed")
        self.assertIsNone(self.report["check"])
        self.assertIn("FAIL generate: the 8-question check could not start its verifier or the litert-lm command "
                      "(exit 127)", stderr.getvalue())
        self.assertIn("restore the pinned litert-lm command", stderr.getvalue())
        self.assertNotIn("open a model request", stderr.getvalue())

    def test_lora_merge_casts_one_adapted_layer_at_a_time_and_unloads(self):
        ops = []

        class BaseLayer:
            def __init__(self, name):
                self.name = name
                self.weight = SimpleNamespace(dtype="bf16")

            def to(self, *, dtype):
                ops.append((self.name, dtype))
                self.weight.dtype = dtype

        class AdapterLayer:
            def __init__(self, name):
                self.base = BaseLayer(name)

            def get_base_layer(self):
                return self.base

            def merge(self, *, safe_merge):
                ops.append((self.base.name, "merge", self.base.weight.dtype, safe_merge))

        layers = [AdapterLayer("q"), object(), AdapterLayer("k")]
        model = SimpleNamespace(modules=lambda: iter(layers), unload=lambda: "saved base")
        self.assertEqual(gemma4_v2.merge_lora_layers(model, AdapterLayer, "f32"), "saved base")
        self.assertEqual(ops, [("q", "f32"), ("q", "merge", "f32", True), ("q", "bf16"),
                               ("k", "f32"), ("k", "merge", "f32", True), ("k", "bf16")])

    def test_bf16_rewrite_refusal_preserves_the_message_and_stops_release(self):
        _, _, stderr = self.run_with_tiny_export(
            self.make_tiny_bundle(mixed_types=True), "base", expected_exit=3)
        self.assertIn("refused: a buffer is shared by a BFLOAT16 tensor", stderr)
        self.assertEqual(self.report["failure"]["stage"], "bundle")
        self.assertFalse((self.out / "base_int4.litertlm").exists())
        self.assertFalse(tuple(self.out.glob(".gemma4-export-*")))

    def test_bf16_rewrite_refuses_external_buffer_layout(self):
        _, _, stderr = self.run_with_tiny_export(
            self.make_tiny_bundle(external_weights=True), "base", expected_exit=3)
        self.assertIn("refused: external buffers or large custom options", stderr)
        self.assertFalse((self.out / "base_int4.litertlm").exists())


if __name__ == "__main__":
    unittest.main()
