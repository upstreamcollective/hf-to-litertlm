"""hf-to-litertlm: a Hugging Face model repo to a .litertlm bundle in one command, on a pinned toolchain.

    hf-to-litertlm <folder or org/model> [--revision SHA] [--dry-run] [--check] [--smoke] ...
    hf-to-litertlm verify out/<model>/<model>.lock.json   rebuild from the lock: MATCH = the same bytes
    hf-to-litertlm check out/<model>/<model>_int8.litertlm   load it on litert-lm (CPU), an echo prompt

The first form is scripts/convert.py of https://github.com/john-rocky/hf-to-litertlm: it writes
out/<model>/<model>_<recipe>.litertlm, <model>.lock.json and convert_report.json under the current
directory. --check runs the entry gate only (downloads no weights); --smoke runs the one-prompt check
instead of the 8-question gate. `hf-to-litertlm <org>/<model> --help` lists every flag; it runs inside
the toolchain environment, so on a new machine it installs that first. Exit code of the first form: 0 =
converted and passed the gate, 1 = the gate or the check failed, 2 = refused or stopped before any
weight file is downloaded, 3 = any later failure. verify exits 0 on MATCH, check exits 0 on PASS.

This package carries those scripts with the repo's pyproject.toml and uv.lock. Each run puts them in
~/.cache/hf-to-litertlm/<version>/ (HF_TO_LITERTLM_HOME replaces ~/.cache/hf-to-litertlm) and runs
them there with `uv run --frozen`: the packages are the ones uv.lock pins, the environment `uv sync`
makes in a clone, so a lock the package wrote verifies in a clone and a clone's lock verifies from the
package, when both have the same uv.lock and Python patch version. The first run installs that
environment there.

--dry-run shows the selected project and recipe commands. With --available-ram-gib it tests
the Gemma 4 RAM stop at a simulated capacity, without executing a conversion.
"""
import argparse
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import types
import zipfile
from pathlib import Path

from . import __version__

SUBCOMMANDS = {"verify": "scripts/verify_lock.py", "check": "scripts/check_bundle.py"}
PAYLOAD = Path(__file__).with_name("payload.zip")
# a user's own venv or project env would put the toolchain somewhere uv.lock does not describe
DROP_ENV = ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT")


def home():
    root = os.environ.get("HF_TO_LITERTLM_HOME") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.join(str(Path.home()), ".cache"), "hf-to-litertlm")
    return Path(root) / __version__


def unpack():
    """Write every payload file whose bytes are not the manifest's, then return the directory."""
    d = home()
    with zipfile.ZipFile(PAYLOAD) as z:
        manifest = json.loads(z.read("MANIFEST.json"))
        for rel, sha in manifest["files"].items():
            p = d / rel
            if not (p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == sha):
                p.parent.mkdir(parents=True, exist_ok=True)
                tmp = p.with_name(f"{p.name}.tmp{os.getpid()}")
                tmp.write_bytes(z.read(rel))
                os.replace(tmp, p)
            # an entry keeps its git mode: verify_quality.py runs litert_lm_cli_verifier.py as a program
            if (z.getinfo(rel).external_attr >> 16) & 0o111 and not os.access(p, os.X_OK):
                os.chmod(p, (p.stat().st_mode & 0o777) | 0o111)
    return d


def uv_bin():
    try:
        from uv import find_uv_bin
        return find_uv_bin()
    except (ImportError, FileNotFoundError):
        found = shutil.which("uv")
    if not found:
        sys.exit("FAIL setup: uv not found -> next: pip install uv, or run uvx hf-to-litertlm")
    return found


def load_payload_script(payload_dir, name):
    """Load a small Python module from the wheel before unpacking the payload."""
    rel = f"scripts/{name}.py"
    module = types.ModuleType(name)
    module.__file__ = str(payload_dir / rel)
    with zipfile.ZipFile(PAYLOAD) as payload:
        exec(compile(payload.read(rel), module.__file__, "exec"), vars(module))
    return module


def routing_args(argv):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False, exit_on_error=False)
    parser.add_argument("model", nargs="?")
    for option in ("revision", "out", "recipe", "min-correct", "gate-script", "max-minutes", "task"):
        parser.add_argument(f"--{option}")
    try:
        return parser.parse_known_args(argv)[0]
    except argparse.ArgumentError:
        # The original converter owns CLI validation.
        return None


def dry_run_args(argv):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--available-ram-gib", type=float)
    args, remaining = parser.parse_known_args(argv)
    if args.available_ram_gib is not None:
        if not args.dry_run:
            parser.error("--available-ram-gib requires --dry-run")
        if not math.isfinite(args.available_ram_gib) or args.available_ram_gib < 0:
            parser.error("--available-ram-gib must be finite and nonnegative")
    return args, remaining


def stop(reason, *, hint=None):
    formatter = load_payload_script(home().resolve(), "failure_line")
    print(formatter.format_line("route", formatter.first_line(reason), hint or formatter.REQUEST_HINT),
          file=sys.stderr)
    return 2


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[0] in ("-V", "--version"):
        print(f"hf-to-litertlm {__version__}")
        return 0
    subcommand = argv.pop(0) if argv[0] in SUBCOMMANDS else None
    script = SUBCOMMANDS[subcommand] if subcommand else "scripts/convert.py"
    dry_run = False
    project = None
    if subcommand is None:
        dry, argv = dry_run_args(argv)
        dry_run = dry.dry_run
    if subcommand is None and not {"--help", "-h"}.intersection(argv):
        args = routing_args(argv)
        source = Path(args.model).expanduser() if args and args.model else None
        if source is not None and source.is_dir():
            recipe = load_payload_script(home().resolve(), "gemma4_v2")
            linux = recipe.supported_platform()
            available = ram_error = None
            if linux and dry.available_ram_gib is None:
                # Sample before an adapter's Hub config lookup.
                try:
                    available = recipe.available_ram_bytes()
                except (OSError, KeyError, ValueError) as e:
                    ram_error = e
            try:
                uses_v2, base, base_revision = recipe.probe(source)
            except recipe.ProbeError as e:
                return stop(str(e), hint=e.hint)
            except (OSError, TypeError, ValueError) as e:
                return stop(str(e))
            if not uses_v2:
                return stop("local folders convert only for Gemma 4 E2B",
                            hint="pass the model's Hugging Face id instead")
            if not dry_run and not linux:
                return stop(recipe.PLATFORM_REFUSAL, hint=recipe.PLATFORM_HINT)
            if dry_run:
                print("route: gemma4_v2 (Linux x86-64)", flush=True)
            if dry.available_ram_gib is not None:
                available = dry.available_ram_gib * 2**30
                print(f"RAM preflight (simulated): {dry.available_ram_gib:.1f} GiB available", flush=True)
            elif ram_error is not None:
                return stop(f"could not read available RAM ({ram_error})")
            if available is not None and (reason := recipe.ram_refusal(available)):
                return stop(reason, hint=recipe.RAM_HINT)
            if str(source) != args.model:
                argv[argv.index(args.model)] = str(source)
            project = recipe.PROJECT
    d = home() if dry_run else unpack()
    env = {k: v for k, v in os.environ.items() if k not in DROP_ENV}
    env["HF_TO_LITERTLM_PACKAGE"] = __version__
    cmd = [uv_bin(), "run", "--frozen", "--project", str(project or d), "python", str(d / script), *argv]
    if dry_run:
        if project is not None:
            if not linux:
                print("execution requires Linux x86-64")
            name = source.resolve().name
            out = (Path(args.out) if args.out else Path("out") / name).resolve()
            src = source.resolve()
            if base is not None:
                print(f"merge: PEFT on CPU, {source} into {base}"
                      + (f" @ {base_revision}" if base_revision else ""))
                prefix = "gemma4-merged-" if "--keep-merged" in argv else ".gemma4-merged-"
                src = str(out / (prefix + "<temporary>"))
            raw = out / ".gemma4-export-<temporary>"
            print(f"export: {recipe.format_export(src, raw)}")
            fix = recipe.fix_command("python", raw / "model.litertlm", raw / "model.fp32.litertlm")
            print(f"post-export: {shlex.join(fix)}")
        else:
            print("route: main (existing converter)")
        print(f"command: {shlex.join(cmd)}")
        return 0
    if os.name == "nt":
        return subprocess.call(cmd, env=env)
    os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    sys.exit(main())
