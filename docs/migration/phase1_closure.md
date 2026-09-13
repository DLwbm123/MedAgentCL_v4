# Phase 1 Closure

## Upstream CLI Audit

Phase 1 added six lines to `swift/cli/main.py` so that an empty command,
`-h`, or `--help` printed the route list. The fixed v4.4.1 implementation
otherwise indexes `sys.argv[1:][0]`, so upstream `swift --help` raises a
`KeyError` while normal subcommands continue to work.

This is not required for registry, training, inference, or subcommand routing.
The change was therefore classified under closure rule A, removed from the
upstream package, and moved to `scripts/swift_cli.sh`. The wrapper only handles
top-level help and delegates every real command unchanged with `exec`.

Evidence is saved under `output/phase2/preflight/`:

- `swift_cli_main_diff.patch`
- `swift_cli_main_audit.md`
- `cli_closure_check.log`

No upstream core patch is retained, so `PATCHES.md` is not required.

## SFT Argument Parsing

`swift sft --help` in v4.4.1 shows only `tuner_backend`. This is not the full
Hugging Face argument parser. `swift/cli/sft.py::try_init_unsloth()` creates a
temporary `argparse.ArgumentParser` with help enabled; it consumes `--help` and
exits before `sft_main()` constructs the 347-field `SftArguments` parser.

The closure test directly constructs `HfArgumentParser([SftArguments])` and
performs a model-weight-free `SftArguments` construction using Qwen3 registry
metadata. The planned `freeze_*`, tuner, target module, external plugin,
dataset, template, and `use_hf` fields are present. `dataset_format` is not a
v4.4.1 field and must not be passed by migrated commands; v4 consumes the
messages/images schema directly.

## Conda Closure

The environment is recorded by `conda-explicit-lock.txt`,
`conda-packages.json`, and `environment_full.yml`, all generated from the live
environment without solving or rebuilding it.

The preserved decord dry-run contains 131 LINK, 4 UNLINK, and 131 FETCH
actions. Python remained 3.12.7 but moved from defaults build
`h5148396_0` to conda-forge build `hc5c86c4_0_cpython`. OpenSSL moved from
defaults 3.5.7 to conda-forge 3.6.3. The transaction linked NumPy 2.5.1,
FFmpeg 8.0.1, decord 0.6.0, Python ABI 3.12, and native multimedia codecs.

The mixed Pip/Conda test covers NumPy/Torch round trips, SciPy, pandas,
PyArrow, datasets, PIL/torchvision, decord, transformers, and editable
ms-swift. It passes without loading model weights.
