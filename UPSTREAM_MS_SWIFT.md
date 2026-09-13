# ms-swift Upstream Baseline

- Upstream repository: `https://github.com/modelscope/ms-swift.git`
- Fixed release tag: `v4.4.1`
- Fixed upstream commit: `98a09c18cdf95ff07051324b9b8cc90f5184b24b`
- Initialization date: `2026-07-13`
- Local integration branch: `medagentcl-v4`
- Project root: `/root/MedAgentCL_v4`

The repository was cloned directly from the official upstream tag. The tag commit
must remain an ancestor of the local branch. Reproduce the local editable package
only after installing the locked third-party environment:

```bash
/root/anaconda3/envs/medagentcl_v4/bin/pip install \
  --no-deps --no-build-isolation -e /root/MedAgentCL_v4
```

`requirements-lock.txt` intentionally excludes the editable `ms-swift` entry.

## Package Sources

- PyTorch 2.5.1 CUDA 12.4 wheels: `https://download.pytorch.org/whl/cu124`
- General Python packages: `https://pypi.org/simple`
- Python 3.12.7 and `decord` build: Conda channels `conda-forge`, then `defaults`
- `decord`: exact Conda build `0.6.0=np2py312h48de876_2`
- NumPy: exact Conda build `2.5.1=py312h33ff503_0`

The PyPI `decord==0.6.0` distribution advertises a generic filename but contains
the internal wheel tag `cp36-cp36m-manylinux2010_x86_64`. Pip 26.1.2 therefore
reports it as unsupported on Python 3.12 even though importing the extension was
possible. Phase 1 uses the official conda-forge Python 3.12 build instead of
editing wheel metadata. This keeps the requested `decord` version unchanged.

## Local Compatibility Patch

The fixed v4.4.1 CLI dispatcher indexes its first argument before handling top-level
help, so upstream `swift --help` raises `KeyError`. The local branch adds an early
help branch in `swift/cli/main.py`; command routing and command-specific help remain
unchanged. This is covered by the offline Phase 1 CLI smoke test.

`requirements-lock.txt` is a version lock, not a cryptographic hash lock.
