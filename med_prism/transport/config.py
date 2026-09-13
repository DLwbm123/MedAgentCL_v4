from dataclasses import asdict, dataclass
import math


def parse_rank(value):
    if str(value).lower() in {"none", "unbudgeted"}:
        return None
    rank = int(value)
    if rank < 0:
        raise ValueError("repair rank must be nonnegative")
    return rank


@dataclass(frozen=True)
class TPMConfig:
    mode: str = "transport"
    calibration_samples: int = 64
    holdout_samples: int = 64
    tokens_per_sample: int = 8
    repair_rank: int | None = 2
    eta: float = 0.1
    max_relative_edit: float = 0.05
    solve_dtype: str = "float32"
    fallback_fp64: bool = False
    seed: int = 42
    strict: bool = False
    shuffle_teacher_pairing: bool = False
    require_holdout_nonincrease: bool = False
    numerical_atol: float = 1e-8
    numerical_rtol: float = 1e-6
    edit_tolerance: float = 1e-6

    def __post_init__(self):
        if self.mode not in {"off", "transport"}:
            raise ValueError("Unknown TPM mode")
        if min(self.calibration_samples, self.holdout_samples, self.tokens_per_sample) < 1:
            raise ValueError("Calibration counts must be positive")
        if self.repair_rank is not None and (type(self.repair_rank) is not int or self.repair_rank < 0):
            raise ValueError("Invalid repair rank")
        if not math.isfinite(self.eta) or self.eta <= 0:
            raise ValueError("eta must be finite and positive")
        for name in ("max_relative_edit", "numerical_atol", "numerical_rtol", "edit_tolerance"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(name)
        if self.solve_dtype not in {"float32", "float64"}:
            raise ValueError("Only float32/float64 solves are supported")

    def to_dict(self):
        return asdict(self)


def add_tpm_arguments(parser):
    defaults = TPMConfig()
    parser.add_argument("--tpm-mode", choices=["off", "transport"], default=defaults.mode)
    parser.add_argument("--tpm-repair-rank", type=parse_rank, default=2)
    parser.add_argument("--tpm-solve-dtype", choices=["float32", "float64"], default="float32")
    for name in ("calibration_samples", "holdout_samples", "tokens_per_sample", "seed"):
        parser.add_argument("--tpm-" + name.replace("_", "-"), type=int, default=getattr(defaults, name))
    for name in ("eta", "max_relative_edit"):
        parser.add_argument("--tpm-" + name.replace("_", "-"), type=float, default=getattr(defaults, name))
    for name in ("fallback_fp64", "strict", "shuffle_teacher_pairing", "require_holdout_nonincrease"):
        parser.add_argument("--tpm-" + name.replace("_", "-"), action="store_true")


def config_from_args(args):
    return TPMConfig(**{key[4:]: value for key, value in vars(args).items() if key.startswith("tpm_")})
