"""Load and validate the YAML configuration into frozen dataclasses."""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml

KNOWN_BACKBONES = ("maxvit_tiny_tf_512", "convnext_tiny")
KNOWN_DATASETS = ("aptos2019", "idrid", "messidor2")
KNOWN_XAI_METHODS = ("gradcam_pp", "shap", "lime", "attention")
MINORITY_MODES = ("none", "oversample")


class ConfigError(ValueError):
    """Raised when the configuration is invalid; the message names the offending key."""


@dataclass(frozen=True)
class PathsCfg:
    raw: Path
    manifests: Path
    processed: Path
    splits: Path
    checkpoints: Path
    results: Path


@dataclass(frozen=True)
class ClaheCfg:
    clip_limit: float = 2.0
    tile_grid: tuple[int, int] = (8, 8)


@dataclass(frozen=True)
class IdridCfg:
    grading_images: str = "grading/images"
    grading_labels: str = "grading/labels*.csv"
    id_column: str = "Image name"
    grade_column: str = "Retinopathy grade"
    seg_images: str = "segmentation/images"
    seg_masks: str = "segmentation/masks"


@dataclass(frozen=True)
class Messidor2Cfg:
    images: str = "images"
    label_file: str = "labels.csv"
    id_column: str = "image_id"
    grade_column: str = "adjudicated_dr_grade"
    grade_map: dict = field(default_factory=dict)
    detection_map: dict = field(default_factory=dict)
    detection_definition: str = "referable"


@dataclass(frozen=True)
class DataCfg:
    image_size: int
    split_ratios: tuple[float, float, float]
    datasets: tuple[str, ...] = KNOWN_DATASETS
    crop_threshold: int = 10
    clahe: ClaheCfg = field(default_factory=ClaheCfg)
    detection_positive_min_grade: int = 1
    idrid: IdridCfg = field(default_factory=IdridCfg)
    messidor2: Messidor2Cfg = field(default_factory=Messidor2Cfg)


@dataclass(frozen=True)
class ModelCfg:
    backbone: str
    pretrained: bool = True
    head_hidden: int = 256
    dropout: float = 0.3


@dataclass(frozen=True)
class EarlyStoppingCfg:
    monitor: str = "selection_score"
    patience: int = 8


@dataclass(frozen=True)
class LossCfg:
    lambda_det: float = 0.4
    lambda_grade: float = 0.6
    grade_class_weights: str = "inverse_sqrt_freq"


@dataclass(frozen=True)
class TrainCfg:
    epochs: int
    batch_size: int
    grad_accum: int = 1
    optimizer: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.05
    scheduler: str = "cosine"
    warmup_epochs: int = 2
    amp: bool = True
    deterministic: bool = True
    num_workers: int = 4
    grad_clip_norm: float = 1.0
    early_stopping: EarlyStoppingCfg = field(default_factory=EarlyStoppingCfg)
    loss: LossCfg = field(default_factory=LossCfg)


@dataclass(frozen=True)
class MinorityCfg:
    mode: str = "oversample"
    target_grades: tuple[int, ...] = (3, 4)
    factor: int = 3


@dataclass(frozen=True)
class XAICfg:
    methods: tuple[str, ...] = KNOWN_XAI_METHODS
    gradcam_layer: str = "auto"
    fidelity_k_percent: tuple[float, ...] = (10, 5, 20)
    shap: dict = field(default_factory=dict)
    lime: dict = field(default_factory=dict)


@dataclass(frozen=True)
class EvalCfg:
    bootstrap_iterations: int = 1000
    detection_threshold: str = "youden"
    chance_monte_carlo: int = 1000


@dataclass(frozen=True)
class Config:
    paths: PathsCfg
    data: DataCfg
    model: ModelCfg
    train: TrainCfg
    seed: int = 42
    deterministic: bool = True
    minority: MinorityCfg = field(default_factory=MinorityCfg)
    xai: XAICfg = field(default_factory=XAICfg)
    eval: EvalCfg = field(default_factory=EvalCfg)

    def to_dict(self) -> dict[str, Any]:
        """Plain-python snapshot of the resolved config, safe to dump as YAML/JSON."""
        return _plain(dataclasses.asdict(self))


def _plain(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, Path):
        return obj.as_posix()
    return obj


def _coerce(value: Any, hint: Any, key: str) -> Any:
    origin = get_origin(hint)
    if dataclasses.is_dataclass(hint):
        if not isinstance(value, Mapping):
            raise ConfigError(f"'{key}' must be a mapping, got {type(value).__name__}")
        return _build(hint, value, key)
    if hint is Path:
        if not isinstance(value, (str, Path)):
            raise ConfigError(f"'{key}' must be a path string, got {value!r}")
        return Path(value)
    if origin is tuple:
        if not isinstance(value, Sequence) or isinstance(value, str):
            raise ConfigError(f"'{key}' must be a list, got {value!r}")
        args = get_args(hint)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(v, args[0], f"{key}[{i}]") for i, v in enumerate(value))
        if len(args) != len(value):
            raise ConfigError(f"'{key}' must have {len(args)} items, got {len(value)}")
        return tuple(_coerce(v, a, f"{key}[{i}]") for i, (v, a) in enumerate(zip(value, args, strict=True)))
    if hint is dict:
        if not isinstance(value, Mapping):
            raise ConfigError(f"'{key}' must be a mapping, got {value!r}")
        return dict(value)
    if hint is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"'{key}' must be true or false, got {value!r}")
        return value
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"'{key}' must be an integer, got {value!r}")
        return value
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"'{key}' must be a number, got {value!r}")
        return float(value)
    if hint is str:
        if not isinstance(value, str):
            raise ConfigError(f"'{key}' must be a string, got {value!r}")
        return value
    return value


def _build(cls: Any, raw: Mapping[str, Any], prefix: str = "") -> Any:
    hints = get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(raw) - set(fields))
    if unknown:
        raise ConfigError(f"unknown config key '{prefix}{'.' if prefix else ''}{unknown[0]}'")
    kwargs: dict[str, Any] = {}
    for name, f in fields.items():
        key = f"{prefix}.{name}" if prefix else name
        if name not in raw:
            if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
                raise ConfigError(f"missing required config key '{key}'")
            continue
        kwargs[name] = _coerce(raw[name], hints[name], key)
    return cls(**kwargs)


def _deep_merge(base: dict, extra: Mapping) -> dict:
    out = dict(base)
    for k, v in extra.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_merge(dict(out[k]), v)
        else:
            out[k] = v
    return out


def _load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"config file {path} must contain a mapping at the top level")
    base = raw.pop("base", None)
    if base is not None:
        raw = _deep_merge(_load_yaml(path.parent / base), raw)
    return raw


def apply_overrides(raw: dict, overrides: Sequence[str]) -> dict:
    """Apply ``key.subkey=value`` overrides; values are parsed as YAML scalars."""
    out = _deep_merge({}, raw)
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override '{item}' must have the form key.subkey=value")
        dotted, value = item.split("=", 1)
        node = out
        parts = dotted.strip().split(".")
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = yaml.safe_load(value)
    return out


def validate(cfg: Config) -> None:
    d = cfg.data
    if d.image_size < 224:
        raise ConfigError(f"'data.image_size' must be an int >= 224, got {d.image_size}")
    if any(r <= 0 for r in d.split_ratios) or abs(sum(d.split_ratios) - 1.0) > 1e-6:
        raise ConfigError(
            f"'data.split_ratios' must be positive and sum to 1.0, got {list(d.split_ratios)}"
        )
    bad = [x for x in d.datasets if x not in KNOWN_DATASETS]
    if bad:
        raise ConfigError(f"'data.datasets' has unknown dataset '{bad[0]}'; known: {KNOWN_DATASETS}")
    if not 1 <= d.detection_positive_min_grade <= 4:
        raise ConfigError("'data.detection_positive_min_grade' must be in 1..4")
    if cfg.model.backbone not in KNOWN_BACKBONES:
        raise ConfigError(
            f"'model.backbone' is not a known backbone: '{cfg.model.backbone}'; known: {KNOWN_BACKBONES}"
        )
    if not 0.0 <= cfg.model.dropout < 1.0:
        raise ConfigError(f"'model.dropout' must be in [0, 1), got {cfg.model.dropout}")
    m = cfg.minority
    if m.mode not in MINORITY_MODES:
        raise ConfigError(f"'minority.mode' must be one of {MINORITY_MODES}, got '{m.mode}'")
    if not set(m.target_grades) <= {0, 1, 2, 3, 4}:
        raise ConfigError(
            f"'minority.target_grades' must be a subset of {{0..4}}, got {list(m.target_grades)}"
        )
    if m.factor < 1:
        raise ConfigError(f"'minority.factor' must be >= 1, got {m.factor}")
    t = cfg.train
    for name in ("epochs", "batch_size", "grad_accum"):
        if getattr(t, name) < 1:
            raise ConfigError(f"'train.{name}' must be >= 1, got {getattr(t, name)}")
    if t.lr <= 0:
        raise ConfigError(f"'train.lr' must be > 0, got {t.lr}")
    bad = [x for x in cfg.xai.methods if x not in KNOWN_XAI_METHODS]
    if bad:
        raise ConfigError(f"'xai.methods' has unknown method '{bad[0]}'; known: {KNOWN_XAI_METHODS}")
    if not cfg.xai.fidelity_k_percent or any(not 0 < k < 100 for k in cfg.xai.fidelity_k_percent):
        raise ConfigError("'xai.fidelity_k_percent' must be a non-empty list of values in (0, 100)")
    if cfg.eval.bootstrap_iterations < 1:
        raise ConfigError("'eval.bootstrap_iterations' must be >= 1")


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> Config:
    raw = apply_overrides(_load_yaml(Path(path)), overrides)
    cfg = _build(Config, raw)
    validate(cfg)
    return cfg
