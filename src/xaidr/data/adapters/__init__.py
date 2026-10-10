"""Dataset adapters: raw download -> canonical manifest rows."""
from __future__ import annotations

from xaidr.config import DataCfg
from xaidr.data.adapters.aptos2019 import Aptos2019Adapter
from xaidr.data.adapters.base import DataMissingError, DatasetAdapter
from xaidr.data.adapters.eyepacs import EyepacsAdapter
from xaidr.data.adapters.idrid import IdridAdapter
from xaidr.data.adapters.messidor2 import Messidor2Adapter

__all__ = [
    "Aptos2019Adapter", "DataMissingError", "DatasetAdapter", "EyepacsAdapter",
    "IdridAdapter", "Messidor2Adapter", "get_adapter",
]


def get_adapter(name: str, data_cfg: DataCfg) -> DatasetAdapter:
    if name == "aptos2019":
        return Aptos2019Adapter(data_cfg.detection_positive_min_grade)
    if name == "idrid":
        return IdridAdapter(data_cfg.idrid, data_cfg.detection_positive_min_grade)
    if name == "messidor2":
        return Messidor2Adapter(data_cfg.messidor2)
    if name == "eyepacs":
        return EyepacsAdapter()
    raise KeyError(f"unknown dataset '{name}'")
