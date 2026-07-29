"""
utils.py — общие мелкие хелперы: JSON-энкодер numpy, GPU, формат времени, статистика спикеров.

Рендер отчёта/транскрипта → report.py; пост-обработка транскрипции → problem_zones.py.
"""

import gc
import json
import numpy as np
import torch


# ─────────────────────────────────────────────
# JSON-энкодер с поддержкой numpy типов
# ─────────────────────────────────────────────

class NpEncoder(json.JSONEncoder):
    """JSON-encoder с поддержкой numpy scalar/array/bool."""
    def default(self, obj):
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


# ─────────────────────────────────────────────
# GPU-утилиты
# ─────────────────────────────────────────────

def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def gpu_mem():
    if torch.cuda.is_available():
        u = torch.cuda.memory_allocated() / 1e9
        t = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"{u:.1f}/{t:.1f} GB"
    return "CPU"


# ─────────────────────────────────────────────
# Форматирование времени
# ─────────────────────────────────────────────

def fmt_time(s):
    h, m = int(s // 3600), int(s % 3600 // 60)
    return f"{h:02d}:{m:02d}:{s % 60:06.3f}"


def fmt_time_short(s):
    h, m, sec = int(s // 3600), int(s % 3600 // 60), int(s % 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


# ─────────────────────────────────────────────
# Статистика по спикерам
# ─────────────────────────────────────────────

def compute_stats(segments):
    stats = {}
    for seg in segments:
        sp = seg.get("speaker", "UNKNOWN")
        dur = seg["end"] - seg["start"]
        if sp not in stats:
            stats[sp] = {"count": 0, "duration": 0.0, "words": 0}
        stats[sp]["count"] += 1
        stats[sp]["duration"] += dur
        stats[sp]["words"] += len(seg["text"].split())
    return stats
