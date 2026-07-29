"""
checkpoints.py — пер-этапный кэш (resume) + манифест статусов этапов.

При сбросе/обрыве одного этапа повторный запуск продолжает с места остановки, а не с нуля.
Манифест (_ckpt/manifest.json) фиксирует статус и тайминг каждого этапа — для наблюдаемости.
"""

import json
import os
import shutil
import time

from .utils import NpEncoder

# Человекочитаемые имена этапов для логов и манифеста.
STAGE_NAMES = {
    2: "transcribe", 3: "postprocess", 4: "align",
    5: "diarize", 6: "analytics", 7: "llm", 8: "assemble",
}


def ckpt_dir(work_dir, base_name):
    d = os.path.join(work_dir, f"_{base_name}_ckpt")
    os.makedirs(d, exist_ok=True)
    return d


def _ckpt_path(work_dir, base_name, stage):
    return os.path.join(work_dir, f"_{base_name}_ckpt", f"s{stage}.json")


def save_ckpt(work_dir, base_name, stage, data):
    """Сохраняет результат этапа и помечает его в манифесте как done."""
    ckpt_dir(work_dir, base_name)  # гарантируем существование папки _{base}_ckpt
    with open(_ckpt_path(work_dir, base_name, stage), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, cls=NpEncoder)
    mark_stage(work_dir, base_name, stage, "done")


def load_ckpt(work_dir, base_name, stage):
    path = _ckpt_path(work_dir, base_name, stage)
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"   ⚠ Чекпоинт s{stage} повреждён ({e}) — пересчитываем этап")
            try:
                os.remove(path)
            except OSError:
                pass
    return None


def clear_ckpt(work_dir, base_name):
    shutil.rmtree(os.path.join(work_dir, f"_{base_name}_ckpt"), ignore_errors=True)


# ─────────────────────────────────────────────
# Манифест статусов этапов
# ─────────────────────────────────────────────

def _manifest_path(work_dir, base_name):
    return os.path.join(ckpt_dir(work_dir, base_name), "manifest.json")


def read_manifest(work_dir, base_name):
    path = _manifest_path(work_dir, base_name)
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"stages": {}}


def mark_stage(work_dir, base_name, stage, status, seconds=None, extra=None):
    """Фиксирует статус этапа (done/failed/skipped/cached) + время в манифесте."""
    man = read_manifest(work_dir, base_name)
    rec = {"name": STAGE_NAMES.get(stage, str(stage)), "status": status,
           "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    if seconds is not None:
        rec["seconds"] = round(seconds, 1)
    if extra:
        rec.update(extra)
    man.setdefault("stages", {})[str(stage)] = rec
    try:
        with open(_manifest_path(work_dir, base_name), "w", encoding="utf-8") as f:
            json.dump(man, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def log_stage(stage, msg):
    """Единый формат лога этапа: [s5/diarize] ...."""
    print(f"[s{stage}/{STAGE_NAMES.get(stage, stage)}] {msg}")
