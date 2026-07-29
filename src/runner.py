"""
runner.py — точка входа: python -m src.runner

Два источника видео (mode в config/settings.yaml):
  "yadisk" — сканирует папку на Яндекс.Диске, скачивает видео по одному,
             транскрибирует с диаризацией спикеров, выгружает результат рядом
             с видео (или в output_folder) и удаляет локальную копию видео;
  "local"  — сканирует локальную папку (например, смонтированный Google Drive),
             результаты остаются локально рядом с видео (work_dir).

Конфигурация — config/settings.yaml (+ переопределения через env PIPELINE_*).
Секреты — env HF_TOKEN (диаризация pyannote), YANDEX_TOKEN (режим yadisk).
"""

import gc
import os
import sys
import warnings
from pathlib import Path

import yaml

warnings.filterwarnings("ignore")

# Живой вывод этапов при запуске из subprocess (иначе stdout буферизуется блоками).
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


def load_config(config_path=None):
    """Читает config/settings.yaml; значения можно переопределить через env PIPELINE_*."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    _bool = lambda v: v.strip().lower() not in ("false", "0", "no")
    _overrides = [
        ("PIPELINE_MODE",             "mode",              str),
        ("PIPELINE_VIDEOS_FOLDER",    "videos_folder",     str),
        ("PIPELINE_OUTPUT_FOLDER",    "output_folder",     str),
        ("PIPELINE_LOCAL_DIR",        "local_dir",         str),
        ("PIPELINE_LOCAL_FILE",       "local_file",        str),
        ("PIPELINE_WORK_DIR",         "work_dir",          str),
        ("PIPELINE_DOWNLOAD_DIR",     "download_dir",      str),
        ("PIPELINE_MAX_VIDEOS",       "max_videos",        int),
        ("PIPELINE_WHISPER_MODEL",    "whisper_model",     str),
        ("PIPELINE_LANGUAGE",         "language",          str),
        ("PIPELINE_INITIAL_PROMPT",   "initial_prompt",    str),
        ("PIPELINE_MIN_SPEAKERS",     "min_speakers",      int),
        ("PIPELINE_MAX_SPEAKERS",     "max_speakers",      int),
        ("PIPELINE_MIN_DURATION_SEC", "min_duration_sec",  int),
        ("PIPELINE_SKIP_DONE",        "skip_already_transcribed", _bool),
        ("PIPELINE_DIARIZE",          "diarize",           _bool),
        ("PIPELINE_USE_SEMANTIC_VAD", "use_semantic_vad",  _bool),
    ]
    applied = []
    for env_key, cfg_key, cast in _overrides:
        val = os.environ.get(env_key)
        if val is not None:
            cfg[cfg_key] = cast(val)
            applied.append(f"  {cfg_key} = {cfg[cfg_key]}")
    if applied:
        print("⚙️  Настройки из переменных среды:")
        for line in applied:
            print(line)
    return cfg


def main(config_path=None):
    cfg = load_config(config_path)

    import torch
    import torchaudio
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]

    from .transcription import load_silero_vad
    from .storage import (get_pending_video_list, download_single_item,
                          get_local_video_list, resolve_to_video_files,
                          upload_results, upload_files_to_folder,
                          load_progress, save_progress, upload_progress)
    from .utils import compute_stats
    from .stages import process_video
    from .checkpoints import clear_ckpt

    HF_TOKEN     = os.environ.get("HF_TOKEN", "")
    YANDEX_TOKEN = os.environ.get("YANDEX_TOKEN", "")

    MODE          = cfg.get("mode", "yadisk")
    WORK_DIR      = cfg.get("work_dir", "/content/results")
    DOWNLOAD_DIR  = cfg.get("download_dir", "/content/videos_tmp")
    VIDEOS_FOLDER = cfg.get("videos_folder", "/Transcribe")
    OUTPUT_FOLDER = cfg.get("output_folder", "/Transcribe/_results")
    LOCAL_DIR     = cfg.get("local_dir", "")
    LOCAL_FILE    = cfg.get("local_file", "")
    MAX_VIDEOS    = cfg.get("max_videos", 0)
    SKIP_DONE     = cfg.get("skip_already_transcribed", True)
    DIARIZE       = cfg.get("diarize", True)
    MIN_DURATION  = cfg.get("min_duration_sec", 30)

    VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
    ZIP_EXTENSIONS   = {".zip"}
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    if torch.cuda.is_available():
        device = "cuda:0"
        print(f"GPU: {torch.cuda.get_device_name(0)} | "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = "cpu"
        print("⚠ GPU не найден — обработка на CPU будет ОЧЕНЬ медленной. "
              "В Colab: Среда выполнения → Сменить среду выполнения → T4 GPU.")

    if DIARIZE and not HF_TOKEN:
        print("HF_TOKEN отсутствует — диаризация отключена")
        DIARIZE = False

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Очередь ──
    if MODE == "yadisk":
        queue_items = get_pending_video_list(YANDEX_TOKEN, VIDEOS_FOLDER, WORK_DIR,
                                             MAX_VIDEOS, SKIP_DONE, VIDEO_EXTENSIONS, ZIP_EXTENSIONS)
    elif MODE == "local" and LOCAL_FILE and os.path.exists(LOCAL_FILE):
        ext = os.path.splitext(LOCAL_FILE)[-1].lower()
        is_zip = ext in ZIP_EXTENSIONS
        queue_items = [{"local_path": LOCAL_FILE, "name": os.path.basename(LOCAL_FILE),
                        "is_zip": is_zip, "_is_resolved": not is_zip}]
    elif MODE == "local" and LOCAL_DIR and os.path.isdir(LOCAL_DIR):
        queue_items = get_local_video_list(LOCAL_DIR, WORK_DIR, DOWNLOAD_DIR,
                                           MAX_VIDEOS, SKIP_DONE, VIDEO_EXTENSIONS, ZIP_EXTENSIONS)
        for it in queue_items:
            it["_is_resolved"] = not it.get("is_zip")
    else:
        print(f"Источник видео не задан или не найден (mode={MODE}).")
        queue_items = []

    print(f"\nИтого элементов в очереди: {len(queue_items)}")

    # ── Silero VAD (опционально) ──
    silero_model, silero_utils = None, None
    if cfg.get("use_silero_vad", False):
        silero_model, silero_utils = load_silero_vad(
            device, cfg.get("silero_threshold", 0.15),
            cfg.get("silero_min_speech_ms", 100), cfg.get("silero_min_silence_ms", 600))

    progress_file = os.path.join(WORK_DIR, "_progress.json")
    progress = load_progress(progress_file)

    ctx = {
        "cfg": cfg, "work_dir": WORK_DIR, "hf_token": HF_TOKEN,
        "device": device, "diarize": DIARIZE,
        "whisper_model": cfg["whisper_model"],
        "use_noise_reduce": cfg.get("use_noise_reduce", False),
        "enable_pass2": cfg.get("enable_pass2", True),
        "min_duration_sec": MIN_DURATION,
        "silero_model": silero_model, "silero_utils": silero_utils,
    }

    # ── Главный цикл: скачать (если нужно) → обработать → выгрузить (если yadisk) ──
    n_done = 0
    total = len(queue_items)
    processed_count = 0
    queue = list(queue_items)

    while queue:
        item = queue.pop(0)

        if not item.get("_is_resolved"):
            tag = "ZIP" if item.get("is_zip") else "видео"
            processed_count += 1
            print(f"\n{'='*65}\n[{processed_count}/{total}] {tag} {item['name']}\n{'='*65}")
            if MODE == "yadisk":
                size_mb = item.get("size", 0) / (1024 * 1024)
                print(f"   📥 Скачивание с Яндекс.Диска ({size_mb:.0f} MB)...")
                resolved = download_single_item(YANDEX_TOKEN, item, DOWNLOAD_DIR,
                                                VIDEO_EXTENSIONS, ZIP_EXTENSIONS)
            else:
                resolved = resolve_to_video_files([item], DOWNLOAD_DIR, VIDEO_EXTENSIONS, ZIP_EXTENSIONS)
            for r in reversed(resolved):
                r["_is_resolved"] = True
                queue.insert(0, r)
            continue

        vf = item
        print(f"\n{'─'*65}\nОбработка: {vf['name']}\n{'─'*65}")
        try:
            res = process_video(vf, ctx)
        except Exception as e:
            print(f"   ОШИБКА обработки видео: {e} — пропуск")
            continue

        if res["status"] == "skipped":
            print(f"   Пропущено: {res.get('reason')}")
            continue

        n_done += 1
        _summary(res, compute_stats)

        # ── Выгрузка результатов ──
        upload_ok = True
        if MODE == "yadisk" and YANDEX_TOKEN:
            upload_ok = False
            try:
                if vf.get("remote_path"):
                    # Кладём рядом с оригинальным видео на Диске
                    target_folder = os.path.dirname(vf["remote_path"])
                    upload_files_to_folder(YANDEX_TOKEN, target_folder, res["saved_files"])
                else:
                    upload_results(YANDEX_TOKEN, OUTPUT_FOLDER, res["base_name"], res["saved_files"])
                upload_ok = True
            except Exception as e:
                print(f"   ⚠ Яндекс.Диск: не удалось выгрузить результат: {str(e)[:100]}")

        progress["transcribed"].append(vf.get("remote_path") or vf["name"])
        if vf.get("from_zip"):
            progress["transcribed"].append(vf["from_zip"])
        save_progress(progress, progress_file)
        if MODE == "yadisk":
            upload_progress(YANDEX_TOKEN, VIDEOS_FOLDER, progress_file)
        clear_ckpt(WORK_DIR, res["base_name"])

        # Локальную копию видео убираем: если скачано с Диска — только когда выгрузка
        # результата подтверждена; если это временный файл из ZIP — всегда.
        if MODE == "yadisk" and upload_ok:
            try:
                os.remove(res["file_path"])
            except OSError:
                pass
        elif vf.get("from_zip"):
            try:
                os.remove(vf["local_path"])
            except OSError:
                pass

    where = f"Яндекс.Диск ({VIDEOS_FOLDER})" if MODE == "yadisk" else WORK_DIR
    print(f"\n{'='*65}\nГОТОВО: {n_done} видео | результаты: {where}\n{'='*65}")


def _summary(res, compute_stats):
    """Краткий итог по видео в консоль."""
    dur = res["audio_duration"]
    el = res["elapsed"]
    print(f"\n{'─'*55}")
    print(f"{res['file_name']}: {el:.0f}с ({el/60:.1f} мин) | RTF {el/max(dur,1):.2f}x")
    print(f"   Сегментов: {res['n_raw']} raw → {len(res['segments'])} final")
    stats = compute_stats([s for s in res["segments"] if not s.get("_is_placeholder")])
    total_speech = sum(s["duration"] for s in stats.values()) if stats else 0
    for sp, st in sorted(stats.items(), key=lambda x: -x[1]["duration"]):
        pct = st["duration"] / total_speech * 100 if total_speech > 0 else 0
        print(f"   {sp}: {st['count']} реплик, {st['duration']:.0f}с ({pct:.0f}%)")


if __name__ == "__main__":
    main()
