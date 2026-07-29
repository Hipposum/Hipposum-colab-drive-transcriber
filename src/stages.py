"""
stages.py — обработка ОДНОГО видео по этапам с пер-этапным кэшем.

process_video() прогоняет: аудио → транскрибация → пост-обработка/Pass2 → выравнивание →
диаризация → сборка транскрипта в {base}.docx. Каждый этап кэшируется: при обрыве
повторный запуск продолжает с места остановки.
"""

import os
import time
from pathlib import Path

from .transcription import (
    load_audio, apply_noise_reduction, transcribe_pass1, retranscribe_zones, run_alignment,
)
from .problem_zones import detect_problem_zones, merge_pass2_segments
from .diarization import run_diarization
from .report import build_docx
from .utils import free_gpu
from .checkpoints import load_ckpt, save_ckpt, mark_stage, log_stage, drop_ckpt

# Версия логики пост-обработки (детект проблемных зон + слияние Pass1/Pass2).
# Бампни при любом изменении этой логики — иначе повторный запуск в той же
# Colab-сессии (work_dir/чекпоинты переживают повторный запуск ячеек, если
# рантайм не перезапускался) тихо подставит результат, посчитанный СТАРЫМ
# кодом, и правки в problem_zones.py/stages.py не будет видно вообще.
POSTPROCESS_VERSION = "8"
TRANSCRIPTION_VERSION = "6"


def process_video(vf, ctx):
    """
    Обрабатывает одно видео. Возвращает dict со статусом и артефактами.
    status: 'done' | 'skipped'.
    """
    cfg          = ctx["cfg"]
    DEVICE       = ctx["device"]
    WORK_DIR     = ctx["work_dir"]
    HF_TOKEN     = ctx["hf_token"]
    DIARIZE      = ctx["diarize"]
    WHISPER_MODEL = ctx["whisper_model"]
    silero_model = ctx["silero_model"]
    silero_utils = ctx["silero_utils"]
    USE_NR       = ctx["use_noise_reduce"]
    ENABLE_PASS2 = ctx["enable_pass2"]
    MIN_DURATION = ctx["min_duration_sec"]

    file_path = vf["local_path"]
    file_name = vf["name"]
    base_name = Path(file_name).stem
    total_start = time.time()

    def skip(reason):
        return {"status": "skipped", "reason": reason, "base_name": base_name,
                "file_name": file_name, "file_path": file_path}

    # ── Чекпоинты ──
    ckpt2 = load_ckpt(WORK_DIR, base_name, 2)
    ckpt3 = load_ckpt(WORK_DIR, base_name, 3)
    ckpt4 = load_ckpt(WORK_DIR, base_name, 4)
    ckpt5 = load_ckpt(WORK_DIR, base_name, 5)

    # Транскрибация (s2) инвалидируется при изменении параметров Whisper в settings.yaml
    if ckpt2 is not None and ckpt2.get("version") != TRANSCRIPTION_VERSION:
        print(f"   ⚠ Изменились настройки транскрибации (кэш v{ckpt2.get('version')} → "
              f"v{TRANSCRIPTION_VERSION}) — пересчитываем этапы 2–5")
        ckpt2 = ckpt3 = ckpt4 = ckpt5 = None
        drop_ckpt(WORK_DIR, base_name, 2)
        drop_ckpt(WORK_DIR, base_name, 3)
        drop_ckpt(WORK_DIR, base_name, 4)
        drop_ckpt(WORK_DIR, base_name, 5)

    # Пост-обработка (s3) и всё, что от неё зависит (s4 выравнивание, s5 диаризация),
    if ckpt3 is not None and ckpt3.get("version") != POSTPROCESS_VERSION:
        print(f"   ⚠ Логика пост-обработки изменилась (кэш v{ckpt3.get('version')} → "
              f"v{POSTPROCESS_VERSION}) — пересчитываем этапы 3–5")
        ckpt3 = ckpt4 = ckpt5 = None
        drop_ckpt(WORK_DIR, base_name, 3)
        drop_ckpt(WORK_DIR, base_name, 4)
        drop_ckpt(WORK_DIR, base_name, 5)

    found = [s for s, c in [(2,ckpt2),(3,ckpt3),(4,ckpt4),(5,ckpt5)] if c]
    if found:
        print(f"   Найден чекпоинт: этапы {found} — продолжаем с места остановки")

    need_audio = not (ckpt2 and ckpt3 and ckpt4 and ckpt5)

    # ── Этап 1: аудио ──
    if need_audio:
        log_stage(1, "загрузка аудио")
        t0 = time.time()
        try:
            audio, audio_duration = load_audio(file_path)
        except Exception as e:
            print(f"   ОШИБКА загрузки аудио: {e} — пропуск")
            mark_stage(WORK_DIR, base_name, 2, "failed", extra={"error": str(e)[:120]})
            return skip("audio_load_error")
        print(f"   {audio_duration:.0f}с ({audio_duration/60:.1f} мин) | {time.time()-t0:.1f}с")
        if audio_duration < MIN_DURATION:
            print(f"   Короткое видео (<{MIN_DURATION}с) — пропуск")
            return skip("too_short")
        if USE_NR and not ckpt2:
            log_stage(1, "шумоподавление")
            audio = apply_noise_reduction(audio, cfg.get("nr_prop_decrease", 0.4),
                                          cfg.get("nr_stationary", False))
    else:
        audio = None
        audio_duration = ckpt2.get("audio_duration", 0)
        log_stage(1, f"аудио — из кэша ({audio_duration/60:.1f} мин)")

    # ── Этап 2: транскрибация ──
    if ckpt2:
        log_stage(2, "транскрибация — из кэша")
        result = {"segments": ckpt2["segments"]}
        n_raw = ckpt2["n_raw"]
    else:
        log_stage(2, f"транскрибация Pass1 ({WHISPER_MODEL})")
        t0 = time.time()
        try:
            result, n_raw, _ = transcribe_pass1(audio, cfg, DEVICE, silero_model, silero_utils)
        except Exception as e:
            print(f"   ОШИБКА транскрибации: {e} — пропуск")
            mark_stage(WORK_DIR, base_name, 2, "failed", extra={"error": str(e)[:120]})
            free_gpu()
            return skip("transcribe_error")
        save_ckpt(WORK_DIR, base_name, 2, {"segments": result["segments"], "n_raw": n_raw,
                                           "audio_duration": round(audio_duration, 1),
                                           "version": TRANSCRIPTION_VERSION})
        mark_stage(WORK_DIR, base_name, 2, "done", seconds=time.time()-t0)

    if n_raw == 0 and not ckpt3:
        print("   0 сегментов — речь не обнаружена, пропуск")
        free_gpu()
        return skip("no_speech")

    # ── Этап 3: пост-обработка + Pass2 ──
    if ckpt3:
        log_stage(3, "пост-обработка — из кэша")
        segments_merged = ckpt3["segments_merged"]
        pass2_log = ckpt3["pass2_log"]
        n_removed = ckpt3["n_removed"]
    else:
        log_stage(3, "пост-обработка + " + ("Pass2" if ENABLE_PASS2 else "без Pass2"))
        t0 = time.time()
        clean_segs, problem_zones, removal_log = detect_problem_zones(
            result["segments"], n_raw, audio=audio, sr=16000,
            pass2_low_confidence=cfg.get("pass2_low_confidence", 0.45))
        n_removed = len(removal_log)
        pass2_log = []
        if ENABLE_PASS2 and problem_zones:
            recovered_segs, pass2_log = retranscribe_zones(audio, problem_zones, cfg, DEVICE)
            segments_merged = merge_pass2_segments(clean_segs, recovered_segs)
        else:
            placeholders = [{
                "start": round(z["start"], 3), "end": round(z["end"], 3),
                "text": f"[неразборчиво — {z['end'] - z['start']:.1f}с]",
                "speaker": "UNKNOWN", "_is_placeholder": True,
            } for z in problem_zones]
            segments_merged = merge_pass2_segments(clean_segs, placeholders)
        n_recovered = sum(1 for l in pass2_log if l.get("status") == "recovered")
        n_placeholders = sum(1 for s in segments_merged if s.get("_is_placeholder"))
        print(f"   Итого: {len(segments_merged)} сегм. | восстановлено: {n_recovered} | "
              f"неразборчиво: {n_placeholders}")
        save_ckpt(WORK_DIR, base_name, 3, {"segments_merged": segments_merged,
                                           "pass2_log": pass2_log, "n_removed": n_removed,
                                           "n_recovered": n_recovered,
                                           "version": POSTPROCESS_VERSION})
        mark_stage(WORK_DIR, base_name, 3, "done", seconds=time.time()-t0)

    # ── Этап 4: выравнивание ──
    if ckpt4:
        log_stage(4, "выравнивание — из кэша")
        all_segments = ckpt4["all_segments"]
        placeholders_list = ckpt4["placeholders"]
        aligned_result = {"word_segments": ckpt4["word_segments"]}
    else:
        log_stage(4, "выравнивание (wav2vec2)")
        t0 = time.time()
        aligned_result, all_segments, placeholders_list = run_alignment(
            audio, segments_merged, cfg, DEVICE)
        print(f"   {time.time()-t0:.0f}с")
        save_ckpt(WORK_DIR, base_name, 4, {"all_segments": all_segments,
                                           "placeholders": placeholders_list,
                                           "word_segments": aligned_result.get("word_segments", [])})
        mark_stage(WORK_DIR, base_name, 4, "done", seconds=time.time()-t0)

    # ── Этап 5: диаризация ──
    if ckpt5:
        log_stage(5, "диаризация — из кэша")
        segments = ckpt5["segments"]
    elif DIARIZE and HF_TOKEN:
        t0 = time.time()
        try:
            llm_pipeline = None
            if cfg.get("use_semantic_vad"):
                from transformers import pipeline
                print("   Загрузка микро-LLM (Qwen2.5-0.5B-Instruct)...")
                llm_pipeline = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct", device=DEVICE, torch_dtype="auto")
                
            segments = run_diarization(audio, aligned_result, all_segments,
                                       placeholders_list, cfg, DEVICE, HF_TOKEN, llm_pipeline)
            
            if llm_pipeline:
                del llm_pipeline
                free_gpu()

            save_ckpt(WORK_DIR, base_name, 5, {"segments": segments})
            mark_stage(WORK_DIR, base_name, 5, "done", seconds=time.time()-t0)
        except Exception as e:
            print(f"   ОШИБКА диаризации: {e} — продолжаем без спикеров")
            mark_stage(WORK_DIR, base_name, 5, "failed", extra={"error": str(e)[:120]})
            segments = all_segments
            free_gpu()
    else:
        log_stage(5, "диаризация пропущена")
        mark_stage(WORK_DIR, base_name, 5, "skipped")
        segments = all_segments

    # Аудио больше не нужно
    if audio is not None:
        del audio
        free_gpu()

    # ── Этап 6: сборка .docx (дешёвый, из кэша этапов) ──
    log_stage(6, "сборка .docx")
    video_dir = os.path.join(WORK_DIR, base_name)
    os.makedirs(video_dir, exist_ok=True)
    docx_path = os.path.join(video_dir, f"{base_name}.docx")
    try:
        build_docx(segments, file_name, docx_path)
    except Exception as e:
        print(f"   ОШИБКА сборки .docx: {e} — пропуск (видео не удаляется, при повторном "
              f"запуске транскрибация подхватится из кэша, останется только пересобрать файл)")
        mark_stage(WORK_DIR, base_name, 6, "failed", extra={"error": str(e)[:120]})
        return skip("docx_build_error")
    print(f"   {os.path.basename(docx_path)} ({os.path.getsize(docx_path)/1024:.0f} KB)")
    mark_stage(WORK_DIR, base_name, 6, "done")

    elapsed = time.time() - total_start
    return {"status": "done", "base_name": base_name, "file_name": file_name,
            "file_path": file_path, "saved_files": [docx_path],
            "audio_duration": audio_duration, "elapsed": elapsed,
            "n_raw": n_raw, "segments": segments}
