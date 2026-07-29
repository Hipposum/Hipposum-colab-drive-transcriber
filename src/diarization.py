"""
diarization.py — диаризация спикеров.

Основная модель: pyannote/speaker-diarization-community-1 (2025) — заметно точнее 3.1,
ловит перекрытие речи и умеет «exclusive» режим (лучше ложится на словные таймкоды).
Зовём pyannote.audio.Pipeline НАПРЯМУЮ и сами строим DataFrame для assign_word_speakers —
это развязывает нас от версии обёртки whisperx. Фолбэк — старый путь whisperx + 3.1.
"""

import time
from .utils import free_gpu


# Микро-фрагмент: только такие сегменты считаем артефактом диаризации.
# ВАЖНО: порог намеренно МАЛЕНЬКИЙ. Короткий ответ ученика ("да", "три", "правильно")
# длится ~0.5-1с — если поглощать всё <1.5с, такие реплики припишутся учителю и
# ученик «исчезнет». Поэтому трогаем только совсем крошечные фрагменты И только когда
# они зажаты между двумя репликами ОДНОГО спикера (явный признак разрыва одной фразы).
MIN_SPEAKER_SEGMENT_SEC = 0.6
MAX_MERGE_WORDS = 1     # не сливаем если в фрагменте есть реальные слова (>1 слова = речь)
MAX_MERGE_GAP_SEC = 0.25  # фрагмент сливаем только если он ВПЛОТНУЮ к соседям (середина фразы)


def _split_segments_by_speaker(segments):
    """
    Разрезает сегменты по границам смены спикера на уровне СЛОВ.

    WhisperX назначает одного спикера на весь сегмент (по большинству слов). Если
    внутри сегмента учитель закончил и сразу начал ученик — весь сегмент получает
    одного спикера, и реплики «перемешиваются». Здесь мы используем пословную
    разметку (word["speaker"]) и режем сегмент на части по сменам спикера.

    Безопасность: режем ТОЛЬКО если у всех слов есть тайминги (start/end) и есть
    ≥2 разных спикера. Иначе сегмент остаётся как есть (graceful fallback).
    """
    out = []
    n_split = 0
    for seg in segments:
        if seg.get("_is_placeholder"):
            out.append(seg)
            continue
        words = seg.get("words") or []
        distinct = {w.get("speaker") for w in words if w.get("speaker")}
        # Нечего резать: <2 спикеров среди слов или слов слишком мало
        if len(distinct) < 2 or len(words) < 2:
            out.append(seg)
            continue
        # Требуем тайминги у всех слов — иначе не режем (избегаем нулевых/кривых границ)
        if not all(("start" in w and "end" in w) for w in words):
            out.append(seg)
            continue

        # Группируем подряд идущие слова по спикеру (слово без спикера наследует текущего)
        runs = []
        cur = None
        for w in words:
            wsp = w.get("speaker") or (cur["speaker"] if cur else seg.get("speaker"))
            if cur is None or wsp != cur["speaker"]:
                cur = {"speaker": wsp, "words": [w]}
                runs.append(cur)
            else:
                cur["words"].append(w)

        if len(runs) <= 1:
            out.append(seg)
            continue

        for run in runs:
            ws = run["words"]
            text = " ".join(w.get("word", "").strip() for w in ws).strip()
            if not text:
                continue
            new_seg = dict(seg)
            new_seg["speaker"] = run["speaker"]
            new_seg["text"] = text
            new_seg["words"] = ws
            new_seg["start"] = round(min(w["start"] for w in ws), 3)
            new_seg["end"]   = round(max(w["end"] for w in ws), 3)
            out.append(new_seg)
        n_split += 1

    if n_split:
        print(f"   Пост-диаризация: разрезано {n_split} сегментов со сменой спикера")
    return out


def _merge_short_speaker_segments(segments, min_dur=MIN_SPEAKER_SEGMENT_SEC):
    """
    Пост-обработка: чинит ошибку диаризации В СЕРЕДИНЕ фразы — когда одно слово
    внутри непрерывной речи приписано другому спикеру (A → [слово] → A).

    Сливаем фрагмент обратно ТОЛЬКО при ВСЕХ условиях:
      - длительность < 0.6с и ≤1 слова (микро-фрагмент)
      - оба соседа — один и тот же спикер
      - фрагмент ВПЛОТНУЮ к соседям (зазоры < 0.25с) — признак разрыва одной фразы

    Последнее условие критично: реальный ответ ученика ("да", "нет") идёт с паузами
    вокруг (смена говорящего) — такие фрагменты НЕ трогаем, ученик не исчезает.
    """
    if not segments:
        return segments

    result = [dict(s) for s in segments]  # shallow copy

    changed = True
    passes = 0
    while changed and passes < 5:
        changed = False
        passes += 1
        for i, seg in enumerate(result):
            if seg.get("_is_placeholder"):
                continue
            dur = seg.get("end", 0) - seg.get("start", 0)
            if dur >= min_dur:
                continue
            if len(seg.get("text", "").split()) > MAX_MERGE_WORDS:
                continue
            if i == 0 or i == len(result) - 1:
                continue
            prev, nxt = result[i - 1], result[i + 1]
            if prev.get("_is_placeholder") or nxt.get("_is_placeholder"):
                continue
            prev_speaker, next_speaker = prev.get("speaker"), nxt.get("speaker")
            if not (prev_speaker and next_speaker and prev_speaker == next_speaker):
                continue
            # Зазоры до соседей — реальный ответ ученика имеет паузы, ошибка диаризации нет
            gap_before = seg.get("start", 0) - prev.get("end", 0)
            gap_after  = nxt.get("start", 0) - seg.get("end", 0)
            if gap_before > MAX_MERGE_GAP_SEC or gap_after > MAX_MERGE_GAP_SEC:
                continue
            if result[i].get("speaker") != prev_speaker:
                result[i]["speaker"] = prev_speaker
                changed = True

    n_fixed = sum(
        1 for orig, new in zip(segments, result)
        if orig.get("speaker") != new.get("speaker")
    )
    if n_fixed:
        print(f"   Пост-диаризация: слито {n_fixed} микро-фрагментов в середине фразы")

    return result


def _annotation_to_df(annotation):
    """pyannote Annotation → pandas.DataFrame [start, end, speaker] для assign_word_speakers."""
    import pandas as pd
    rows = []
    for turn, _track, speaker in annotation.itertracks(yield_label=True):
        rows.append({"start": turn.start, "end": turn.end, "speaker": speaker})
    df = pd.DataFrame(rows, columns=["start", "end", "speaker"])
    # whisperx.assign_word_speakers ожидает также колонку 'segment' (Segment-объект)
    if not df.empty:
        from pyannote.core import Segment
        df["segment"] = df.apply(lambda r: Segment(r["start"], r["end"]), axis=1)
    return df


def _diarize_direct(audio, model_name, hf_token, device, min_spk, max_spk, exclusive, batch):
    """Прямой вызов pyannote.audio.Pipeline (community-1 и др.). Возвращает DataFrame."""
    import torch
    import numpy as np
    from pyannote.audio import Pipeline

    pl = Pipeline.from_pretrained(model_name, token=hf_token)
    if pl is None:
        raise RuntimeError(f"Pipeline.from_pretrained вернул None (нет доступа к {model_name}?)")
    try:
        pl.to(torch.device(device))
    except Exception:
        pass

    # Батч инференса (если поддерживается версией) — best-effort
    try:
        if batch:
            for attr in ("segmentation_batch_size", "embedding_batch_size"):
                if hasattr(pl, attr):
                    setattr(pl, attr, int(batch))
    except Exception:
        pass

    wav = torch.from_numpy(np.ascontiguousarray(audio)).float().unsqueeze(0)
    audio_in = {"waveform": wav, "sample_rate": 16000}

    kw = {}
    if min_spk is not None:
        kw["min_speakers"] = min_spk
    if max_spk is not None:
        kw["max_speakers"] = max_spk

    out = pl(audio_in, **kw)

    # community-1 возвращает объект с .speaker_diarization / .exclusive_speaker_diarization;
    # старые пайплайны — Annotation напрямую.
    if exclusive and hasattr(out, "exclusive_speaker_diarization"):
        ann = out.exclusive_speaker_diarization
    elif hasattr(out, "speaker_diarization"):
        ann = out.speaker_diarization
    else:
        ann = out

    del pl
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return _annotation_to_df(ann)


def _diarize_whisperx(audio, model_name, hf_token, device, min_spk, max_spk, batch):
    """Фолбэк: старый путь через whisperx.DiarizationPipeline (pyannote 3.x)."""
    from whisperx.diarize import DiarizationPipeline
    dm = DiarizationPipeline(model_name=model_name, token=hf_token, device=device)
    try:
        _pa = getattr(dm, "model", None)
        if _pa is not None and batch:
            for attr in ("segmentation_batch_size", "embedding_batch_size"):
                if hasattr(_pa, attr):
                    setattr(_pa, attr, int(batch))
    except Exception:
        pass
    kw = {}
    if min_spk is not None:
        kw["min_speakers"] = min_spk
    if max_spk is not None:
        kw["max_speakers"] = max_spk
    out = dm(audio, **kw)

    del dm
    if device != "cpu":
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return out


def run_diarization(audio, aligned_result, all_segments, placeholders_list, config, device, hf_token):
    """
    Запускает диаризацию и присваивает спикеров сегментам.

    Основная модель — community-1 через прямой вызов pyannote.audio. При сбое (нет доступа
    к gated-модели / несовместимая версия) — фолбэк на whisperx + diarize_fallback_model.

    Возвращает итоговый список сегментов с полем 'speaker'.
    """
    from whisperx.diarize import assign_word_speakers

    DIARIZE_MODEL  = config["diarize_model"]
    FALLBACK_MODEL = config.get("diarize_fallback_model", "pyannote/speaker-diarization-3.1")
    MIN_SPEAKERS   = config.get("min_speakers")
    MAX_SPEAKERS   = config.get("max_speakers")
    EXCLUSIVE      = config.get("diarize_exclusive", True)
    DIARIZE_BATCH  = config.get("diarize_batch_size", 32)

    print(f"\nЭтап 5/7: Диаризация ({DIARIZE_MODEL})")
    print(f"   Спикеров: min={MIN_SPEAKERS}, max={MAX_SPEAKERS} | exclusive={EXCLUSIVE}")
    t0 = time.time()

    try:
        diarize_raw = _diarize_direct(audio, DIARIZE_MODEL, hf_token, device,
                                      MIN_SPEAKERS, MAX_SPEAKERS, EXCLUSIVE, DIARIZE_BATCH)
        print(f"   pyannote.audio (прямой вызов): {len(diarize_raw)} сегментов диаризации")
    except Exception as e:
        print(f"   ⚠ {DIARIZE_MODEL} не сработал ({str(e)[:90]})")
        print(f"   ↳ фолбэк: whisperx + {FALLBACK_MODEL}")
        diarize_raw = _diarize_whisperx(audio, FALLBACK_MODEL, hf_token, device,
                                        MIN_SPEAKERS, MAX_SPEAKERS, DIARIZE_BATCH)

    real_segs = [s for s in all_segments if not s.get("_is_placeholder")]
    assign_result = {
        "segments": real_segs,
        "word_segments": aligned_result.get("word_segments", [])
    }
    assign_result = assign_word_speakers(diarize_raw, assign_result)

    # Пост-обработка спикеров:
    # 1) разрезаем сегменты со сменой спикера (фразы перестают «перемешиваться»)
    # 2) сливаем микро-артефакты обратно (разрывы одной фразы)
    assign_result["segments"] = _split_segments_by_speaker(assign_result["segments"])
    assign_result["segments"] = _merge_short_speaker_segments(assign_result["segments"])

    segments = sorted(
        assign_result["segments"] + placeholders_list,
        key=lambda s: s["start"]
    )
    n_speakers = len(set(s.get("speaker", "?") for s in assign_result["segments"]))
    print(f"   {n_speakers} спикеров | {time.time()-t0:.0f}с")

    free_gpu()
    return segments
