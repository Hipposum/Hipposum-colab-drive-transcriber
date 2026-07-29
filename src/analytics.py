"""
analytics.py — лёгкие фактические метрики урока (без баллов):
               формат урока, баланс речи учитель/ученики, длинные паузы.

Принцип: только факты, которые реально помогают методисту и подсказывают LLM.
Никаких производных оценок 0–10, слов-паразитов, вопросов, темпа — это шум.
"""

from .utils import fmt_time_short, compute_stats


# ─────────────────────────────────────────────
# Формат урока (подсказка; финальный тип ставит LLM по тексту)
# ─────────────────────────────────────────────

# Пороги «значимого» спикера — РЕАЛЬНОГО участника, а не артефакта диаризации.
# ВАЖНО: пороги АБСОЛЮТНЫЕ, не пропорция от длины урока.
# Пропорциональный порог (например 5% времени) на длинном уроке требует от ученика
# говорить минуты напролёт — иначе активные ученики ошибочно отбрасываются.
SIGNIFICANT_SPEAKER_MIN_SEC  = 12.0   # суммарно говорил хотя бы ~12 секунд
SIGNIFICANT_SPEAKER_MIN_SEGS = 2      # хотя бы в 2 разных репликах (отсекает одиночные артефакты)


def detect_lesson_format(segments):
    """
    Грубая подсказка формата: individual | group.

    Считает «значимых» спикеров — реальных участников (учитель + активные ученики).
    Это только ПОДСКАЗКА для отчёта и LLM; третий тип (trial/пробное) и финальное
    решение определяет LLM по смыслу транскрипта.

    Возвращает (format, teacher_speaker, n_significant, n_raw).
    """
    stats = compute_stats([s for s in segments if not s.get("_is_placeholder")])
    if not stats:
        return "individual", None, 0, 0

    teacher = max(stats, key=lambda s: stats[s]["duration"])

    real_labels = {sp for sp in stats if sp not in ("UNKNOWN", "?", "SPEAKER_??")}
    n_raw = len(real_labels)

    significant = {sp for sp in real_labels
                   if stats[sp]["duration"] >= SIGNIFICANT_SPEAKER_MIN_SEC
                   and stats[sp]["count"] >= SIGNIFICANT_SPEAKER_MIN_SEGS}
    n_sig = len(significant)

    fmt = "individual" if n_sig <= 2 else "group"
    return fmt, teacher, n_sig, n_raw


def build_speaker_labels(segments, min_segments=2):
    """
    Единая карта raw_speaker → 'Спикер N' для ВСЕХ файлов (транскрипт + отчёт).

    Нумерация по убыванию времени речи: «Спикер 1» = кто говорит больше всех
    (обычно учитель). Спикеры с <min_segments репликами в карту не попадают →
    форматтеры покажут их как «Спикер ?» (вероятный артефакт диаризации).
    """
    SKIP = {"UNKNOWN", "?", "SPEAKER_??", ""}
    stats = {}
    for s in segments:
        if s.get("_is_placeholder"):
            continue
        sp = s.get("speaker", "")
        if not sp or sp in SKIP:
            continue
        d = stats.setdefault(sp, {"dur": 0.0, "cnt": 0})
        d["dur"] += s.get("end", 0) - s.get("start", 0)
        d["cnt"] += 1
    named = [sp for sp, d in stats.items() if d["cnt"] >= min_segments]
    named.sort(key=lambda sp: -stats[sp]["dur"])
    return {sp: f"Спикер {i}" for i, sp in enumerate(named, 1)}


# ─────────────────────────────────────────────
# Баланс речи учитель / ученики (только проценты, без баллов)
# ─────────────────────────────────────────────

def analyze_teacher_student_balance(segments, teacher_speaker):
    """Доля времени речи: учитель vs ученики (проценты, без оценок)."""
    by_speaker = {}
    total = 0.0
    for seg in segments:
        if seg.get("_is_placeholder"):
            continue
        sp = seg.get("speaker", "?")
        dur = seg["end"] - seg["start"]
        by_speaker[sp] = by_speaker.get(sp, 0) + dur
        total += dur
    teacher_pct = by_speaker.get(teacher_speaker, 0) / total * 100 if total > 0 else 0
    return {
        "teacher_pct": round(teacher_pct, 1),
        "students_pct": round(100 - teacher_pct, 1),
        "by_speaker": {sp: round(dur / total * 100, 1) for sp, dur in by_speaker.items()} if total > 0 else {},
    }


# ─────────────────────────────────────────────
# Паузы / «мёртвый эфир»
# ─────────────────────────────────────────────

def analyze_pauses(segments):
    """Длинные паузы между репликами — реальный сигнал методисту (зависание, тишина)."""
    pauses = []
    reals = [s for s in segments if not s.get("_is_placeholder")]
    for i in range(len(reals) - 1):
        gap = reals[i + 1]["start"] - reals[i]["end"]
        if gap > 0.5:
            pauses.append({"start": round(reals[i]["end"], 1), "duration": round(gap, 1)})
    total_pause = sum(p["duration"] for p in pauses)
    longest = max(pauses, key=lambda p: p["duration"]) if pauses else None
    return {
        "total_pause_sec": round(total_pause, 1),
        "count": len(pauses),
        "longest": {"time": fmt_time_short(longest["start"]),
                    "duration": longest["duration"]} if longest else None,
        "pauses_over_10s": [{"time": fmt_time_short(p["start"]), "duration": p["duration"]}
                            for p in pauses if p["duration"] > 10],
    }


# ─────────────────────────────────────────────
# Общий запуск аналитики
# ─────────────────────────────────────────────

def run_all_analytics(segments, n_raw, n_removed, pass2_log):
    """Считает лёгкий набор фактических метрик и возвращает единый словарь."""
    print("\nАналитика...")

    lesson_fmt, teacher, n_speakers, n_speakers_raw = detect_lesson_format(segments)
    speaker_labels = build_speaker_labels(segments)
    teacher_label = speaker_labels.get(teacher, "Спикер ?")
    balance = analyze_teacher_student_balance(segments, teacher)
    pauses  = analyze_pauses(segments)

    reals = [s for s in segments if not s.get("_is_placeholder")]
    duration_min = round(reals[-1].get("end", 0) / 60, 1) if reals else 0.0

    n_recovered = sum(1 for l in pass2_log if l.get("status") == "recovered") if pass2_log else 0
    n_placeholders = sum(1 for s in segments if s.get("_is_placeholder"))

    print(f"   Формат (подсказка): {lesson_fmt} | Учитель: {teacher} ({teacher_label}) | "
          f"Спикеров: {n_speakers} (диаризация выделила {n_speakers_raw})")
    print(f"   Баланс: учитель {balance['teacher_pct']:.0f}% | ученики {balance['students_pct']:.0f}%")
    if pauses["longest"]:
        print(f"   Длинных пауз (>10с): {len(pauses['pauses_over_10s'])} | "
              f"самая длинная {pauses['longest']['duration']:.0f}с в {pauses['longest']['time']}")

    return {
        "lesson_info": {
            "format": lesson_fmt,                 # подсказка individual|group (LLM уточняет, +trial)
            "teacher_speaker": teacher,
            "teacher_label": teacher_label,       # 'Спикер N' учителя — одинаков в транскрипте и отчёте
            "speaker_labels": speaker_labels,     # единая карта raw → 'Спикер N' (по времени речи)
            "n_speakers": n_speakers,             # значимых участников (учитель + активные ученики)
            "n_speakers_raw": n_speakers_raw,     # сколько голосов выделила диаризация (может быть неточно)
        },
        "transcription_quality": {
            "total_raw": n_raw, "hallucinations_removed": n_removed,
            "pass2_recovered": n_recovered, "placeholders": n_placeholders,
            "final_segments": len(segments),
        },
        "balance": balance,
        "pauses": pauses,
        "lesson_duration_min": duration_min,
    }
