"""
problem_zones.py — пост-обработка транскрипции: детекция галлюцинаций Whisper,
                   поиск проблемных зон для прохода 2, слияние Pass1 + Pass2.

Выделено из utils.py: один файл — одна тема (легче читать и править).
"""

import re
import numpy as np


# ─────────────────────────────────────────────
# Паттерны галлюцинаций Whisper
# ─────────────────────────────────────────────

HALLUCINATION_PATTERNS = [
    # ── Субтитры / кредиты ──
    r"[Рр]едактор субтитров",
    r"[Кк]орректор\s+[А-Я]\.",
    r"[Сс]убтитры\s+(сделан|создан|выполнен|подготовлен)",
    r"[Сс]убтитры?\s+[Пп]еревод",
    r"[Пп]одготовлено.*субтитр",
    r"[Пп]одписывайтесь на канал",
    r"[Сс]пасибо за (просмотр|подписку|внимание)",
    r"[Дд]о новых встреч",
    r"[Дд]о следующего (выпуска|видео|эфира)",
    r"[Нн]е забудьте подписаться",
    r"[Сс]тавьте лайк",
    r"DimaTorzok",                             # нотарный паттерн для русских видео
    r"[Пп]родолжение следует",
    # ── Нарративные вставки (Whisper заполняет тишину шаблоном из initial_prompt) ──
    r"[Уу]читель и учени\w* обсуждают",
    r"обсуждают задачи",
    r"решают примеры вместе",
    r"[Уу]рок продолжается",
    r"[Уу]чебный процесс",
    r"[Оо]ни (работают|решают|обсуждают|разбирают)",
    r"[Рр]ечь академическая",
    r"[Вв]озможны термины из",
    r"[Оо]нлайн.?урок с репетитором",
    r"[Уу]читель объясняет материал",
    r"[Уу]ченик задаёт вопросы",
    r"[Рр]асшифровка.*урок",
    # ── Пустые/мусорные сегменты ──
    r"^\.+$",
    r"^\s*\.{2,}\s*$",
    r"^[\s\.\,\!\?\-\_]+$",
    r"^\s*Музыка\s*$",
    r"^\s*♪",
    r"^\[музыка\]$",
    r"^\[аплодисменты\]$",
    r"^\[смех\]$",
]
_HALL_RE = [re.compile(p, re.IGNORECASE) for p in HALLUCINATION_PATTERNS]

# Порог avg_logprob ниже которого сегмент считается ненадёжным
# Whisper: 0 = уверен, -0.5 = умеренно, -1.0+ = очень неуверен / возможная галлюцинация
LOGPROB_HALLUCINATION_THRESHOLD = -1.0   # ниже этого + высокий no_speech_prob → удалить
LOGPROB_LOWCONF_THRESHOLD       = -0.8   # ниже этого → отправить в Pass 2
NO_SPEECH_HALLUCINATION_THRESHOLD = 0.6  # выше этого (при низком logprob) → удалить


def is_hallucination(text):
    t = text.strip()
    if not t:
        return True
    return any(p.search(t) for p in _HALL_RE)


def is_hallucination_segment(seg):
    """
    Проверяет сегмент по тексту + по метаданным Whisper (logprob, no_speech_prob).
    Использует сигналы уверенности модели — эффективнее чем паттерны для некоторых галлюцинаций.
    """
    text = seg.get("text", "").strip()
    if is_hallucination(text):
        return True, "pattern"

    avg_logprob    = seg.get("avg_logprob",    0.0)
    no_speech_prob = seg.get("no_speech_prob", 0.0)

    # Если модель очень неуверена И высокая вероятность тишины → галлюцинация
    if avg_logprob < LOGPROB_HALLUCINATION_THRESHOLD and no_speech_prob > NO_SPEECH_HALLUCINATION_THRESHOLD:
        return True, f"low_logprob+no_speech (lp={avg_logprob:.2f} ns={no_speech_prob:.2f})"

    return False, None


# ─────────────────────────────────────────────
# Обнаружение речи в промежутке
# ─────────────────────────────────────────────

def check_speech_in_gap(audio, gap_start, gap_end, sr=16000):
    """Проверяет наличие речи в промежутке между сегментами."""
    s = max(0, int(gap_start * sr))
    e = min(len(audio), int(gap_end * sr))
    if e <= s or e - s < sr // 4:
        return False
    chunk = audio[s:e].astype(np.float64)
    rms = float(np.sqrt(np.mean(chunk ** 2)))
    if rms < 0.003:
        return False
    active_ratio = float(np.mean(np.abs(chunk) > 0.01))
    return active_ratio > 0.05


# ─────────────────────────────────────────────
# Объединение зон
# ─────────────────────────────────────────────

def merge_zones(zones):
    """Объединяет перекрывающиеся проблемные зоны."""
    if not zones:
        return zones
    zones = sorted(zones, key=lambda z: z["start"])
    merged = [zones[0]]
    for z in zones[1:]:
        if z["start"] <= merged[-1]["end"] + 1.0:
            merged[-1]["end"] = max(merged[-1]["end"], z["end"])
            merged[-1]["reason"] += "+" + z["reason"]
        else:
            merged.append(z)
    return merged


# ─────────────────────────────────────────────
# Обнаружение проблемных зон
# ─────────────────────────────────────────────

def detect_problem_zones(segments, n_raw, audio=None, sr=16000,
                         pass2_low_confidence=0.45):
    """
    Фильтрует галлюцинации и находит зоны для прохода 2.

    Порядок проверок (от самого надёжного к менее надёжному):
    1. Паттерн-галлюцинация (текстовые паттерны)
    2. Метаданные Whisper (avg_logprob + no_speech_prob) — высокая точность
    3. Слишком короткий сегмент
    4. Скользящее окно повторений (последние 5 уникальных текстов)
    5. Низкая уверенность слов → Pass 2
    """
    clean = []
    problems = []
    removed = []

    # Скользящее окно по ВРЕМЕНИ для поиска повторений.
    # Whisper-петли (модель зацикливается и штампует одну фразу подряд) случаются
    # В ТЕЧЕНИЕ СЕКУНД. Короткие частые слова («угу», «да», «нет», «ну», «так») —
    # самые частые слова в любом уроке и совершенно законно повторяются (разными
    # людьми, в разных репликах) чаще, чем раз в 1.5 минуты — широкое окно принимало
    # обычную речь за галлюцинацию и вырезало реальные реплики. Поэтому короткий
    # текст (≤ SHORT_TEXT_LEN симв.) считаем повтором только при повторении СРАЗУ
    # (секунды — признак настоящей петли), а не где-то в пределах всего урока.
    # Длинные фразы (полное предложение) случайно совпасть не могут — для них
    # оставляем широкое окно.
    SHORT_TEXT_LEN = 15
    SHORT_REPEAT_WINDOW_SEC = 6
    REPEAT_TIME_WINDOW_SEC = 90
    recent_texts = []  # list of (text_lower, t_start)

    for i, seg in enumerate(segments):
        text = seg.get("text", "").strip()
        dur  = seg.get("end", 0) - seg.get("start", 0)
        t_start = seg.get("start", 0)

        # ── 1. Паттерн-галлюцинация + метаданные Whisper ──────────────
        hall, reason = is_hallucination_segment(seg)
        if hall:
            problems.append({"start": t_start, "end": seg.get("end", 0),
                             "reason": "hallucination", "original_text": text})
            removed.append({"reason": "hallucination", "text": text,
                            "time": f"{t_start:.1f}s", "detail": reason or "pattern"})
            continue

        # ── 2. Слишком короткий / пустой ───────────────────────────────
        # Убираем короткий текст ТОЛЬКО если модель сама сомневалась (высокий
        # no_speech_prob или низкий avg_logprob) — иначе рискуем стереть чётко
        # произнесённое короткое слово («да», «но») только из-за его длины.
        no_speech_prob = seg.get("no_speech_prob", 0.0)
        seg_avg_logprob = seg.get("avg_logprob", 0.0)
        if dur < 0.3 and len(text) <= 2 and (no_speech_prob > 0.3 or seg_avg_logprob < -0.6):
            removed.append({"reason": "too_short", "text": text,
                            "time": f"{t_start:.1f}s"})
            continue

        # ── 3. Повторяющийся текст в скользящем временно́м окне ───────
        # Убираем устаревшие записи: для коротких текстов — своё узкое окно
        # (ловит только настоящие петли декодера), для длинных — широкое.
        is_short_text = len(text.strip()) <= SHORT_TEXT_LEN
        window = SHORT_REPEAT_WINDOW_SEC if is_short_text else REPEAT_TIME_WINDOW_SEC
        recent_texts = [(t, ts, w) for t, ts, w in recent_texts if t_start - ts <= w]

        text_lower = text.lower()
        is_repeat = False
        for prev_text, prev_time, prev_window in recent_texts:
            # Совпадение засчитываем, только если попадает в окно ОБЕИХ записей —
            # короткая фраза не должна «продлевать жизнь» длинной записи и наоборот.
            if t_start - prev_time > min(window, prev_window):
                continue
            if text_lower == prev_text:
                is_repeat = True
                break
            # Near-duplicate: один текст содержит другой на ≥80% (только для длинных)
            if not is_short_text and len(text_lower) > 15 and len(prev_text) > 15:
                shorter, longer = sorted([text_lower, prev_text], key=len)
                if shorter in longer and len(shorter) / len(longer) > 0.8:
                    is_repeat = True
                    break
        if is_repeat:
            removed.append({"reason": "duplicate", "text": text,
                            "time": f"{t_start:.1f}s"})
            continue

        recent_texts.append((text_lower, t_start, window))

        # ── 4. Низкая уверенность слов → Pass 2 ──────────────────────
        words = seg.get("words", [])
        avg_logprob = seg.get("avg_logprob", 0.0)

        # Сначала проверяем avg_logprob на уровне сегмента
        if avg_logprob < LOGPROB_LOWCONF_THRESHOLD:
            problems.append({"start": t_start, "end": seg.get("end", 0),
                             "reason": "low_logprob", "original_text": text,
                             "avg_logprob": round(avg_logprob, 3)})
        elif words and len(words) > 2:
            confidences = [w.get("score", 1.0) for w in words if "score" in w]
            if confidences:
                avg_conf = sum(confidences) / len(confidences)
                if avg_conf < pass2_low_confidence:
                    problems.append({"start": t_start, "end": seg.get("end", 0),
                                     "reason": "low_confidence", "original_text": text,
                                     "avg_confidence": round(avg_conf, 3)})

        clean.append(seg)

    # ── Проверка промежутков между сегментами ─────────────────────────
    for i in range(len(clean) - 1):
        gap = clean[i+1]["start"] - clean[i]["end"]
        gap_start = clean[i]["end"]
        gap_end = clean[i+1]["start"]
        if 0.5 < gap < 3.0:
            words_end   = clean[i].get("words", [])
            words_start = clean[i+1].get("words", [])
            low_end   = words_end   and words_end[-1].get("score", 1.0) < 0.5
            low_start = words_start and words_start[0].get("score", 1.0) < 0.5
            if low_end or low_start:
                problems.append({"start": gap_start - 1.0, "end": gap_end + 1.0,
                                 "reason": "chunk_boundary", "original_text": ""})
        elif 3.0 < gap < 60.0:
            has_speech = True
            if audio is not None and len(audio) > 0:
                has_speech = check_speech_in_gap(audio, gap_start, gap_end, sr)
            if has_speech:
                problems.append({"start": gap_start, "end": gap_end,
                                 "reason": "large_gap_with_speech",
                                 "original_text": f"gap {gap:.1f}s"})
    
    # ── Проверка начала и конца файла ─────────────────────────────────
    if clean:
        # Проверяем начало
        if clean[0]["start"] > 3.0:
            gap_start = 0.0
            gap_end = clean[0]["start"]
            if gap_end < 120.0:
                has_speech = True
                if audio is not None and len(audio) > 0:
                    has_speech = check_speech_in_gap(audio, gap_start, gap_end, sr)
                if has_speech:
                    problems.append({"start": gap_start, "end": gap_end,
                                     "reason": "missing_start",
                                     "original_text": f"gap {gap_end:.1f}s"})
        
        # Проверяем конец
        if audio is not None and len(audio) > 0:
            audio_dur = len(audio) / sr
            if audio_dur - clean[-1]["end"] > 3.0:
                gap_start = clean[-1]["end"]
                gap_end = audio_dur
                if gap_end - gap_start < 120.0:
                    if check_speech_in_gap(audio, gap_start, gap_end, sr):
                        problems.append({"start": gap_start, "end": gap_end,
                                         "reason": "missing_end",
                                         "original_text": f"gap {gap_end - gap_start:.1f}s"})

    problems = merge_zones(problems)

    if removed:
        reasons = {}
        for r in removed:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        print(f"\n   Удалено {len(removed)} из {n_raw} сегментов:")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            lbl = {"hallucination": "галлюцинации", "too_short": "короткие",
                   "duplicate": "дубликаты"}.get(reason, reason)
            print(f"      {lbl}: {count}")
    if problems:
        print(f"   Проблемных зон для прохода 2: {len(problems)}")

    return clean, problems, removed


# ─────────────────────────────────────────────
# Объединение сегментов Pass1 + Pass2
# ─────────────────────────────────────────────

FALLBACK_COVER_THRESHOLD = 0.15  # Порог перекрытия по времени для сегментов без пословных таймингов


def _words_in_any_range(words, ranges, margin=0.5):
    """Индексы слов, чьи тайминги пересекаются хотя бы с одним диапазоном (start - margin, end + margin)."""
    hit = set()
    for i, w in enumerate(words):
        ws, we = w.get("start"), w.get("end")
        if ws is None or we is None:
            continue
        if any((ws < r_end + margin and we > r_start - margin) for r_start, r_end in ranges):
            hit.add(i)
    return hit


def _split_into_runs(words, excluded_idx):
    """Режет список слов на непрерывные куски, выбрасывая слова с индексом в excluded_idx."""
    runs, cur = [], []
    for i, w in enumerate(words):
        if i in excluded_idx:
            if cur:
                runs.append(cur)
                cur = []
        else:
            cur.append(w)
    if cur:
        runs.append(cur)
    return runs


def _segment_from_words(template_seg, words):
    """Собирает новый сегмент из подмножества слов исходного (сохраняя остальные поля)."""
    text = " ".join(w.get("word", "").strip() for w in words).strip()
    if not text or len(text.replace(".", "").replace(",", "").strip()) == 0:
        return None
    new_seg = dict(template_seg)
    new_seg["words"] = words
    new_seg["text"] = text
    new_seg["start"] = round(min(w["start"] for w in words), 3)
    new_seg["end"] = round(max(w["end"] for w in words), 3)
    return new_seg


def merge_pass2_segments(clean_segments, recovered_segments):
    """
    Объединяет сегменты прохода 1 (clean_segments) и прохода 2 (recovered_segments) без
    дублирования И без потери контента.

    Выбрасывает из Pass 1 слова/сегменты, перекрывающиеся с диапазонами Pass 2
    с учетом запаса (margin 0.5s), предотвращая дубликаты.
    """
    if not recovered_segments:
        return sorted(clean_segments, key=lambda s: s.get("start", 0))

    ranges = [(r["start"], r["end"]) for r in recovered_segments]
    result = []

    for c in clean_segments:
        words = c.get("words") or []
        has_valid_word_ts = words and all(w.get("start") is not None and w.get("end") is not None for w in words)
        
        if not has_valid_word_ts:
            c_dur = max(0.01, c["end"] - c["start"])
            overlap = sum(max(0.0, min(c["end"], r_end) - max(c["start"], r_start))
                          for r_start, r_end in ranges)
            if overlap / c_dur < FALLBACK_COVER_THRESHOLD:
                result.append(c)
            continue

        excluded = _words_in_any_range(words, ranges, margin=0.4)
        if not excluded:
            result.append(c)
            continue
        
        # Если исключены все слова — сегмент полностью заменен Pass 2
        if len(excluded) == len(words):
            continue

        for run in _split_into_runs(words, excluded):
            seg_from_words = _segment_from_words(c, run)
            if seg_from_words is not None:
                result.append(seg_from_words)

    result += list(recovered_segments)
    result.sort(key=lambda s: s["start"])
    return result
