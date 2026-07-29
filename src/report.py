"""
report.py — человекочитаемый вывод: отчёт методиста, транскрипт и единый документ.

Главный результат — build_full_document(): сверху отчёт от ИИ, ниже подробная
транскрипция с диаризацией (без дублирующей шапки). Это единственный
человекочитаемый файл урока ({base}.txt).
"""

from .utils import fmt_time_short
from .naming import extract_teacher, extract_date

W = 66
SEP  = "═" * W
SEP2 = "─" * W
SKIP_SPEAKERS = {"UNKNOWN", "?", "SPEAKER_??", ""}

# Человекочитаемые названия критериев. Чтобы ДОБАВИТЬ критерий — допиши его в
# prompts/modules/03_evaluation_criteria.md и сюда (для красивого названия в отчёте).
# Отчёт рендерит любые критерии из ответа LLM — незнакомые покажутся по ключу.
CRITERION_NAMES = {
    "quality_demo":        "Качество демонстрации",
    "background":          "Задний фон / помехи",
    "homework_discussion": "Обсуждение ДЗ",
    "communication":       "Коммуникация с учениками",
    "answering_questions": "Ответы на вопросы",
    "practice":            "Практика на уроке",
    "subject_knowledge":   "Знание темы",
    "new_homework":        "Новое ДЗ",
    "group_dynamics":      "Групповая динамика",
    "structure_clarity":   "Структура и понятность",
}

_VERDICT_ICON = {
    "В норме": "✓", "Спорно": "!", "Особое внимание": "✗", "Без оценки": "·",
    "Легко": "✓", "Средне": "!", "Сложно": "✗",
}
_SEV_ICON  = {"high": "🔴", "medium": "🟠", "low": "🟡"}
_SEV_ORDER = {"high": 0, "medium": 1, "low": 2}
_TYPE_RU   = {"trial": "Пробное", "group": "Групповой", "individual": "Индивидуальный"}


def is_no_assessment(v) -> bool:
    if v is None:
        return True
    v_str = str(v).strip().lower().replace(".", "").replace('"', '').replace("'", "")
    if not v_str or v_str in ("—", "-", "none", "null", ""):
        return True
    if "без оценки" in v_str or "не оценива" in v_str or "no assessment" in v_str or "нет оценки" in v_str:
        return True
    return False
def _to_str(c):
    """LLM иногда отдаёт comment списком/словарём — приводим к строке."""
    if isinstance(c, str):
        return c
    if isinstance(c, (list, tuple)):
        return " ".join(_to_str(x) for x in c if x is not None)
    if isinstance(c, dict):
        return " ".join(f"{k}: {_to_str(v)}" for k, v in c.items())
    return "" if c is None else str(c)


def _wrap(text, indent="  "):
    """Перенос текста по ширине W с отступом. Возвращает список строк."""
    out, cur = [], ""
    for word in _to_str(text).split():
        if not cur:
            cur = indent + word
        elif len(cur) + 1 + len(word) <= W:
            cur += " " + word
        else:
            out.append(cur)
            cur = indent + word
    if cur:
        out.append(cur)
    return out


def _speaker_labeler(analytics):
    """Возвращает функцию raw_speaker → 'Спикер N' по единой карте из analytics."""
    smap = (analytics or {}).get("lesson_info", {}).get("speaker_labels") or {}

    def label(raw):
        if not raw or raw in SKIP_SPEAKERS:
            return raw or "?"
        return smap.get(raw, "Спикер ?")
    return label


# ─────────────────────────────────────────────
# Отчёт методиста
# ─────────────────────────────────────────────

def format_report(analytics, llm_result, audio_duration, file_name="", path=None):
    """Лаконичный отчёт: шапка фактов + оценка + живой комментарий + критерии + события."""
    analytics = analytics or {}
    llm = llm_result if isinstance(llm_result, dict) else {}
    lines = []

    topic   = _to_str(llm.get("lesson_topic", "")).strip()
    teacher = extract_teacher(file_name, path=path)
    date    = extract_date(file_name)

    head = "ОТЧЁТ"
    if topic:
        head += f": {topic}"
    if teacher:
        head += f" — {teacher}"
    lines += [SEP, "  " + head + (f"    {date}" if date else ""), SEP]

    # ── Шапка фактов (один раз, без дублей в транскрипте) ──
    info = analytics.get("lesson_info", {})
    ltype = llm.get("lesson_type") or info.get("format") or ""
    type_ru = _TYPE_RU.get(ltype, ltype or "—")
    sc = llm.get("students_count")
    names = llm.get("students_names") or []
    who = ""
    if sc is not None:
        who = f" ({sc} уч.{': ' + ', '.join(map(str, names)) if names else ''})"
    h = int(audio_duration // 3600)
    m = int(audio_duration % 3600 // 60)
    lines.append(f"  Тип: {type_ru}{who}  •  {h}ч {m:02d}мин")

    bal = analytics.get("balance", {})
    if bal:
        flags = []
        if llm.get("recording_complete") is False:
            flags.append("⚠ запись неполная")
        long_pauses = [p for p in analytics.get("pauses", {}).get("pauses_over_10s", [])
                       if p.get("duration", 0) >= 60]
        if long_pauses:
            flags.append(f"⚠ длинные паузы: {len(long_pauses)}")
        flag_s = "    " + "  ".join(flags) if flags else ""
        lines.append(f"  Баланс: учитель {bal.get('teacher_pct',0):.0f}% / "
                     f"ученики {bal.get('students_pct',0):.0f}%{flag_s}")

    score = llm.get("overall_score")
    if score is not None:
        lines.append(f"  Оценка: {score}/10")
    lines.append("")

    if not llm:
        lines += ["  LLM-анализ не выполнен (нет результата модели).", "", SEP]
        return "\n".join(lines)

    # ── Живой комментарий методиста (главное) ──
    comment = _to_str(llm.get("comment", "")).strip()
    if comment:
        lines.append("  КОММЕНТАРИЙ")
        lines.append(SEP2)
        lines += _wrap(comment)
        lines.append("")

    # ── Критерии: вывод + обоснование + пример из текста ──
    criteria = llm.get("criteria")
    if isinstance(criteria, dict) and criteria:
        lines.append("  КРИТЕРИИ")
        lines.append(SEP2)
        ordered = list(CRITERION_NAMES.keys()) + [k for k in criteria if k not in CRITERION_NAMES]
        for key in ordered:
            c = criteria.get(key)
            if not isinstance(c, dict):
                continue
            name = CRITERION_NAMES.get(key, key)
            verdict = _to_str(c.get("verdict", "")).strip() or "—"
            if is_no_assessment(verdict):
                continue
            icon = _VERDICT_ICON.get(verdict, "·")
            lines.append(f"  {icon} {name} — {verdict}")
            detail = _to_str(c.get("detail", "")).strip()
            if detail:
                lines += _wrap(detail, indent="      ")
            example = _to_str(c.get("example", "")).strip()
            if example:
                lines += _wrap(f"пример: {example}", indent="      ")
        lines.append("")

    # ── Критические события ──
    events = llm.get("critical_events")
    if isinstance(events, list) and events:
        lines.append("  ⚠️  КРИТИЧЕСКИЕ СОБЫТИЯ")
        lines.append(SEP2)
        for ev in sorted(events, key=lambda e: _SEV_ORDER.get((e or {}).get("severity"), 3)):
            if not isinstance(ev, dict):
                continue
            ic = _SEV_ICON.get(ev.get("severity"), "•")
            t = ev.get("time", "")
            tp = ev.get("type", "")
            head = f"  {ic} {t}".rstrip()
            if tp:
                head += f"  [{tp}]"
            lines.append(head)
            desc = _to_str(ev.get("description", "")).strip()
            if desc:
                lines += _wrap(desc, indent="      ")
        lines.append("")

    # ── Этапы урока ──
    timeline = llm.get("timeline")
    if isinstance(timeline, list) and timeline:
        lines.append("  ЭТАПЫ УРОКА")
        lines.append(SEP2)
        for st in timeline:
            if not isinstance(st, dict):
                continue
            dur = st.get("duration_min", "?")
            lines.append(f"  {st.get('start','?')}–{st.get('end','?')}  "
                         f"({dur} мин)  {_to_str(st.get('stage',''))}")
        lines.append("")

    # ── Расход токенов (мелким блоком) ──
    tok = llm.get("token_usage")
    if tok:
        lines.append("  РАСХОД ТОКЕНОВ")
        lines.append(SEP2)
        lines.append(f"  вход {tok.get('prompt','—')} · выход {tok.get('completion','—')} · "
                     f"итого {tok.get('total','—')}  (~{tok.get('cost_rub', 0):.3f} ₽)")
        lines.append("")

    lines.append(SEP)
    return "\n".join(lines)


# ─────────────────────────────────────────────
# Транскрипт (без дублирующей шапки)
# ─────────────────────────────────────────────

def format_transcript_body(segments, analytics=None):
    """Диаризованный транскрипт: реплики со спикерами и таймкодами. Шапки нет — она в отчёте."""
    label = _speaker_labeler(analytics)
    lines = [SEP, "  ТРАНСКРИПЦИЯ", SEP, ""]
    cur_speaker = None
    for seg in sorted(segments, key=lambda s: s.get("start", 0)):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        sp = seg.get("speaker", "?")
        if seg.get("_is_placeholder"):
            lines.append(f"  [{fmt_time_short(seg.get('start',0))}] {text}")
            cur_speaker = None
            continue
        if sp != cur_speaker:
            lines.append("")
            lines.append(f"── {fmt_time_short(seg.get('start',0))}  {label(sp)} " + "─" * 10)
            cur_speaker = sp
        lines += _wrap(text)
    lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# Единый документ: отчёт + транскрипт
# ─────────────────────────────────────────────

def build_full_document(analytics, llm_result, segments, audio_duration, file_name="", path=None):
    """Главный результат урока: сверху отчёт от ИИ, ниже подробная транскрипция."""
    report = format_report(analytics, llm_result, audio_duration, file_name, path=path)
    transcript = format_transcript_body(segments, analytics)
    return report + "\n\n\n" + transcript
