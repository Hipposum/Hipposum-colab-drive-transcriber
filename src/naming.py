"""
naming.py — извлечение преподавателя и даты из имени файла видео.

Соглашение об именовании: 'DD.MM.YY ... (Фамилия) ....mp4'
Используется отчётом (report.py) —
держим в одном месте, чтобы логика не дублировалась.
"""

import re


def normalize_teacher_name(name: str) -> str:
    """
    Приводит фамилию из родительного падежа в именительный для однозначных случаев.
    Обрабатывает только адъективные фамилии (у них нет неоднозначности):
      Стамплевской → Стамплевская
      Клиновой     → Клиновая
      Добровольского → Добровольский
      Мирного      → Мирный
    Стандартные фамилии (-ина, -ова) не трогаем — они могут быть именительным женского рода.
    Если нет паттерна — возвращает имя без изменений.
    """
    if not name:
        return name
    n = name.strip()
    # Мужские адъективные в родительном: -ского / -цкого / -ного / -дного / -зного → -ский / -цкий ...
    for suf_gen, suf_nom in [
        ("ского", "ский"),
        ("цкого", "цкий"),
        ("зкого", "зкий"),
        ("ного",  "ный"),
        ("дного", "дный"),
        ("рного", "рный"),
        ("льного", "льный"),
        ("вого",  "вый"),
        ("рого",  "рый"),
        ("кого",  "кий"),
    ]:
        if n.lower().endswith(suf_gen):
            return n[: -len(suf_gen)] + suf_nom
    # Женские адъективные в родительном: -ской / -цкой / -ной / -овой (адъективная) → -ская / -цкая ...
    for suf_gen, suf_nom in [
        ("ской",  "ская"),
        ("цкой",  "цкая"),
        ("зкой",  "зкая"),
        ("ной",   "ная"),
        ("рной",  "рная"),
        ("дной",  "дная"),
        ("льной", "льная"),
        ("вой",   "вая"),
        ("рой",   "рая"),
    ]:
        if n.lower().endswith(suf_gen):
            return n[: -len(suf_gen)] + suf_nom
    return n


def extract_teacher_from_path(path):
    """
    Извлекает имя преподавателя только из пути папки (например, 'Кабинет Антипенко' -> 'Антипенко').
    """
    if path:
        clean_path = str(path).replace("\\", "/")
        parts = clean_path.split("/")
        for part in parts:
            part_lower = part.lower()
            if "кабинет" in part_lower or "кабине" in part_lower:
                for prefix in ["кабинет", "кабине"]:
                    if part_lower.startswith(prefix):
                        extracted = part[len(prefix):].strip()
                        extracted = re.sub(r"^[._\s\-]+", "", extracted)
                        extracted = re.sub(r"[._\s\-]+$", "", extracted)
                        if extracted:
                            return extracted
    return ""


def extract_teacher(file_name, path=None):
    """
    Извлекает имя преподавателя.
    Сначала пытается вытащить из пути папки (например, 'Кабинет Антипенко' -> 'Антипенко').
    Если не найдено, пытается вытащить из скобок в имени файла: '...(Смирнова)...' -> 'Смирнова'.
    В обоих случаях применяет normalize_teacher_name() для адъективных фамилий.
    """
    t = extract_teacher_from_path(path)
    if t:
        return t

    if file_name:
        m = re.search(r"\(([^)\d][^)]*)\)", file_name)
        if m:
            return normalize_teacher_name(m.group(1).strip())

    return ""


def extract_date(file_name):
    """Дата из начала имени 'DD.MM.YY...' → 'YYYY-MM-DD'. Иначе ''."""
    if not file_name:
        return ""
    m = re.match(r"\s*(\d{1,2})[.\-](\d{1,2})[.\-](\d{2,4})", file_name)
    if not m:
        return ""
    d, mo, y = m.groups()
    y = int(y)
    if y < 100:
        y += 2000
    try:
        return f"{y:04d}-{int(mo):02d}-{int(d):02d}"
    except Exception:
        return ""
