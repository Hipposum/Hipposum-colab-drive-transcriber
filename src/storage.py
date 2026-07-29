"""
storage.py — сканирование локальной папки с видео (например, смонтированный Google Drive),
             распаковка ZIP, файл прогресса.
"""

import os
import json
import shutil
import zipfile


# ─────────────────────────────────────────────
# ZIP
# ─────────────────────────────────────────────

def extract_videos_from_zip(zip_path, dest_dir, video_extensions, zip_extensions):
    """Распаковывает ZIP и возвращает список видеофайлов (рекурсивно)."""
    extracted = []
    print(f"   Распаковка {os.path.basename(zip_path)}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for member in zf.namelist():
                name = os.path.basename(member)
                if not name:
                    continue
                ext = os.path.splitext(name)[-1].lower()
                if ext in video_extensions:
                    out_path = os.path.join(dest_dir, name)
                    counter = 1
                    base, e = os.path.splitext(out_path)
                    while os.path.exists(out_path):
                        out_path = f"{base}_{counter}{e}"
                        counter += 1
                    with zf.open(member) as src, open(out_path, 'wb') as dst:
                        shutil.copyfileobj(src, dst)
                    extracted.append(out_path)
                    print(f"      {name}")
                elif ext in zip_extensions:
                    tmp_zip = os.path.join(dest_dir, name)
                    with zf.open(member) as src, open(tmp_zip, 'wb') as dst:
                        shutil.copyfileobj(src, dst)
                    extracted.extend(extract_videos_from_zip(tmp_zip, dest_dir, video_extensions, zip_extensions))
                    os.remove(tmp_zip)
    except zipfile.BadZipFile as e:
        print(f"   Повреждённый ZIP: {e}")
    return extracted


def resolve_to_video_files(raw_files, download_dir, video_extensions, zip_extensions):
    """Распаковывает ZIP-файлы и возвращает плоский список видео."""
    result = []
    for vf in raw_files:
        local_path = vf["local_path"]
        ext = os.path.splitext(local_path)[-1].lower()
        if ext in zip_extensions:
            print(f"\nZIP: {vf['name']}")
            extracted = extract_videos_from_zip(local_path, download_dir, video_extensions, zip_extensions)
            os.remove(local_path)
            for ep in extracted:
                result.append({"local_path": ep, "name": os.path.basename(ep),
                               "from_zip": vf["name"]})
            print(f"   Извлечено: {len(extracted)} видео")
        else:
            result.append(vf)
    return result


# ─────────────────────────────────────────────
# Прогресс
# ─────────────────────────────────────────────

def load_progress(progress_file):
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {"transcribed": []}


def save_progress(progress, progress_file):
    tmp = progress_file + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)
    os.replace(tmp, progress_file)


# ─────────────────────────────────────────────
# Локальная папка (Google Drive в Colab и т.п.)
# ─────────────────────────────────────────────

def get_local_video_list(local_dir, work_dir, download_dir,
                         max_videos, skip_already_transcribed,
                         video_extensions, zip_extensions):
    """
    Рекурсивно сканирует локальную папку и возвращает очередь видео.

    ZIP-архивы копируются в download_dir перед обработкой: дальше по конвейеру
    архив удаляется после распаковки, а оригинал пользователя трогать нельзя.
    """
    progress_file = os.path.join(work_dir, "_progress.json")
    progress = load_progress(progress_file)
    done = set(progress.get("transcribed", []))

    items = []
    for root, _dirs, files in os.walk(local_dir):
        # Папку результатов не сканируем
        if os.path.commonpath([os.path.abspath(root), os.path.abspath(work_dir)]) == os.path.abspath(work_dir):
            continue
        for fn in sorted(files):
            ext = os.path.splitext(fn)[-1].lower()
            if ext not in video_extensions and ext not in zip_extensions:
                continue
            full = os.path.join(root, fn)
            if skip_already_transcribed and (fn in done or full in done):
                continue
            items.append({"local_path": full, "name": fn,
                          "is_zip": ext in zip_extensions,
                          "size": os.path.getsize(full)})

    items.sort(key=lambda x: x["name"])
    if max_videos and max_videos > 0:
        items = items[:max_videos]

    for it in items:
        if it["is_zip"]:
            os.makedirs(download_dir, exist_ok=True)
            safe_copy = os.path.join(download_dir, it["name"])
            shutil.copy2(it["local_path"], safe_copy)
            it["local_path"] = safe_copy

    n_skipped = len(done)
    print(f"Папка {local_dir}: {len(items)} в очереди"
          + (f" (пропущено уже обработанных: {n_skipped})" if skip_already_transcribed and n_skipped else ""))
    return items
