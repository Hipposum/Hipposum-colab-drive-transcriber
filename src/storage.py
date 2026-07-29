"""
storage.py — источники видео: Яндекс.Диск (сканирование, скачивание, выгрузка результатов)
             или локальная папка (смонтированный Google Drive и т.п.); распаковка ZIP,
             файл прогресса.
"""

import os
import json
import shutil
import time
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
                result.append({"local_path": ep, "remote_path": vf.get("remote_path"),
                               "name": os.path.basename(ep), "from_zip": vf["name"]})
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


def upload_progress(yandex_token, videos_folder, progress_file):
    """Сохраняет файл прогресса на Яндекс.Диск, чтобы не транскрибировать повторно."""
    if not yandex_token or not os.path.exists(progress_file):
        return
    try:
        import yadisk as yadisk_lib
        yd = yadisk_lib.YaDisk(token=yandex_token)
        remote = f"{videos_folder}/transcription_progress.json"
        if yd.exists(remote):
            yd.remove(remote)
        yd.upload(progress_file, remote)
        print(f"   Прогресс сохранён на Я.Диск: {remote}")
    except Exception as e:
        print(f"   Не удалось сохранить прогресс на Я.Диск: {e}")


# ─────────────────────────────────────────────
# Яндекс.Диск: сканирование
# ─────────────────────────────────────────────

SCAN_DEPTH = 5


def scan_yadisk_folder(yd, folder_path, extensions, max_count, skip_paths=None, depth=0):
    """
    Рекурсивно сканирует папку Яндекс.Диска и собирает файлы с нужными расширениями.
    Возвращает список dicts: {path, name, size, is_zip}.
    """
    skip_paths = set(skip_paths or [])
    videos = []

    def _scan(folder, d):
        if d > SCAN_DEPTH or len(videos) >= max_count:
            return
        try:
            items = list(yd.listdir(folder, fields=["name", "path", "type", "size"], limit=500))
        except Exception as e:
            print(f"   {folder}: {e}")
            return
        for item in items:
            if len(videos) >= max_count:
                return
            if item.type == "file":
                ext = os.path.splitext(item.name)[-1].lower()
                if ext in extensions and item.path not in skip_paths:
                    videos.append({
                        "path": item.path, "name": item.name,
                        "size": item.size or 0, "is_zip": ext == ".zip"
                    })
        for d_item in [i for i in items if i.type == "dir"]:
            if len(videos) >= max_count:
                return
            # Папку с результатами (та же, куда пишем выгрузку) не сканируем
            if d_item.name.startswith("_"):
                continue
            _scan(d_item.path, d + 1)

    _scan(folder_path, depth)
    return videos[:max_count]


def get_pending_video_list(yandex_token, videos_folder, work_dir,
                           max_videos, skip_already_transcribed,
                           video_extensions, zip_extensions):
    """
    Получает список видео с Яндекс.Диска с учётом прогресса, без скачивания файлов.
    Возвращает список dicts: {path, name, size, is_zip}.
    """
    import yadisk as yadisk_lib

    if not yandex_token:
        print("YANDEX_TOKEN не задан!")
        return []

    yd = yadisk_lib.YaDisk(token=yandex_token)
    print("Проверка токена...", end=" ")
    if not yd.check_token():
        print("Токен недействителен!")
        return []
    print("OK")

    all_accepted = video_extensions | zip_extensions
    progress_file = os.path.join(work_dir, "_progress.json")

    # Загружаем прогресс с Яндекс.Диска (переживает перезапуск Colab-сессии)
    try:
        progress_remote = f"{videos_folder}/transcription_progress.json"
        if yd.exists(progress_remote):
            yd.download(progress_remote, progress_file)
    except Exception:
        pass

    progress = load_progress(progress_file)
    skip = set(progress["transcribed"]) if skip_already_transcribed else set()

    print(f"Сканирование {videos_folder}...")
    found = scan_yadisk_folder(yd, videos_folder, all_accepted, max_videos or 10_000, skip)
    batch = found[:max_videos] if max_videos else found
    print(f"   Найдено: {len(found)} | В запуске: {len(batch)} | Пропущено: {len(skip)}")
    return batch


# ─────────────────────────────────────────────
# Яндекс.Диск: скачивание видео
# ─────────────────────────────────────────────

def download_single_item(yandex_token, video_meta, download_dir, video_extensions, zip_extensions):
    """
    Скачивает ровно один элемент (видео или ZIP) с Яндекс.Диска.
    Распаковывает архивы (если ZIP). Возвращает список локальных файлов готовых к обработке.
    """
    import yadisk as yadisk_lib

    yd = yadisk_lib.YaDisk(token=yandex_token)

    name = video_meta['name']
    local_path = os.path.join(download_dir, name)
    counter = 1
    base, ext = os.path.splitext(local_path)
    while os.path.exists(local_path):
        local_path = f"{base}_{counter}{ext}"
        counter += 1

    try:
        t0 = time.time()
        yd.download(video_meta['path'], local_path)
        actual_size = os.path.getsize(local_path) / (1024 * 1024)
        print(f"   {actual_size:.0f} MB за {time.time()-t0:.0f}с")

        raw_files = [{"local_path": local_path, "remote_path": video_meta['path'],
                      "name": name, "is_zip": video_meta.get("is_zip", False)}]
        return resolve_to_video_files(raw_files, download_dir, video_extensions, zip_extensions)
    except Exception as e:
        print(f"   Ошибка скачивания: {str(e)[:100]}")
        if os.path.exists(local_path):
            try:
                os.remove(local_path)
            except OSError:
                pass
        return []


# ─────────────────────────────────────────────
# Яндекс.Диск: выгрузка результатов
# ─────────────────────────────────────────────

def _yadisk_mkdirs(yd, path):
    """Создаёт директорию и все родительские папки на Яндекс.Диске."""
    parts = [p for p in path.strip('/').split('/') if p]
    current = ""
    for part in parts:
        current = f"{current}/{part}"
        try:
            if not yd.exists(current):
                yd.mkdir(current)
        except Exception:
            pass  # уже существует или гонка — продолжаем


def upload_results(yandex_token, output_folder, base_name, saved_files):
    """Загружает итоговые файлы урока в {output_folder}/{base_name}/ на Яндекс.Диске."""
    import yadisk as yadisk_lib
    yd = yadisk_lib.YaDisk(token=yandex_token)
    target_folder = f"{output_folder}/{base_name}"
    _yadisk_mkdirs(yd, target_folder)
    for sf in saved_files:
        remote = f"{target_folder}/{os.path.basename(sf)}"
        try:
            if yd.exists(remote):
                yd.remove(remote)
            yd.upload(sf, remote)
            print(f"   -> {remote}")
        except Exception as e:
            print(f"   Ошибка загрузки {os.path.basename(sf)}: {e}")


def upload_files_to_folder(yandex_token, target_folder, saved_files):
    """Загружает файлы непосредственно в указанную папку на Яндекс.Диске (рядом с видео)."""
    import yadisk as yadisk_lib
    yd = yadisk_lib.YaDisk(token=yandex_token)
    _yadisk_mkdirs(yd, target_folder)
    for sf in saved_files:
        remote = f"{target_folder}/{os.path.basename(sf)}"
        try:
            if yd.exists(remote):
                yd.remove(remote)
            yd.upload(sf, remote)
            print(f"   -> {remote}")
        except Exception as e:
            print(f"   Ошибка загрузки {os.path.basename(sf)}: {e}")


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
