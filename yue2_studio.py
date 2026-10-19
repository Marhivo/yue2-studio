# -*- coding: utf-8 -*-
"""
YuE2 Studio - önálló, helyi YuE2 zenegeneráló asztali alkalmazás (PySide6)

Egyetlen fájl. Csak a YuE2-3B motort használja (audio.cpp natív szerver).
Funkciók:
  * Telepítő: motor (audiocpp_server, CUDA) + modellek letöltése, folytatható
  * Szerver indítás / leállítás / újraindítás, napló, GPU-memória kijelzés
  * Generálás: dalszöveg-szerkesztő, stílus-tag építő, COT mód, modell/VAE
    választás, LoRA (AR + NAR), minden mintavételezési paraméter, seed,
    sorozat (batch), feladatsor (queue), presetek
  * ABC-kotta: szerkesztés, kinyerés hangfájlból (SheetSage2), ×2/×3
    hosszabbítás, ének-szólam törlése, mentés/betöltés
  * Könyvtár: SQLite, lejátszó, kedvencek, megjegyzés, MP3 export,
    beállítások visszatöltése
  * Opcionális helyi Ollama segéd: dalszöveg- és stílusjavaslat

Indítás:   python yue2_studio.py
Függőség:  pip install PySide6
Opcionális: ffmpeg a PATH-on (MP3 export, nem-WAV referencia hang)
"""
from __future__ import annotations

import base64
from html import escape as html_escape
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QProcess, QProcessEnvironment, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QColor, QFont, QKeySequence, QShortcut, QTextCursor
from PySide6.QtWidgets import (
    QLayout, QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSlider, QSpinBox, QSplitter,
    QTabWidget, QTableWidget, QTableWidgetItem, QTextBrowser, QToolButton, QVBoxLayout, QWidget, QRadioButton,
)

try:
    from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
    HAS_MULTIMEDIA = True
except Exception:  # pragma: no cover
    HAS_MULTIMEDIA = False

APP_NAME = "YuE2 Studio"
APP_VERSION = "1.4"
IS_WINDOWS = sys.platform == "win32"
NO_WINDOW = 0x08000000 if IS_WINDOWS else 0

# ---------------------------------------------------------------------------
# Útvonalak
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
SETTINGS_FILE = APP_DIR / "yue2_studio_beallitasok.json"
DB_FILE = APP_DIR / "yue2_studio.db"
PRESET_DIR = APP_DIR / "presetek"
PROJECT_DIR = APP_DIR / "projektek"
SESSION_FILE = PROJECT_DIR / "_munkamenet.y2proj"
DEFAULT_PROJECT = "Névtelen projekt"

DEFAULT_SETTINGS = {
    "engine_dir": str(APP_DIR / "motor"),
    "models_dir": str(APP_DIR / "modellek"),
    "output_dir": str(APP_DIR / "kimenet"),
    "port": 8080,
    "gpu_device": "0",
    "backend": "vulkan",
    "ffmpeg": "",
    "auto_mp3": False,
    "auto_start_server": True,
    "unload_other_model": True,
    "ollama_url": "http://127.0.0.1:11434",
    "ollama_model": "",
    "show_tooltips": True,
    "ui_lang": "hu",
    "tour_offered": False,
    "last_form": {},
}

# ---------------------------------------------------------------------------
# Letölthető csomagok
# ---------------------------------------------------------------------------
ENGINE_VERSION = "v0.8.1"
ENGINE_FILES = [
    ("https://github.com/0xShug0/audio.cpp/releases/download/v0.8.1/audio-v0.8.1-bin-windows-x64-vulkan.zip",
     "c787971e025ba8ef900f0482a2cc36a049367081fe89f4841aae521a0b49de32"),
]

HF_YUE2 = "https://huggingface.co/audio-cpp/Yue2-3B-GGUF/resolve/main/"
HF_SHEET = "https://huggingface.co/audio-cpp/SheetSage2-GGUF/resolve/main/"
YUE2_DIRNAME = "Yue2-3B-GGUF"
SHEET_DIRNAME = "SheetSage2-GGUF"
SIDECARS = ["sidecars/yue2-model-config.json", "sidecars/yue2-generation-config.json",
            "sidecars/yue2-qwen.tiktoken", "sidecars/yue2-vae-config.json"]


def _yue_files(*names):
    return [(HF_YUE2 + n, f"{YUE2_DIRNAME}/{n}") for n in [*SIDECARS, *names]]


PACKAGES = {
    "main_q8": {"label": "YuE2 fő modell Q8_0 (ajánlott, ~4 GB)", "files": _yue_files("yue2-3b-q8_0.gguf"),
                "check": f"{YUE2_DIRNAME}/yue2-3b-q8_0.gguf", "default": True},
    "main_q4": {"label": "YuE2 fő modell Q4_0 (kevesebb VRAM, ~2,7 GB)", "files": _yue_files("yue2-3b-q4_0.gguf"),
                "check": f"{YUE2_DIRNAME}/yue2-3b-q4_0.gguf", "default": False},
    "main_bf16": {"label": "YuE2 fő modell BF16 (legjobb, ~7 GB, LoRA-hoz)", "files": _yue_files("yue2-3b-bf16.gguf"),
                  "check": f"{YUE2_DIRNAME}/yue2-3b-bf16.gguf", "default": False},
    "vae_f16": {"label": "VAE F16 (kötelező)", "files": _yue_files("yue2-vae-f16.gguf"),
                "check": f"{YUE2_DIRNAME}/yue2-vae-f16.gguf", "default": True},
    "vae_f32": {"label": "VAE F32 (opcionális)", "files": _yue_files("yue2-vae-f32.gguf"),
                "check": f"{YUE2_DIRNAME}/yue2-vae-f32.gguf", "default": False},
    "sheetsage": {"label": "SheetSage2 (hangból ABC-kotta, ~2,7 GB)", "files": [(HF_SHEET + "sheetsage2-orig.gguf",
                  f"{SHEET_DIRNAME}/sheetsage2-orig.gguf")], "check": f"{SHEET_DIRNAME}/sheetsage2-orig.gguf",
                  "default": True},
    "htdemucs": {"label": "HTDemucs - sávokra bontás (DAW, kicsi)", "files": [(
        "https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main/HTDemucs-GGUF/htdemucs-q8_0.gguf",
        "HTDemucs-GGUF/htdemucs-q8_0.gguf")], "check": "HTDemucs-GGUF/htdemucs-q8_0.gguf", "default": True},
    "muscriptor": {"label": "MuScriptor - hangból MIDI (DAW, ~0,4 GB)", "files": [(
        "https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main/MuScriptor-Small-GGUF/muscriptor-small-f32.gguf",
        "MuScriptor-Small-GGUF/muscriptor-small-f32.gguf")], "check": "MuScriptor-Small-GGUF/muscriptor-small-f32.gguf",
        "default": True},
    "qwen3_asr": {"label": "Qwen3-ASR 1.7B - dalszöveg-felismerés (Átdolgozás, ~2 GB)", "files": [(
        "https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main/Qwen3-ASR-1.7B-GGUF/qwen3-asr-1.7b-q8_0.gguf",
        "Qwen3-ASR-1.7B-GGUF/qwen3-asr-1.7b-q8_0.gguf")], "check": "Qwen3-ASR-1.7B-GGUF/qwen3-asr-1.7b-q8_0.gguf",
        "default": True},
    "lora_inst": {"label": "Instrumentális AR LoRA (CC BY-NC)", "files": [(
        "https://huggingface.co/Mothersuperior/YuE2-instrumental-cot-full-loras/resolve/main/ar_lora_inst_v3abc.safetensors",
        f"{YUE2_DIRNAME}/ar_lora_inst_v3abc.safetensors")], "check": f"{YUE2_DIRNAME}/ar_lora_inst_v3abc.safetensors",
        "default": False},
    "lora_nar": {"label": "Valósághű hangzás NAR LoRA v4 (CC BY-NC)", "files": [(
        "https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4/resolve/main/nar_lora_joint_v4.safetensors",
        f"{YUE2_DIRNAME}/nar_lora_joint_v4.safetensors")], "check": f"{YUE2_DIRNAME}/nar_lora_joint_v4.safetensors",
        "default": False},
}

# ---------------------------------------------------------------------------
# Paraméter definíciók (név, felirat, típus, min, max, lépés, alapérték, leírás)
# ---------------------------------------------------------------------------
SAMPLING = [
    ("temperature", "Temperature", float, 0.0, 5.0, 0.05, "Véletlenszerűség. Magasabb = bátrabb, változatosabb."),
    ("top_p", "Top-p", float, 0.0, 1.0, 0.01, "Nucleus sampling. Alacsonyabb = óvatosabb választás."),
    ("top_k", "Top-k", int, 1, 2000, 1, "Csak a k legvalószínűbb tokenből választ."),
    ("repetition_penalty", "Ismétlés büntetés", float, 0.001, 5.0, 0.005, "Ismétlődés elleni büntetés."),
    ("penalty_window", "Büntetés ablak", int, 1, 20000, 1, "Hány előző tokenre figyel a büntetés."),
    ("min_tokens", "Min. tokenek", int, 0, 30000, 10, "Ennyi token előtt nem állhat le."),
    ("max_tokens", "Max. tokenek", int, 1, 30000, 100, "Felső korlát - a szám maximális hosszát is adja."),
]
SEM_DEFAULTS = {"temperature": 1.0, "top_p": 0.95, "top_k": 100, "repetition_penalty": 1.2,
                "penalty_window": 50, "min_tokens": 200, "max_tokens": 9000}
ABC_DEFAULTS = {"temperature": 0.7, "top_p": 0.9, "top_k": 30, "repetition_penalty": 1.005,
                "penalty_window": 100, "min_tokens": 32, "max_tokens": 4096}

ARENAS = [
    ("model_weight_context_mb", "Fő modell súly arena (MB)", 6144),
    ("vae_weight_context_mb", "VAE súly arena (MB)", 1536),
    ("ar_prefill_graph_arena_mb", "AR prefill gráf (MB)", 4096),
    ("ar_decode_graph_arena_mb", "AR decode gráf (MB)", 1536),
    ("nar_graph_arena_mb", "NAR gráf (MB)", 6144),
    ("vae_graph_arena_mb", "VAE gráf (MB)", 1536),
]
WEIGHT_TYPES = ["native", "f32", "f16", "bf16", "q8_0", "q4_0", "q4_k"]

TAG_BANK = {
    "Nyelv": ["English", "Hungarian", "German", "Spanish", "Japanese", "Korean", "Chinese"],
    "Műfaj": ["pop", "rock", "indie pop", "indie rock", "folk", "blues", "jazz", "soul", "R&B", "hip hop",
              "electronic", "synthwave", "ambient", "metal", "punk", "country", "reggae", "funk", "disco",
              "house", "techno", "classical", "orchestral", "cinematic", "lo-fi", "chanson", "gospel",
              "latin", "bossa nova", "delta blues", "folk rock", "dream pop", "trip hop", "musical"],
    "Hangulat": ["happy", "sad", "melancholic", "energetic", "dreamy", "dark", "romantic", "epic", "calm",
                 "nostalgic", "aggressive", "uplifting", "mysterious", "smoky", "late-night", "bittersweet"],
    "Hangszer": ["acoustic guitar", "electric guitar", "slide guitar", "piano", "rhodes", "synth", "strings",
                 "violin", "cello", "saxophone", "trumpet", "harmonica", "accordion", "bass guitar",
                 "upright bass", "drums", "soft drums", "808", "organ", "flute", "cimbalom", "brass section"],
    "Ének": ["male vocal", "female vocal", "warm lead vocal", "raspy vocal", "soulful vocal", "choir",
             "duet", "whispered vocal", "powerful vocal", "falsetto", "backing vocals"],
    "Tempó": ["slow tempo", "mid tempo", "fast tempo", "ballad", "groovy", "70 bpm", "90 bpm",
              "110 bpm", "120 bpm", "140 bpm", "12-bar"],
    "Produkció": ["polished mix", "polished demo mix", "lo-fi recording", "live recording", "vintage",
                  "warm analog", "studio quality", "reverb-heavy", "minimal", "wide stereo"],
}

LYRICS_EXAMPLE = """[Verse]
Soft morning light is touching the window.
I hear the city waking below.
[Chorus]
Stay with the rhythm, let it carry us home.
Sing with the sunrise, we are never alone.
[Verse]
Footsteps on the river, the bridges are glowing.
Every little shadow is learning to go.
[Chorus]
Stay with the rhythm, let it carry us home.
Sing with the sunrise, we are never alone.
[Outro]
...we are never alone."""

STYLE_EXAMPLE = "English, indie pop, bright acoustic guitar, soft drums, warm lead vocal, polished demo mix"


# ---------------------------------------------------------------------------
# Segédfüggvények
# ---------------------------------------------------------------------------
def find_remiqora_yue2() -> Path | None:
    """A Remiqora asztali app már telepített YuE2 motorja (szerver + modellek)."""
    cands = []
    la = os.environ.get("LOCALAPPDATA")
    if la:
        cands.append(Path(la) / "Remiqora" / "engines" / "YuE2")
    cands.append(Path.home() / "AppData" / "Local" / "Remiqora" / "engines" / "YuE2")
    cands.append(Path(r"E:\AI\YuE2-3B"))
    for c in cands:
        if (c / "models" / YUE2_DIRNAME).is_dir() and any(c.rglob("audiocpp_server*")):
            return c
    return None


def load_settings() -> dict:
    s = dict(DEFAULT_SETTINGS)
    first_run = not SETTINGS_FILE.is_file()
    try:
        if not first_run:
            s.update(json.loads(SETTINGS_FILE.read_text(encoding="utf-8")))
    except Exception:
        pass
    if first_run:
        rq = find_remiqora_yue2()
        if rq:
            s["engine_dir"] = str(rq)
            s["models_dir"] = str(rq / "models")
            save_settings(s)
    return s


def save_settings(s: dict) -> None:
    try:
        SETTINGS_FILE.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def fwd(p) -> str:
    """A natív szerver JSON-parsere a Windows backslash-t rosszul kezeli."""
    return str(p).replace("\\", "/")


def find_ffmpeg(cfg: dict) -> str | None:
    c = (cfg.get("ffmpeg") or "").strip()
    if c and Path(c).is_file():
        return c
    for name in ("ffmpeg", "ffmpeg.exe"):
        f = shutil.which(name)
        if f:
            return f
    local = APP_DIR / "ffmpeg.exe"
    return str(local) if local.is_file() else None


def safe_name(s: str, maxlen: int = 50) -> str:
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", s).strip().replace(" ", "_")
    return (s[:maxlen] or "yue2").rstrip("._")


def open_in_explorer(path: Path) -> None:
    try:
        if IS_WINDOWS:
            if path.is_file():
                subprocess.Popen(["explorer", "/select,", str(path)])
            else:
                os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(path if path.is_dir() else path.parent)])
    except Exception:
        pass


def abc_mute_vocal(abc: str) -> str | None:
    lines = abc.replace("\r\n", "\n").split("\n")
    out, removed, skipping = [], 0, False
    for line in lines:
        if re.match(r"^V:\s*\S+\s*$", line) and "=" not in line:
            skipping = bool(re.match(r"^V:\s*[Vv]ocal\s*$", line))
            if skipping:
                removed += 1
                continue
        if not skipping:
            out.append(line)
    if not removed:
        return None
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip() + "\n"


def abc_extend(abc: str, times: int) -> str | None:
    if times < 2:
        return abc
    lines = abc.replace("\r\n", "\n").split("\n")
    starts = [i for i, l in enumerate(lines) if re.match(r"^V:\s*\S+\s*$", l) and "=" not in l]
    if len(starts) < 2:
        return None
    header = lines[:starts[0]]
    blocks = []
    for k, s in enumerate(starts):
        b = lines[s:(starts[k + 1] if k + 1 < len(starts) else len(lines))]
        while b and not b[-1].strip():
            b.pop()
        blocks.append(b)
    out = list(header)
    for _ in range(times):
        for b in blocks:
            out += [""] + b
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip() + "\n"


def abc_stats(abc: str) -> str:
    if not abc.strip():
        return "Üres"
    voices = sorted(set(re.findall(r"^V:\s*(\S+)\s*$", abc, re.M)))
    bars = abc.count("|")
    q = re.search(r"^Q:\s*(.+)$", abc, re.M)
    k = re.search(r"^K:\s*(.+)$", abc, re.M)
    m = re.search(r"^M:\s*(.+)$", abc, re.M)
    parts = [f"{len(abc.splitlines())} sor", f"~{bars} ütemvonal"]
    if voices:
        parts.append("szólamok: " + ", ".join(voices))
    if m:
        parts.append("ütem: " + m.group(1).strip())
    if q:
        parts.append("tempó: " + q.group(1).strip())
    if k:
        parts.append("hangnem: " + k.group(1).strip())
    return " · ".join(parts)


# ---------------------------------------------------------------------------
# Adatbázis
# ---------------------------------------------------------------------------
class TrackDB:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS tracks(
                id INTEGER PRIMARY KEY AUTOINCREMENT, created TEXT, title TEXT, style TEXT, lyrics TEXT,
                seed INTEGER, cot TEXT, model TEXT, params TEXT, wav_path TEXT, abc_path TEXT,
                duration REAL, gen_secs REAL, fav INTEGER DEFAULT 0, note TEXT DEFAULT '')""")
            cols = [r[1] for r in c.execute("PRAGMA table_info(tracks)")]
            if "project" not in cols:
                c.execute("ALTER TABLE tracks ADD COLUMN project TEXT DEFAULT ''")

    def _conn(self):
        return sqlite3.connect(self.path, timeout=10)

    def add(self, **kw) -> int:
        cols = ",".join(kw)
        q = ",".join("?" * len(kw))
        with self.lock, self._conn() as c:
            cur = c.execute(f"INSERT INTO tracks({cols}) VALUES({q})", list(kw.values()))
            return cur.lastrowid

    def all(self, search: str = "", fav_only: bool = False, project: str | None = None) -> list[dict]:
        """project=None: minden dal; különben csak az adott projekt dalai ('' = projekt nélküliek)."""
        sql = "SELECT * FROM tracks WHERE 1=1"
        args: list = []
        if project is not None:
            sql += " AND COALESCE(project,'')=?"
            args.append(project)
        if search:
            sql += " AND (title LIKE ? OR style LIKE ? OR lyrics LIKE ? OR note LIKE ?)"
            args += [f"%{search}%"] * 4
        if fav_only:
            sql += " AND fav=1"
        sql += " ORDER BY id DESC"
        with self.lock, self._conn() as c:
            c.row_factory = sqlite3.Row
            return [dict(r) for r in c.execute(sql, args).fetchall()]

    def get(self, tid: int) -> dict | None:
        with self.lock, self._conn() as c:
            c.row_factory = sqlite3.Row
            r = c.execute("SELECT * FROM tracks WHERE id=?", (tid,)).fetchone()
            return dict(r) if r else None

    def update(self, tid: int, **kw) -> None:
        sets = ",".join(f"{k}=?" for k in kw)
        with self.lock, self._conn() as c:
            c.execute(f"UPDATE tracks SET {sets} WHERE id=?", [*kw.values(), tid])

    def delete(self, tid: int) -> None:
        with self.lock, self._conn() as c:
            c.execute("DELETE FROM tracks WHERE id=?", (tid,))

    def projects(self) -> list[str]:
        with self.lock, self._conn() as c:
            return [r[0] for r in c.execute("SELECT DISTINCT COALESCE(project,'') FROM tracks ORDER BY 1")]

    def retag(self, old: str, new: str) -> None:
        with self.lock, self._conn() as c:
            c.execute("UPDATE tracks SET project=? WHERE COALESCE(project,'')=?", (new, old))

    def by_wav(self, wav: str) -> dict | None:
        with self.lock, self._conn() as c:
            c.row_factory = sqlite3.Row
            r = c.execute("SELECT * FROM tracks WHERE wav_path=?", (wav,)).fetchone()
            return dict(r) if r else None


# ---------------------------------------------------------------------------
# Natív szerver API kliens (audiocpp_server)
# ---------------------------------------------------------------------------
class ApiError(Exception):
    pass


class Yue2Api:
    def __init__(self, port: int):
        self.base = f"http://127.0.0.1:{port}"
        self.loaded_sessions: dict[str, dict] = {}

    def _req(self, method: str, path: str, body=None, timeout: float | None = 30.0):
        data = None
        headers = {}
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")
            try:
                j = json.loads(msg)
                msg = j.get("error", {}).get("message") if isinstance(j.get("error"), dict) else (j.get("error") or msg)
            except Exception:
                pass
            raise ApiError(f"HTTP {e.code}: {str(msg)[:800]}") from None
        except urllib.error.URLError as e:
            raise ApiError(f"A szerver nem érhető el: {e.reason}") from None
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {"raw": raw.decode("utf-8", "replace")}

    def health(self) -> bool:
        try:
            self._req("GET", "/health", timeout=1.5)
            return True
        except Exception:
            return False

    def models(self) -> list[dict]:
        return self._req("GET", "/v1/models").get("data", []) or []

    def loaded_ids(self) -> set[str]:
        try:
            return {m.get("id") for m in self.models() if m.get("loaded")}
        except Exception:
            return set()

    def load(self, mid: str, path: str, family: str, task: str, session_options: dict) -> None:
        self._req("POST", "/v1/models/load", {
            "id": mid, "path": fwd(path), "family": family, "task": task, "mode": "offline",
            "load_options": {}, "session_options": session_options}, timeout=900)
        self.loaded_sessions[mid] = dict(session_options)

    def unload(self, mid: str) -> None:
        try:
            self._req("POST", "/v1/models/unload", {"id": mid}, timeout=120)
        except ApiError:
            pass
        self.loaded_sessions.pop(mid, None)

    def ensure(self, mid, path, family, task, session_options, unload_others=True, log=None) -> None:
        loaded = self.loaded_ids()
        if unload_others:
            for other in loaded - {mid}:
                if log:
                    log(f"VRAM felszabadítása: {other} kiürítése")
                self.unload(other)
        if mid in loaded and self.loaded_sessions.get(mid) == session_options:
            return
        if mid in loaded:
            if log:
                log("Más modell-beállítás - újratöltés…")
            self.unload(mid)
        if log:
            log(f"{mid} betöltése a GPU-ra… (első alkalommal pár perc)")
        self.load(mid, path, family, task, session_options)

    def run(self, model: str, request: dict) -> dict:
        return self._req("POST", "/v1/tasks/run", {"model": model, "request": request}, timeout=None)


def result_audio(res: dict) -> bytes | None:
    a = res.get("audio")
    if isinstance(a, str) and a:
        return base64.b64decode(a)
    for art in res.get("artifacts") or []:
        meta = art.get("meta") or {}
        fmt = str(meta.get("format") or meta.get("extension") or art.get("id") or "")
        if re.search(r"wav|audio", fmt, re.I) and isinstance(art.get("payload"), str):
            return base64.b64decode(art["payload"])
    return None


def result_abc(res: dict) -> str:
    for art in res.get("artifacts") or []:
        meta = art.get("meta") or {}
        fmt = str(meta.get("format") or meta.get("extension") or meta.get("mime") or art.get("id") or "")
        if re.search(r"abc|score", fmt, re.I) and isinstance(art.get("payload"), str):
            try:
                return base64.b64decode(art["payload"]).decode("utf-8")
            except Exception:
                return art["payload"]
    t = res.get("text")
    return t if isinstance(t, str) and "X:" in t else ""


# ---------------------------------------------------------------------------
# Letöltő szál
# ---------------------------------------------------------------------------
class Downloader(QThread):
    progress = Signal(str, int, int)      # felirat, kész bájt, összes bájt
    overall = Signal(int, int)            # fájl index, fájlok száma
    message = Signal(str)
    done = Signal(bool, str)

    def __init__(self, jobs: list[tuple[str, Path, str | None]], extract_to: Path | None = None):
        super().__init__()
        self.jobs = jobs
        self.extract_to = extract_to
        self._stop = False

    def stop(self):
        self._stop = True

    def _download(self, url: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        have = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": f"{APP_NAME}/{APP_VERSION}"}
        if have:
            headers["Range"] = f"bytes={have}-"
        req = urllib.request.Request(url, headers=headers)
        try:
            r = urllib.request.urlopen(req, timeout=60)
        except urllib.error.HTTPError as e:
            if e.code == 416 and have:
                part.replace(dest)
                return
            raise
        with r:
            status = r.status
            length = int(r.headers.get("Content-Length") or 0)
            if have and status == 200:
                have = 0
                mode = "wb"
            else:
                mode = "ab" if have else "wb"
            total = have + length if length else 0
            with open(part, mode) as f:
                done = have
                last = 0.0
                while True:
                    if self._stop:
                        raise InterruptedError("Megszakítva")
                    chunk = r.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    now = time.time()
                    if now - last > 0.25:
                        self.progress.emit(dest.name, done, total)
                        last = now
        self.progress.emit(dest.name, total or done, total or done)
        part.replace(dest)

    def run(self):
        try:
            n = len(self.jobs)
            for i, (url, dest, sha) in enumerate(self.jobs):
                self.overall.emit(i, n)
                if dest.exists():
                    self.message.emit(f"✓ megvan: {dest.name}")
                    continue
                self.message.emit(f"↓ {dest.name}")
                self._download(url, dest)
                if sha:
                    self.message.emit(f"SHA256 ellenőrzés: {dest.name}")
                    h = hashlib.sha256()
                    with open(dest, "rb") as f:
                        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
                            h.update(chunk)
                    if h.hexdigest().lower() != sha.lower():
                        dest.unlink(missing_ok=True)
                        raise RuntimeError(f"Hibás ellenőrzőösszeg: {dest.name} (töröltem, próbáld újra)")
                if self.extract_to and dest.suffix.lower() == ".zip":
                    self.message.emit(f"Kicsomagolás: {dest.name}")
                    with zipfile.ZipFile(dest) as z:
                        z.extractall(self.extract_to)
            self.overall.emit(n, n)
            self.done.emit(True, "Kész.")
        except InterruptedError:
            self.done.emit(False, "Letöltés megszakítva (később folytatható).")
        except Exception as e:
            self.done.emit(False, f"Hiba: {e}")


# ---------------------------------------------------------------------------
# Generáló szál (feladatsor)
# ---------------------------------------------------------------------------
class GenWorker(QThread):
    log = Signal(str)
    job_started = Signal(int)
    job_finished = Signal(int, dict)
    job_failed = Signal(int, str)
    abc_extracted = Signal(str)
    abc_failed = Signal(str)
    sep_done = Signal(dict, dict)
    midi_done = Signal(str)
    task_failed = Signal(str, str)
    asr_done = Signal(str, str)
    asr_timed = Signal(list, str, float)
    remake_abc = Signal(str)
    idle = Signal()

    def __init__(self, get_cfg, db: TrackDB):
        super().__init__()
        self.get_cfg = get_cfg
        self.db = db
        self.jobs: list[dict] = []
        self.cond = threading.Condition()
        self._quit = False
        self.paused = False

    def add(self, job: dict):
        with self.cond:
            self.jobs.append(job)
            self.cond.notify()

    def clear_pending(self):
        with self.cond:
            self.jobs = [j for j in self.jobs if j.get("_running")]

    def remove(self, jid: int):
        with self.cond:
            self.jobs = [j for j in self.jobs if j["id"] != jid or j.get("_running")]

    def quit_worker(self):
        with self.cond:
            self._quit = True
            self.cond.notify()

    def resume(self):
        with self.cond:
            self.paused = False
            self.cond.notify()

    def run(self):
        while True:
            with self.cond:
                while (not self.jobs or self.paused) and not self._quit:
                    self.cond.wait()
                if self._quit:
                    return
                job = self.jobs[0]
                job["_running"] = True
            try:
                if job["kind"] == "sheetsage":
                    self._do_sheetsage(job)
                elif job["kind"] == "sep":
                    self._do_sep(job)
                elif job["kind"] == "midi":
                    self._do_midi(job)
                elif job["kind"] == "asr":
                    self._do_asr(job)
                else:
                    self.job_started.emit(job["id"])
                    info = self._do_generate(job)
                    self.job_finished.emit(job["id"], info)
            except Exception as e:
                msg = str(e) or e.__class__.__name__
                self.log.emit("HIBA: " + msg)
                if job.get("fail_kind"):
                    self.task_failed.emit(job["fail_kind"], msg)
                elif job["kind"] == "asr":
                    self.task_failed.emit("asr", msg)
                elif job["kind"] == "sheetsage":
                    self.abc_failed.emit(msg)
                elif job["kind"] in ("sep", "midi"):
                    self.task_failed.emit(job["kind"], msg)
                else:
                    self.job_failed.emit(job["id"], msg)
            finally:
                with self.cond:
                    if self.jobs and self.jobs[0] is job:
                        self.jobs.pop(0)
                    if not self.jobs:
                        self.idle.emit()

    def _api(self) -> Yue2Api:
        cfg = self.get_cfg()
        if not hasattr(self, "_apiobj") or self._apiobj.base.endswith(str(cfg["port"])) is False:
            self._apiobj = Yue2Api(int(cfg["port"]))
        return self._apiobj

    def reset_sessions(self):
        if hasattr(self, "_apiobj"):
            self._apiobj.loaded_sessions.clear()

    def _do_generate(self, job: dict) -> dict:
        cfg = self.get_cfg()
        api = self._api()
        if not api.health():
            raise ApiError("A YuE2 szerver nem fut. Indítsd el a 'Telepítés & Szerver' fülön.")
        model_dir = Path(cfg["models_dir"]) / YUE2_DIRNAME
        api.ensure("yue2", str(model_dir), "yue2", "gen", job["session"], cfg.get("unload_other_model", True),
                   self.log.emit)
        req = {"lyrics": job["lyrics"], "seed": int(job["seed"]), "options": job["options"]}
        self.log.emit(f"Generálás… seed={job['seed']}  cot={job['options'].get('cot')}")
        t0 = time.time()
        res = api.run("yue2", req)
        secs = time.time() - t0
        audio = result_audio(res)
        if not audio:
            raise ApiError("A szerver nem adott vissza hangot. Válasz kulcsok: " + ", ".join(res.keys()))
        out_dir = Path(cfg["output_dir"]) / dt.date.today().isoformat()
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%H%M%S")
        base = f"{stamp}_{job['seed']}_{safe_name(job['title'] or job['options'].get('style', ''))}"
        wav = out_dir / f"{base}.wav"
        wav.write_bytes(audio)
        abc_text = result_abc(res) or job["options"].get("abc", "")
        abc_path = ""
        if abc_text:
            ap = out_dir / f"{base}.abc"
            ap.write_text(abc_text, encoding="utf-8")
            abc_path = str(ap)
        timing = res.get("timing") or {}
        dur = (timing.get("audio_duration_ms") or 0) / 1000.0
        if not dur:
            try:
                import wave
                with wave.open(str(wav)) as w:
                    dur = w.getnframes() / float(w.getframerate())
            except Exception:
                dur = 0.0
        if cfg.get("auto_mp3"):
            ff = find_ffmpeg(cfg)
            if ff:
                subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav), "-b:a", "320k",
                                str(wav.with_suffix(".mp3"))], creationflags=NO_WINDOW)
        params = {"form": job.get("form", {}), "options": job["options"], "session": job["session"]}
        tid = self.db.add(created=dt.datetime.now().isoformat(timespec="seconds"), title=job["title"] or base,
                          style=job["options"].get("style", ""), lyrics=job["lyrics"], seed=int(job["seed"]),
                          cot=job["options"].get("cot", ""), model=job["session"].get("yue2.model_gguf", ""),
                          params=json.dumps(params, ensure_ascii=False), wav_path=str(wav), abc_path=abc_path,
                          duration=dur, gen_secs=secs, project=job.get("project", ""))
        self.log.emit(f"✓ Kész: {wav.name}  ({dur:.0f} mp hang, {secs:.0f} mp alatt)")
        return {"track_id": tid, "wav": str(wav), "abc": abc_text, "secs": secs, "dur": dur}

    def _do_sep(self, job: dict):
        cfg = self.get_cfg()
        api = self._api()
        if not api.health():
            raise ApiError("A YuE2 szerver nem fut.")
        mroot = Path(cfg["models_dir"])
        gguf = next((mroot / "HTDemucs-GGUF" / n for n in ("htdemucs-q8_0.gguf", "htdemucs-f16.gguf")
                     if (mroot / "HTDemucs-GGUF" / n).is_file()), None)
        if gguf is None:
            raise ApiError("A HTDemucs modell nincs letöltve. Telepítés & Szerver fül → pipáld be a "
                           "'HTDemucs - sávokra bontás' csomagot → Kijelöltek letöltése.")
        src = Path(job["audio"])
        self.log.emit(f"Sávokra bontás: {src.name} - átalakítás 44,1 kHz WAV-ba…")
        data = read_audio(str(src))
        tmp = DAW_DIR / "temp" / f"sep_{new_id()}.wav"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with PBAudioFile(str(tmp), "w", DAW_SR, 2, bit_depth=16) as f:
            f.write(np.ascontiguousarray(data))
        try:
            api.ensure("htdemucs", str(gguf), "htdemucs", "sep", {}, cfg.get("unload_other_model", True), self.log.emit)
            self.log.emit("HTDemucs dolgozik… (ének, dob, basszus, egyéb)")
            res = api.run("htdemucs", {"audio": fwd(tmp), "options": {}})
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass
        outs = res.get("named_audio_outputs") or []
        if not outs:
            raise ApiError("A HTDemucs nem adott vissza sávokat. Kulcsok: " + ", ".join(res.keys()))
        out_dir = Path(cfg["output_dir"]) / dt.date.today().isoformat() / "savok"
        out_dir.mkdir(parents=True, exist_ok=True)
        base = f"{dt.datetime.now().strftime('%H%M%S')}_{safe_name(src.stem, 40)}"
        stems = {}
        for o in outs:
            sid = str(o.get("id") or f"sav{len(stems) + 1}").lower()
            fp = out_dir / f"{base}_{safe_name(sid, 20)}.wav"
            fp.write_bytes(base64.b64decode(o["audio"]))
            stems[sid] = str(fp)
        if cfg.get("unload_other_model", True):
            api.unload("htdemucs")
        self.log.emit(f"✓ Sávokra bontva: {', '.join(stems)} → {out_dir}")
        self.sep_done.emit(stems, job.get("ctx") or {})

    def _do_midi(self, job: dict):
        cfg = self.get_cfg()
        api = self._api()
        if not api.health():
            raise ApiError("A YuE2 szerver nem fut.")
        gguf = Path(cfg["models_dir"]) / "MuScriptor-Small-GGUF" / "muscriptor-small-f32.gguf"
        if not gguf.is_file():
            raise ApiError("A MuScriptor modell nincs letöltve (Telepítés & Szerver fül).")
        try:
            api.ensure("muscriptor", str(gguf), "muscriptor", "midi", {}, cfg.get("unload_other_model", True), self.log.emit)
            self.log.emit("MuScriptor: hangjegyek felismerése…")
            res = api.run("muscriptor", {"audio": fwd(job["audio"]), "options": {
                "instruments": job.get("instruments", ""), "output_format": "midi", "prelude_forcing": "false"}})
        finally:
            try:
                Path(job["audio"]).unlink()
            except OSError:
                pass
        midi = None
        for a in res.get("artifacts") or []:
            meta = a.get("meta") or {}
            if (a.get("kind") == "midi" or meta.get("format") == "midi") and isinstance(a.get("payload"), str):
                midi = base64.b64decode(a["payload"])
                break
        if not midi:
            raise ApiError("A MuScriptor nem adott vissza MIDI-t (lehet, hogy nem talált hangot ebben a részben).")
        out_dir = DAW_DIR / "midi"
        out_dir.mkdir(parents=True, exist_ok=True)
        fp = out_dir / f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{job.get('name', 'midi')}.mid"
        fp.write_bytes(midi)
        if cfg.get("unload_other_model", True):
            api.unload("muscriptor")
        self.log.emit(f"✓ MIDI kész: {fp}")
        self.midi_done.emit(str(fp))

    def _do_asr(self, job: dict):
        """Dalszöveg-felismerés Qwen3-ASR-rel. Bemenet: 16 kHz mono 16 bites WAV."""
        cfg = self.get_cfg()
        api = self._api()
        if not api.health():
            raise ApiError("A YuE2 szerver nem fut.")
        src = Path(job["audio"])
        self.log.emit(f"Szövegfelismerés: {src.name} - átalakítás 16 kHz mono WAV-ba…")
        with PBAudioFile(str(src)).resampled_to(16000) as f:
            data = f.read(f.frames)
        mono = np.ascontiguousarray(data.mean(axis=0, keepdims=True).astype(np.float32))
        peak = float(np.abs(mono).max()) if mono.size else 0.0
        if peak > 1e-4:
            mono = mono * (0.9 / peak)
        if job.get("timed"):
            if self._do_asr_timed(job, api, cfg, mono[0]):
                return
            self.log.emit("Időzítés nem sikerült (nem találtam jól elkülönülő frázisokat) - felismerés egyben.")
        tmp = DAW_DIR / "temp" / f"asr_{new_id()}.wav"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with PBAudioFile(str(tmp), "w", 16000, 1, bit_depth=16) as f:
            f.write(mono)
        try:
            try:
                api.ensure("qwen3_asr", job["model"], "qwen3_asr", "asr", {}, cfg.get("unload_other_model", True),
                           self.log.emit)
            except Exception as e:
                raise ApiError(f"A Qwen3-ASR modell nem töltődött be ({e}). Lehet, hogy a telepített motor ezt még nem "
                               "ismeri - frissítsd a motort a Telepítés fülön.")
            req = {"audio": fwd(tmp), "text": ""}
            if job.get("language"):
                req["language"] = job["language"]
            self.log.emit("Qwen3-ASR dolgozik… (hosszú dalnál pár perc)")
            res = api.run("qwen3_asr", req)
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass
        text, lang = clean_asr_text(res.get("text") or "")
        if not text:
            raise ApiError("A Qwen3-ASR nem ismert fel szöveget (lehet, hogy instrumentális a dal).")
        if cfg.get("unload_other_model", True):
            api.unload("qwen3_asr")
        self.log.emit(f"✓ Szöveg felismerve ({len(text.split())} szó).")
        self.asr_done.emit(text, job.get("language") or lang)

    def _do_asr_timed(self, job: dict, api, cfg, mono):
        """Frázisonkénti felismerés: az éneket a levegővételeknél darabolja, és minden darabot külön ismer fel.
        Így minden sor pontosan egy énekelt frázis lesz, időbélyeggel."""
        sr = 16000
        phrases = detect_vocal_phrases(mono, sr)
        if len(phrases) < 2:
            return False
        self.log.emit(f"{len(phrases)} énekelt frázist találtam - felismerés frázisonként…")
        try:
            api.ensure("qwen3_asr", job["model"], "qwen3_asr", "asr", {}, cfg.get("unload_other_model", True),
                       self.log.emit)
        except Exception as e:
            raise ApiError(f"A Qwen3-ASR modell nem töltődött be ({e}). Lehet, hogy a telepített motor ezt még nem "
                           "ismeri - frissítsd a motort a Telepítés fülön.")
        tmpdir = DAW_DIR / "temp"
        tmpdir.mkdir(parents=True, exist_ok=True)
        lines, lang = [], ""
        for i, (a, b) in enumerate(phrases):
            lo, hi = max(0, int((a - 0.15) * sr)), min(len(mono), int((b + 0.15) * sr))
            chunk = mono[lo:hi]
            tmp = tmpdir / f"asr_{new_id()}.wav"
            with PBAudioFile(str(tmp), "w", sr, 1, bit_depth=16) as f:
                f.write(np.ascontiguousarray(chunk[None, :]))
            try:
                req = {"audio": fwd(tmp), "text": ""}
                if job.get("language"):
                    req["language"] = job["language"]
                res = api.run("qwen3_asr", req)
            finally:
                try:
                    tmp.unlink()
                except OSError:
                    pass
            text, lg = clean_asr_text(res.get("text") or "")
            lang = lang or lg
            if text:
                lines.append({"start": round(a, 2), "end": round(b, 2), "text": text})
            if (i + 1) % 5 == 0 or i + 1 == len(phrases):
                self.log.emit(f"  … {i + 1}/{len(phrases)} frázis")
        if not lines:
            return False
        if cfg.get("unload_other_model", True):
            api.unload("qwen3_asr")
        self.log.emit(f"✓ {len(lines)} időzített sor felismerve.")
        self.asr_timed.emit(lines, job.get("language") or lang, len(mono) / sr)
        return True

    def _do_sheetsage(self, job: dict):
        cfg = self.get_cfg()
        api = self._api()
        if not api.health():
            raise ApiError("A YuE2 szerver nem fut.")
        src = Path(job["audio"])
        tmp = None
        if src.suffix.lower() != ".wav":
            ff = find_ffmpeg(cfg)
            if not ff:
                raise ApiError("Nem-WAV fájlhoz ffmpeg kell (Beállítások). Vagy adj meg WAV-ot.")
            fd, tmp = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            self.log.emit("Átalakítás WAV-ba (ffmpeg)…")
            p = subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-ac", "1",
                                "-ar", "44100", "-c:a", "pcm_s16le", tmp], capture_output=True,
                               creationflags=NO_WINDOW)
            if p.returncode != 0:
                raise ApiError("ffmpeg hiba: " + p.stderr.decode("utf-8", "replace")[:400])
            src = Path(tmp)
        try:
            gguf = Path(cfg["models_dir"]) / SHEET_DIRNAME / "sheetsage2-orig.gguf"
            if not gguf.is_file():
                raise ApiError("A SheetSage2 modell nincs letöltve.")
            api.ensure("sheetsage2", str(gguf), "sheetsage2", "midi", {}, cfg.get("unload_other_model", True),
                       self.log.emit)
            self.log.emit("Dallam és akkordok felismerése…")
            res = api.run("sheetsage2", {"audio": fwd(src), "options": {}})
            abc = result_abc(res) or (res.get("text") or "")
            if not abc.strip():
                raise ApiError("A SheetSage2 nem adott vissza ABC-kottát.")
            if cfg.get("unload_other_model", True):
                api.unload("sheetsage2")
            if job.get("target") == "remake":
                self.remake_abc.emit(abc)
            else:
                self.abc_extracted.emit(abc)
            self.log.emit("✓ ABC-kotta kinyerve.")
        finally:
            if tmp:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Állapot-lekérdező szál (szerver + GPU) - hogy a felület sose fagyjon be
# ---------------------------------------------------------------------------
class StatusPoller(QThread):
    health = Signal(bool)
    gpu = Signal(str)

    def __init__(self, port: int):
        super().__init__()
        self.port = port
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        smi = shutil.which("nvidia-smi")
        tick = 0
        while not self._stop:
            if tick % 2 == 0:
                self.health.emit(Yue2Api(int(self.port)).health())
            if smi and tick % 4 == 0:
                try:
                    out = subprocess.run([smi, "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                                          "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                         timeout=5, creationflags=NO_WINDOW).stdout.strip().splitlines()
                    if out:
                        name, used, total, util = [x.strip() for x in out[0].split(",")]
                        self.gpu.emit(f"{name}  ·  VRAM {int(used) / 1024:.1f} / {int(total) / 1024:.1f} GB  ·  {util}%")
                except Exception:
                    pass
            for _ in range(10):
                if self._stop:
                    return
                time.sleep(0.1)
            tick += 1


# ---------------------------------------------------------------------------
# Ollama segéd szál
# ---------------------------------------------------------------------------
class OllamaWorker(QThread):
    done = Signal(str, str)   # cél, szöveg
    failed = Signal(str)

    def __init__(self, url: str, model: str, prompt: str, target: str):
        super().__init__()
        self.url, self.model, self.prompt, self.target = url.rstrip("/"), model, prompt, target

    def run(self):
        try:
            body = json.dumps({"model": self.model, "prompt": self.prompt, "stream": False}).encode()
            req = urllib.request.Request(self.url + "/api/generate", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                txt = json.loads(r.read().decode("utf-8")).get("response", "")
            txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S).strip()
            self.done.emit(self.target, txt)
        except Exception as e:
            self.failed.emit(str(e))


def ollama_models(url: str) -> list[str]:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/api/tags", timeout=3) as r:
            return [m["name"] for m in json.loads(r.read().decode()).get("models", [])]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Kis widgetek
# ---------------------------------------------------------------------------
class OptNum(QWidget):
    """Opcionális szám: kikapcsolva a szerver alapértéke érvényes."""

    def __init__(self, kind, lo, hi, step, default, tip=""):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.chk = QCheckBox()
        self.chk.setToolTip("Bekapcsolva saját érték, kikapcsolva a modell alapértéke")
        if kind is float:
            self.spin = QDoubleSpinBox()
            self.spin.setDecimals(3)
            self.spin.setSingleStep(step)
        else:
            self.spin = QSpinBox()
            self.spin.setSingleStep(int(step))
        self.spin.setRange(lo, hi)
        self.spin.setValue(default)
        self.default = default
        self.spin.setEnabled(False)
        self.chk.toggled.connect(self.spin.setEnabled)
        lbl = QLabel(f"alap: {default}")
        lbl.setObjectName("dim")
        lay.addWidget(self.chk)
        lay.addWidget(self.spin, 1)
        lay.addWidget(lbl)
        self.setToolTip(tip)
        self.kind = kind

    def value(self):
        if not self.chk.isChecked():
            return None
        v = self.spin.value()
        return float(v) if self.kind is float else int(v)

    def set_value(self, v):
        if v is None:
            self.chk.setChecked(False)
            self.spin.setValue(self.default)
        else:
            self.chk.setChecked(True)
            self.spin.setValue(v)


def collapsible(title: str, content: QWidget, opened=False) -> QWidget:
    box = QWidget()
    lay = QVBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    btn = QToolButton()
    btn.setText(title)
    btn.setCheckable(True)
    btn.setChecked(opened)
    btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
    btn.setArrowType(Qt.DownArrow if opened else Qt.RightArrow)
    btn.setObjectName("collapse")
    content.setVisible(opened)

    def tog(on):
        content.setVisible(on)
        btn.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
    btn.toggled.connect(tog)
    lay.addWidget(btn)
    lay.addWidget(content)
    return box


DARK_QSS = """
* { font-family: 'Segoe UI', sans-serif; font-size: 10pt; color: #e6e6ea; }
QMainWindow, QWidget { background: #16171c; }
QTabWidget::pane { border: 1px solid #2c2e37; border-radius: 6px; top: -1px; }
QTabBar::tab { background: #1e2027; padding: 8px 16px; border: 1px solid #2c2e37; border-bottom: none;
               border-top-left-radius: 6px; border-top-right-radius: 6px; margin-right: 2px; color: #a8a9b3; }
QTabBar::tab:selected { background: #262833; color: #ffffff; border-color: #7c5cff; }
QGroupBox { border: 1px solid #2c2e37; border-radius: 8px; margin-top: 14px; padding: 10px 8px 8px 8px;
            background: #1b1c22; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #b9a8ff; }
QLineEdit, QPlainTextEdit, QTextBrowser, QSpinBox, QDoubleSpinBox, QComboBox, QTableWidget {
    background: #101116; border: 1px solid #2f313b; border-radius: 6px; padding: 4px;
    selection-background-color: #5a43d6; }
QPlainTextEdit { font-family: Consolas, 'Cascadia Mono', monospace; }
QLineEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border-color: #7c5cff; }
QSpinBox:disabled, QDoubleSpinBox:disabled { color: #5d5f6b; }
QPushButton { background: #2a2c36; border: 1px solid #3a3d4a; border-radius: 6px; padding: 6px 12px; }
QPushButton:hover { background: #343745; border-color: #7c5cff; }
QPushButton:pressed { background: #23242d; }
QPushButton:disabled { color: #5d5f6b; border-color: #2a2c36; }
QPushButton#primary { background: #6a4dff; border-color: #8a72ff; font-weight: 700; padding: 9px 16px; }
QPushButton#primary:hover { background: #7c62ff; }
QPushButton#danger { background: #4a2228; border-color: #7a3540; }
QPushButton#clear { background: #3a2f1a; border-color: #8a6a2a; color: #fcd68a; }
QPushButton#clear:hover { background: #4a3b1f; }
QWidget#projbar { border-bottom: 1px solid #2c2e37; }
QPushButton#tag { padding: 3px 8px; border-radius: 10px; background: #22242d; font-size: 9pt; }
QPushButton#tag:hover { background: #3a2f6e; }
QToolButton#collapse { border: none; color: #b9a8ff; font-weight: 600; padding: 4px; background: transparent; }
QHeaderView::section { background: #1e2027; border: none; border-right: 1px solid #2c2e37; padding: 5px; color: #b9b9c4; }
QTableWidget { gridline-color: #24262e; alternate-background-color: #14151a; }
QProgressBar { border: 1px solid #2f313b; border-radius: 6px; text-align: center; background: #101116; height: 16px; }
QProgressBar::chunk { background: #6a4dff; border-radius: 5px; }
QLabel, QCheckBox { background: transparent; }
QCheckBox::indicator { width: 15px; height: 15px; border: 1px solid #5a5d6b; border-radius: 4px; background: #101116; }
QCheckBox::indicator:hover { border-color: #8a72ff; }
QCheckBox::indicator:checked { background: #6a4dff; border-color: #8a72ff; image: none; }
QCheckBox::indicator:disabled { border-color: #33353f; }
QLabel#dim { color: #80828e; font-size: 9pt; }
QLabel#status_ok { color: #4ade80; font-weight: 700; }
QLabel#status_bad { color: #f87171; font-weight: 700; }
QLabel#status_mid { color: #facc15; font-weight: 700; }
QLabel#h1 { font-size: 15pt; font-weight: 700; color: #ffffff; }
QScrollArea { border: none; }
QSlider::groove:horizontal { height: 6px; background: #2a2c36; border-radius: 3px; }
QSlider::handle:horizontal { width: 14px; margin: -5px 0; background: #8a72ff; border-radius: 7px; }
QSlider::sub-page:horizontal { background: #6a4dff; border-radius: 3px; }
QSplitter::handle { background: #2c2e37; }
QWidget#playerbar { background: #1e2027; border-top: 1px solid #2c2e37; }
QFrame#tourbubble { background: #22232d; border: 2px solid #a78bfa; border-radius: 12px; }
QLabel#tourcount { color: #a78bfa; font-size: 9pt; }
QLabel#tourtitle { color: #ffffff; font-size: 14pt; font-weight: 700; }
QLabel#tourtext { color: #e6e6ea; font-size: 11pt; }
QWidget#playerbar QPushButton { font-size: 15pt; padding: 0px; }
QFrame#trackheader { background: #1b1c23; border-bottom: 1px solid #2a2c35; border-right: 1px solid #2a2c35; }
QFrame#trackheader QPushButton { padding: 3px 4px; }
QPushButton#tog_M:checked { background: #b45309; border-color: #f59e0b; }
QPushButton#tog_S:checked { background: #a16207; border-color: #facc15; color: #000; }
QPushButton#tog_A:checked { background: #854d0e; border-color: #facc15; }
QProgressBar#meter { border: none; border-radius: 2px; background: #0c0d11; }
QProgressBar#meter::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #22c55e, stop:0.75 #facc15, stop:1 #ef4444); border-radius: 2px; }
QToolTip { background: #23242e; color: #ececf1; border: 1px solid #7c5cff; padding: 6px; border-radius: 6px; }
"""


# ---------------------------------------------------------------------------
# Súgó buborékok (tooltipek) - minden felirat egy helyen (később fordítható)
# Formátum: "Cím|Leírás". A leírásban \n = új bekezdés.
# ---------------------------------------------------------------------------
TAB_TIPS = {
    "gen": "Generálás|Itt állítod össze a dalt: szöveg, stílus, mód, paraméterek - és innen indítod a generálást.",
    "abc": "ABC-kotta|A dal zenei terve (dallam + akkordok) kottaszövegként. Szerkesztheted, hosszabbíthatod, "
    "vagy kinyerheted egy meglévő hangfájlból.",
    "lib": "Könyvtár|Minden eddig generált szám: lejátszás, keresés, kedvencek, megjegyzés, MP3 export, "
    "és a beállítások visszatöltése.",
    "daw": "DAW – sávszerkesztő|Többsávos szerkesztő: dalok sávokra bontása (ének, dob, basszus, egyéb), klipek vágása, "
    "mozgatása, effektek, hangerő-automáció, tempó/hangnem, MIDI kotta, keverés exportálása.",
    "score": "Kotta|A zenei tervből vagy MIDI-ből igazi kotta: megjelenítés, transzponálás, lejátszás, nyomtatás, PDF.",
    "setup": "Telepítés & Szerver|A YuE2 motor és a modellek letöltése, valamint a háttérben futó YuE2 szerver "
    "indítása / leállítása és a naplója.",
    "settings": "Beállítások|Mappák, port, ffmpeg, automatikus indítás, Ollama segéd és a súgó buborékok ki/be kapcsolása.",
    "help": "Súgó|Részletes magyar útmutató a használathoz és a paraméterekhez.",
    "remake": "Átdolgozás|Egy meglévő dal (MP3/WAV) újrakészítése új stílusban: ének leválasztása, dallam és "
    "akkordok kinyerése, dalszöveg-felismerés, majd generálás több stílusban egyszerre.",
}

WIDGET_TIPS = {
    # ---- Generálás: dalszöveg
    "ed_lyrics": "Dalszöveg|Ide írd a dal szövegét. A szakaszokat szögletes zárójeles címkékkel jelöld, "
                 "mindegyik külön sorban a szövegblokk előtt (pl. [Verse], [Chorus]).\n"
                 "Tipp: rövid, énekelhető sorok; a refrént minden [Chorus]-ban szó szerint ismételd. "
                 "A YuE2 angolul tanult a legjobban - magyar szöveg is működhet, gyengébb kiejtéssel.",
    "lbl_lyrics": "Szöveg statisztika|Szakaszok, sorok és szavak száma - gyors ellenőrzés a szöveg hosszáról.",
    # ---- stílus
    "ed_style": "Stílus tagek|Vesszővel elválasztott ANGOL kulcsszavak, nem folyó mondat. Ideális 5–8 tag.\n"
                "Ajánlott sorrend: ének nyelve, műfaj, hangulat, 2-3 hangszer, ének típusa, produkció.\n"
                "Pl.: English, delta blues, smoky, slide guitar, harmonica, raspy male vocal, live recording",
    "lbl_style": "Tagek száma|Hány tag van most. 5–8 között a legjobb: kevesebbnél kiszámíthatatlanabb, "
                 "többnél a tagek ütik egymást.",
    "tag_combos.Nyelv": "Ének nyelve|Kiválasztva a tag a stílus végére kerül. Az első tag legyen a nyelv.",
    "tag_combos.Műfaj": "Műfaj|Zenei műfaj tag hozzáadása a stílushoz. Egy, legfeljebb két műfaj ajánlott.",
    "tag_combos.Hangulat": "Hangulat|A dal érzelmi színezete (pl. melancholic, energetic).",
    "tag_combos.Hangszer": "Hangszer|Vezető hangszerek. 2-3 elég - túl sok összemossa a hangzást.",
    "tag_combos.Ének": "Ének típusa|Énekhang neme, karaktere (pl. raspy male vocal, female vocal).",
    "tag_combos.Tempó": "Tempó / ritmus|Sebesség vagy forma (pl. slow tempo, 120 bpm, 12-bar).",
    "tag_combos.Produkció": "Produkció|A felvétel hangzása, keverése (pl. live recording, lo-fi, polished mix).",
    # ---- mód és modell
    "cb_cot": "COT mód (zenei tervezés)|off: nincs terv - leggyorsabb, vázlatokhoz.\n"
              "melody: előbb dallamtervet (ABC-kottát) készít, vagy a te ABC-dat követi.\n"
              "full: dallam + akkord terv - a legjobb dalszerkezet, kicsit lassabb. Ez az alapértelmezett.",
    "chk_use_abc": "Saját ABC-terv használata|Bekapcsolva az 'ABC-kotta' fülön lévő kottát kapja meg a modell "
                   "dallamtervként. Csak melody vagy full módban működik.\n"
                   "Kikapcsolva a modell maga tervez.",
    "chk_instr": "Instrumentális|Ének nélküli számhoz. A szöveg helyére [instrumental] kerül, a mód full-ra vált, "
                 "és ha letöltötted, kiválasztja az instrumentális LoRA-t.\n"
                 "LoRA nélkül maradhat némi zümmögés - ilyenkor az ABC fülön töröld az ének szólamot.",
    "cb_main": "Fő modell|A YuE2 fő hálója (tervezés + hangtokenek).\n"
               "q8_0: ajánlott, jó minőség, ~12 GB VRAM.\nq4_0: kevesebb memória, kicsit gyengébb.\n"
               "bf16: legjobb minőség és LoRA-hoz a legpontosabb, de több VRAM kell.",
    "cb_vae": "VAE (hangdekóder)|A belső kódokból valódi hanghullámot készít. Az f16 a szokásos; "
              "az f32 elméletileg pontosabb, de alig hallható különbséggel.",
    "cb_wt_model": "Fő modell súly tárolás|Hogyan tárolja a GPU a fő modell súlyait. 'native' = ahogy a fájlban van "
                   "(ajánlott). Kisebb típus (pl. q4_0) memóriát spórol minőségvesztéssel.",
    "cb_wt_vae": "VAE súly tárolás|Ugyanez a VAE-re. Általában maradjon 'native'.",
    # ---- seed
    "sp_seed": "Seed (véletlenmag)|Ugyanaz a seed + ugyanazok a beállítások = ugyanaz a dal. "
               "Egy jól sikerült szám seedjét megtartva finomíthatod a paramétereket.",
    "chk_rand": "Véletlen seed|Bekapcsolva minden generálás új, véletlen seedet kap (mindig más eredmény).",
    "sp_batch": "Darabszám|Hány változatot generáljon egyszerre. Mind a feladatsorba kerül és egymás után fut.",
    "cb_batch_mode": "Sorozat seed módja|seed, seed+1…: egymás utáni seedek (visszakereshető).\n"
                     "mind véletlen: teljesen különböző változatok.\n"
                     "azonos seed: ugyanaz a seed többször - paraméterek összehasonlításához.",
    # ---- haladó
    "opt_guid": "Guidance scale|Mennyire szigorúan kövesse a stílust és a szöveget.\n"
                "Magasabb: pontosabb követés, de túl magasan torz, artefaktos lehet. "
                "Alap: 1.01 (off módban), egyébként 1.0. Próbálj 1.0–3.0 között.",
    "opt_steps": "Inferencia lépések|A hangszintézis (NAR) lépéseinek száma. Alap: 8.\n"
                 "Több lépés: néha tisztább hang, de lassabb. 8–16 között érdemes kísérletezni.",
    "cb_ar_lora": "AR LoRA|Kiegészítő adapter a tervezéshez és a szemantikai (zenei tartalom) részhez. "
                  "Pl. az instrumentális adapter - ahhoz full COT kell.\n"
                  "Csak a modellmappában lévő .safetensors fájlok látszanak. Váltáskor a modell újratöltődik.",
    "sp_ar_scale": "AR LoRA erőssége|1.0 = teljes hatás, 0 = kikapcsolva. 0.5–0.8 finomabb hatás, "
                   "1 fölött túlzott, torz lehet.",
    "cb_nar_lora": "NAR LoRA|Kiegészítő adapter a hangzáshoz (akusztikus réteg), pl. a 'realaudio' v4 "
                   "valósághűbb hangszínért.",
    "sp_nar_scale": "NAR LoRA erőssége|1.0 = teljes hatás. 0 a teljes adaptert kikapcsolja.",
    # ---- preset + indítás
    "cb_preset": "Mentett presetek|A teljes űrlap (szöveg, stílus, minden paraméter) elmentett állapotai. "
                 "Válassz egyet, majd 'Betöltés'.",
    "ed_title": "Cím|A szám neve a könyvtárban és a fájlnévben. Üresen hagyva: idő + seed + stílus. "
                "Sorozatnál sorszám kerül mögé.",
    "btn_gen": "Generálás indítása|Az aktuális beállításokkal feladat(ok)at tesz a sorba. "
               "Ha a szerver fut, azonnal indul. Generálás közben is szerkesztheted és sorba állíthatod a következőt.",
    # ---- feladatsor
    "tbl_queue": "Feladatsor|A sorban álló, futó és kész generálások. Kész sorra duplán kattintva "
                 "lejátssza a dalt. A hiba okát az 'Állapot' cella buborékja mutatja.",
    "btn_pause": "Sor szüneteltetése|A futó generálás befejeződik, de a következő nem indul el, amíg újra "
                 "meg nem nyomod.",
    "lbl_last": "Utolsó eredmény|A legutóbb elkészült szám fájlneve, hossza és generálási ideje.",
    "btn_play_last": "Lejátszás|A legutóbb elkészült szám lejátszása a Könyvtár lejátszójában.",
    "btn_folder_last": "Mappa|Megnyitja a Windows Intézőt a fájlra rámutatva.",
    "btn_abc_last": "Generált ABC|A modell által készített zenei tervet átteszi az ABC-kotta fülre - "
                    "ott szerkesztheted, hosszabbíthatod, és újra felhasználhatod.",
    "btn_again": "Ugyanez új seeddel|Pontosan ugyanazokkal a beállításokkal, de új véletlen seeddel "
                 "generál egy újabb változatot.",
    "gen_log": "Napló|Mi történik: modell betöltés, generálás, hibák. Hiba esetén innen másold ki a sorokat.",
    # ---- ABC
    "ed_abc": "ABC-kotta szerkesztő|A dal szimbolikus terve ABC formátumban.\n"
              "Fejléc: X: sorszám, T: cím, M: ütemmutató, L: alap hanghossz, Q: tempó, K: hangnem.\n"
              "V: Vocal / V: Ins = ének és hangszer szólam. Hangok: A B c d (kisbetű = magasabb oktáv), "
              "akkordok idézőjelben: \"Am\".",
    "lbl_abc": "Kotta adatai|Sorok, ütemvonalak, szólamok, ütemmutató, tempó és hangnem gyors áttekintése.",
    "ed_ref": "Referencia hangfájl|Egy meglévő dal, amiből a SheetSage2 kinyeri a dallamot és az akkordokat. "
              "WAV közvetlenül megy; MP3/FLAC/M4A-hoz ffmpeg kell.",
    "btn_extract": "ABC kinyerése|A SheetSage2 modellel felismeri a referencia dal dallamát és harmóniáit, "
                   "és ABC-kottaként beírja a szerkesztőbe. Utána bekapcsolja az ABC használatát.\n"
                   "Pár percig tarthat; a YuE2-t addig kiüríti a VRAM-ból.",
    # ---- Könyvtár
    "ed_search": "Keresés|Szűrés címre, stílusra, dalszövegre és megjegyzésre - gépelés közben.",
    "chk_fav": "Csak kedvencek|Csak a csillaggal megjelölt számok látszanak.",
    "tbl_lib": "Generált számok|Dupla kattintás: lejátszás. Kijelölés után az alsó gombokkal kezelheted. "
               "Szürke sor = a fájl már nincs meg a lemezen.",
    "lbl_now": "Most szól|Az épp betöltött szám címe.",
    "btn_play": "Lejátszás / szünet|Elindítja vagy megállítja a lejátszást. Ha még nincs betöltve szám, a "
                "Könyvtárban kijelöltet (vagy a legutóbbit) indítja.\nBillentyű: Ctrl+Szóköz vagy a billentyűzet "
                "lejátszás gombja.",
    "btn_stop": "Leállítás|Megállítja és az elejére teker.",
    "btn_prev": "Előző szám|A Könyvtár listájában előző szám (a keresés/szűrés szerinti sorrendben).",
    "btn_next": "Következő szám|A Könyvtár listájában következő szám.",
    "chk_autonext": "Folyamatos lejátszás|Bekapcsolva egy szám vége után magától indul a következő - "
                    "így végighallgathatod a változatokat.",
    "btn_now_folder": "Mappa|Megnyitja az Intézőt a most szóló fájlra rámutatva.",
    "sl_pos": "Pozíció|Húzd vagy kattints rá a számban való ugráshoz.",
    "lbl_time": "Idő|Eltelt / teljes hossz.",
    "sl_vol": "Hangerő|Az app saját hangereje (a Windows hangerőtől független).",
    "lib_details": "Részletek|A kijelölt szám minden adata: fájl, modell, seed, stílus, paraméterek, dalszöveg.",
    # ---- Telepítés & szerver
    "lbl_engine": "Motor állapota|Hol található az audiocpp_server.exe - ez futtatja a YuE2-t a videókártyán.",
    "btn_engine": "Motor letöltése|Letölti a kész, lefordított audio.cpp szervert (CUDA) a GitHubról. "
                  "Ha a Remiqora már telepítette, erre nincs szükség.",
    "pkg_checks.main_q8": "Q8_0 fő modell|Ajánlott alapmodell, ~4 GB. A ✓ jelzi, ha már megvan.",
    "pkg_checks.main_q4": "Q4_0 fő modell|Kisebb, ~2,7 GB - kevesebb VRAM, kicsit gyengébb minőség.",
    "pkg_checks.main_bf16": "BF16 fő modell|A legjobb minőség, ~7 GB. LoRA-khoz ez a legpontosabb alap.",
    "pkg_checks.vae_f16": "VAE F16|Kötelező hangdekóder, ~0,3 GB.",
    "pkg_checks.vae_f32": "VAE F32|Opcionális, nagyobb pontosságú dekóder.",
    "pkg_checks.sheetsage": "SheetSage2|Hangfájlból ABC-kottát készít (dallam + akkord felismerés), ~2,7 GB.",
    "pkg_checks.htdemucs": "HTDemucs|Sávokra bontó modell a DAW-hoz: egy dalból ének, dob, basszus és egyéb "
                           "hangszerek sávot készít. Kicsi, gyors.",
    "pkg_checks.qwen3_asr": "Qwen3-ASR|Beszéd- és ének-felismerő modell (~2 GB). Az Átdolgozás fül ezzel írja le egy "
                            "meglévő dal szövegét - magyarul és angolul is ért, zenés felvételen is működik.",
    "pkg_checks.muscriptor": "MuScriptor|Hangból MIDI kottát készít a DAW-ban (jobb klikk egy klipen → MIDI). "
                             "Ha a Remiqora már letöltötte, ✓ jelzi.",
    "pkg_checks.lora_inst": "Instrumentális LoRA|Ének nélküli zenéhez készült adapter. Licenc: CC BY-NC "
                            "(nem kereskedelmi).",
    "pkg_checks.lora_nar": "Realaudio NAR LoRA|Valósághűbb hangszínt adó adapter. Licenc: CC BY-NC.",
    "btn_models": "Letöltés|A bepipált, még hiányzó csomagok letöltése a Hugging Face-ről. "
                  "Megszakítható, később onnan folytatja.",
    "lbl_dl": "Letöltés üzenet|Melyik fájl töltődik épp, vagy a végeredmény.",
    "pb_file": "Aktuális fájl|Az épp letöltődő fájl állapota GB-ban.",
    "pb_all": "Összesen|Hány fájl van kész a letöltendőkből.",
    "btn_dl_stop": "Letöltés megállítása|Félbehagyja; a részleges fájl megmarad, legközelebb onnan folytatja.",
    "lbl_srv": "Szerver állapota|zöld = fut és válaszol, sárga = indul, piros = leállítva. "
               "'(külső)' = nem ez az app indította (pl. a Remiqora).",
    "btn_start": "Szerver indítása|Elindítja a YuE2 szervert a háttérben. A modell csak az első "
                 "generáláskor töltődik a GPU-ra.",
    "btn_stop_srv": "Szerver leállítása|Leállítja a szervert és felszabadítja a teljes VRAM-ot.",
    "btn_restart": "Újraindítás|Leállítja és újraindítja a szervert - ha beragadt, vagy a futó generálást "
                   "meg kell szakítani.",
    "srv_log": "Szerver napló|A YuE2 szerver saját kimenete és az app üzenetei. Hibakereséshez hasznos.",
    # ---- Beállítások
    "set_widgets.engine_dir": "Motor mappa|Ahol az audiocpp_server.exe van (almappákban is keresi). "
                              "Remiqora esetén: AppData\\Local\\Remiqora\\engines\\YuE2",
    "set_widgets.models_dir": "Modellek mappa|Ennek alatta kell lennie a Yue2-3B-GGUF és SheetSage2-GGUF mappáknak.",
    "set_widgets.output_dir": "Kimeneti mappa|Ide kerülnek a generált WAV/MP3/ABC fájlok, napi almappákba.",
    "set_widgets.ffmpeg": "ffmpeg|MP3 exporthoz és nem-WAV referencia hangokhoz kell. Üresen a PATH-on keresi.",
    "sp_port": "Port|A YuE2 szerver helyi portja (alap 8080). A Remiqora is ezt használja - "
               "ne fussanak egyszerre.",
    "ed_gpu": "CUDA_VISIBLE_DEVICES|Melyik videókártyán fusson (0 = az első). Egy kártyánál maradjon 0.",
    "chk_auto": "Automatikus indítás|Az app indulásakor rögtön elindítja a YuE2 szervert.",
    "chk_unload": "Modell kiürítés váltáskor|A YuE2 és a SheetSage2 nem fér el együtt kényelmesen 16 GB-on - "
                  "bekapcsolva váltáskor az egyiket kiüríti. Ajánlott.",
    "chk_mp3": "Automatikus MP3|Minden kész WAV mellé 320 kbps MP3-at is készít (ffmpeg kell).",
    "chk_tips": "Súgó buborékok|Ezek a magyarázó buborékok. Kikapcsolva nem jelennek meg.",
    "ed_oll_url": "Ollama címe|A helyi Ollama szerver címe (alap: http://127.0.0.1:11434).",
    "cb_oll_model": "Ollama modell|Melyik helyi nyelvi modell írja a dalszöveget / stílus tageket. "
                    "'Modellek lekérése' feltölti a listát.",
    # ---- állapotsor
    "sb_server": "Szerver|A YuE2 szerver állapota.",
    "sb_queue": "Feladatsor|Hány feladat van még a sorban (a futóval együtt).",
    "sb_gpu": "Videókártya|Név, foglalt / teljes VRAM és terhelés (nvidia-smi alapján, 4 mp-enként).",
}

# gombok felirat szerint, fülenként (a ciklusban létrehozott névtelen gombokhoz)
BUTTON_TIPS = {
    "gen": {  # Generálás
        "[Intro]": "Intro címke|Bevezető rész, gyakran ének nélkül. A kurzor helyére szúrja be.",
        "[Verse]": "Verse címke|Versszak - a történet fő része. A kurzor helyére szúrja be.",
        "[Pre-Chorus]": "Pre-Chorus címke|Átvezetés a refrén előtt, feszültségépítés.",
        "[Chorus]": "Chorus címke|Refrén - a dal kulcsrésze. Minden refrénben ugyanaz a szöveg legyen.",
        "[Bridge]": "Bridge címke|Hangulatváltó rész a dal vége felé.",
        "[Solo]": "Solo címke|Hangszeres szóló rész.",
        "[Outro]": "Outro címke|A dal lezárása.",
        "[instrumental]": "Instrumental címke|Ének nélküli rész / teljes instrumentális szám jelölése.",
        "Példa": "Példa szöveg|Betölt egy kipróbált angol mintaszöveget - az első teszthez ideális.",
        "Fájlból…": "Betöltés fájlból|Dalszöveg betöltése .txt fájlból.",
        "Mentés…": "Mentés fájlba|A dalszöveg mentése .txt fájlba.",
        "Törlés": "Szöveg törlése|Kiüríti a dalszöveg mezőt.",
        "🤖 Ollama: dalszöveg": "Dalszöveg írása helyi AI-val|Megadsz egy témát, és a helyi Ollama modell "
                               "megírja a szöveget a megfelelő szakaszcímkékkel. Modell: Beállítások fül.",
        "🎲 Véletlen stílus": "Véletlen stílus|Összedob egy véletlen, de jól felépített 8 tagos stílust - "
                             "ötletadónak.",
        "Utolsó tag törlése": "Utolsó tag törlése|Leveszi a stílus utolsó tagját.",
        "Rendezés/duplikátum ki": "Duplikátumok ki|Kiszedi az ismétlődő tageket és rendbe teszi a vesszőket.",
        "🤖 Ollama: leírásból tagek": "Stílus magyar leírásból|Leírod magyarul, milyen hangzást szeretnél, "
                                     "a helyi Ollama modell angol stílus tagekké alakítja.",
        "LoRA fájl hozzáadása…": "LoRA hozzáadása|Bemásol egy .safetensors LoRA fájlt a modellmappába, "
                                 "hogy kiválasztható legyen. Csak nem-ComfyUI (unfused) fájl jó.",
        "Lista frissítése": "Lista frissítése|Újraolvassa a modellmappát (modellek, VAE-k, LoRA-k).",
        "Betöltés": "Preset betöltése|A kiválasztott preset minden beállítását visszaállítja.",
        "Mentés másként…": "Preset mentése|Az aktuális űrlap teljes állapotát elmenti név alatt.",
        "Preset törlése": "Preset törlése|Törli a kiválasztott presetet.",
        "Alaphelyzet": "Alaphelyzet|Mintaszöveg, mintastílus, full mód, minden paraméter alapértéken.",
        "Kijelölt törlése a sorból": "Kijelölt törlése|A kijelölt, még várakozó feladatot kiveszi a sorból. "
                                     "A futót nem állítja le.",
        "Várakozók törlése": "Várakozók törlése|Minden még el nem indult feladatot töröl. A futó befejeződik.",
        "⛔ Azonnali megszakítás": "Azonnali megszakítás|A futó generálást csak a szerver újraindításával "
                                  "lehet megállítani. A várakozó feladatok is törlődnek.",
    },
    "abc": {  # ABC
        "Megnyitás…": "ABC megnyitása|.abc kottafájl betöltése a szerkesztőbe (pl. egy korábbi generálásból).",
        "Mentés…": "ABC mentése|A szerkesztő tartalmának mentése .abc fájlba.",
        "Utolsó generált terv": "Utolsó generált terv|A legutóbbi melody/full generálás zenei tervét tölti be.",
        "Hosszabbítás ×2": "Hosszabbítás ×2|A teljes tervet kétszer egymás után ismétli (a szólamok szinkronban "
                           "maradnak) - hosszabb számot ad.",
        "Hosszabbítás ×3": "Hosszabbítás ×3|A tervet háromszor ismétli.",
        "Ének szólam törlése": "Ének szólam törlése|Kiveszi a 'V: Vocal' blokkokat - instrumentális számnál "
                               "segít a nem kívánt zümmögés ellen.",
        "Törlés": "Törlés|Kiüríti a kottaszerkesztőt.",
        "Tallózás…": "Tallózás|Referencia hangfájl kiválasztása.",
        "🎼 Kottaként": "Kottaként|Ezt a tervet igazi kottaként mutatja meg a Kotta fülön - nyomtatható, PDF-be menthető.",
    },
    "lib": {  # Könyvtár
        "★ Kedvenc": "Kedvenc|Csillag be/ki a kijelölt számon. A 'csak kedvencek' szűrővel gyorsan előkereshető.",
        "✎ Megjegyzés": "Megjegyzés|Saját jegyzet a számhoz (pl. 'jó refrén, gyenge vége'). Kereshető.",
        "Átnevezés": "Átnevezés|A szám címe a könyvtárban (a fájlnév nem változik).",
        "⟲ Beállítások vissza": "Beállítások visszatöltése|A kijelölt szám minden beállítását (szöveg, stílus, "
                                "seed, paraméterek, ABC) visszatölti a Generálás fülre - finomításhoz.",
        "Seed használata": "Seed használata|Csak a seedet veszi át, a véletlen seedet kikapcsolja.",
        "ABC → ABC fül": "ABC megnyitása|A számhoz mentett zenei tervet betölti az ABC-kotta fülre.",
        "📂 Mappa": "Mappa|Megnyitja a Windows Intézőt a fájlra rámutatva.",
        "MP3 export": "MP3 export|320 kbps MP3 készítése címmel és stílussal a metaadatokban (ffmpeg kell).",
        "🗑 Törlés": "Törlés|Kérdez: csak a listából törli, vagy a fájlokat (WAV, MP3, ABC) is.",
        "🎼 Kotta": "Kotta|A kijelölt dal zenei tervét (dallam + akkordok) kottaként mutatja a Kotta fülön.",
        "🎛 DAW-ba": "DAW-ba|A kijelölt dalt új sávként betölti a DAW sávszerkesztőbe.",
        "✂ Sávokra bontás": "Sávokra bontás|A kijelölt dalt 4 sávra bontja (ének, dob, basszus, egyéb), és a "
                           "sávokat a DAW-ba tölti. A HTDemucs modell kell hozzá (Telepítés fül).",
    },
    "setup": {  # Telepítés
        "Mappa megnyitása": "Modellek mappája|Megnyitja a modellek mappáját az Intézőben.",
        "Modellek kiürítése (VRAM)": "VRAM felszabadítása|Kiüríti a betöltött modelleket a videókártyáról, "
                                     "de a szerver tovább fut.",
        "GPU lista": "GPU lista|Kiírja a naplóba, milyen eszközöket lát a YuE2 motor.",
    },
    "settings": {  # Beállítások
        "…": "Tallózás|Mappa / fájl kiválasztása.",
        "Modellek lekérése": "Ollama modellek|Lekéri a helyi Ollama telepített modelljeit.",
        "💾 Mentés": "Mentés|A beállítások mentése. A port változás a szerver újraindítása után él.",
        "Remiqora telepítés keresése": "Remiqora használata|Megkeresi a Remiqora által már letöltött YuE2 motort "
                                       "és modelleket, és ezekre állítja a mappákat - nem kell újra letölteni.",
        "App mappa megnyitása": "App mappa|Megnyitja a YuE2 Studio mappáját (beállítások, adatbázis, presetek).",
    },
}

HEADER_TIPS = {
    "tbl_queue": ["Feladat sorszáma", "A szám címe vagy a stílus eleje", "A generálás seedje",
                  "várakozik / fut / kész / hiba (a hiba oka a cella buborékjában)", "Eltelt idő a futás kezdete óta"],
    "tbl_lib": ["Kedvenc", "Generálás időpontja", "Cím", "Stílus tagek", "Seed - ezzel megismételhető",
                "Zenei tervezési mód", "A szám hossza", "Mennyi ideig generálta", "Saját megjegyzés"],
}

SAMPLING_TIPS = {
    "temperature": "A választás véletlenszerűsége. Magasabb = változatosabb, bátrabb; alacsonyabb = "
                   "kiszámíthatóbb, de unalmasabb lehet.",
    "top_p": "Nucleus sampling: csak a legvalószínűbb lehetőségek ennyi összvalószínűségét veszi figyelembe. "
             "Alacsonyabb = óvatosabb.",
    "top_k": "Minden lépésben csak a k legvalószínűbb lehetőség közül választ.",
    "repetition_penalty": "Büntetés a frissen ismételt tokenekre. Ha a zene 'beragad' egy hurokba, növeld kicsit.",
    "penalty_window": "Hány előző tokent figyel az ismétlés büntetés.",
    "min_tokens": "Ennyi token előtt nem állhat le - a túl rövid eredmény ellen.",
    "max_tokens": "Felső korlát a tokenek számára.",
}
SAMPLING_EXTRA = {
    "sem": "Semantic sampling - a zenei hangtokenek generálása (a végső hangzást befolyásolja).",
    "abc": "ABC planner sampling - a zenei terv (kotta) generálása; csak melody/full módban, ha a modell tervez.",
}
TOKENS_PER_SEC = 25   # YuE2: 48 kHz / 1920-as VAE lépték = 25 latens képkocka (semantic token) másodpercenként


def fmt_mmss(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 60}:{sec % 60:02d}"


def estimate_lyrics_seconds(text: str) -> float | None:
    """Durva becslés: mennyi ideig tart elénekelni a szöveget (+ intro/outro/szóló szakaszok)."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return None
    total, sung = 0.0, 0
    for l in lines:
        m_ = re.match(r"^\[(.+)\]$", l)
        if m_:
            tag = m_.group(1).lower()
            if "instrumental" in tag and len(lines) <= 2:
                return None
            total += 10 if any(k in tag for k in ("intro", "outro", "solo", "instrumental", "break")) else 2
        else:
            sung += 1
            total += max(3.0, 0.55 * len(re.findall(r"\w+", l)) + 1.2)
    return total if sung else None


SEM_MAX_EXTRA = (" A semantic max tokens a szám maximális hosszát is megszabja (25 token = 1 mp). Egyszerűbb a "
                 "„Dal hossza” csúszka - ha az be van kapcsolva, ez a mező nem számít.")

ARENA_TIP = ("Szakértői beállítás: a GPU memóriaterület mérete ehhez a részhez (MB). Csak akkor módosítsd, "
             "ha a naplóban memória (arena / allocation) hiba jelenik meg. Kikapcsolva az alapérték él.")


def clear_button(fn, what: str) -> QPushButton:
    """Egységes „🧹 Tiszta lap” gomb minden fülre."""
    b = QPushButton("🧹 Tiszta lap")
    b.setObjectName("clear")
    b.setToolTip(tip_html("Tiszta lap|" + what + "\nElőtte rákérdez. Ha a munkát meg akarod tartani, előbb mentsd "
                          "a projektet (fent: 💾 Mentés)."))
    b.clicked.connect(fn)
    return b


def tip_html(text: str) -> str:
    text = tr(text)
    if "|" in text:
        title, body = text.split("|", 1)
    else:
        title, body = "", text
    body = body.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    paras = "".join(f"<p style='margin:3px 0'>{p}</p>" for p in body.split("\n") if p.strip())
    head = f"<b style='color:#c9b8ff'>{title}</b>" if title else ""
    width = 380 if len(body) > 120 else 0
    if width:
        return f"<qt><table width='{width}' cellspacing='0' cellpadding='0'><tr><td>{head}{paras}</td></tr></table></qt>"
    return f"<qt>{head}{paras}</qt>"



# ---------------------------------------------------------------------------
# Felület nyelve (i18n). A forrásnyelv a magyar; a fordítások a lang/<kód>.json fájlokban vannak
# ({"magyar szöveg": "fordítás", ...}). Bárki szerkesztheti / javíthatja őket szövegszerkesztővel.
# Nem magyar nyelvnél a Qt szövegbeállító függvényeit becsomagoljuk, így minden felirat, buborék,
# üzenet és napló fordítva jelenik meg. Magyarnál semmi sem változik.
# ---------------------------------------------------------------------------
LANG_DIR = APP_DIR / "lang"
LANG_URL = "https://raw.githubusercontent.com/sanyi1963bp/yue2-studio/main/lang/"
UI_LANGS = [("hu", "Magyar"), ("en", "English"), ("de", "Deutsch"), ("fr", "Français"), ("rom", "Romani (lovári)")]
I18N: dict[str, str] = {}
_I18N_RE = None
_I18N_FRAG: dict[str, str] = {}


def tr(s):
    """Fordítás: pontos egyezés, különben a szövegben lévő ismert részletek cseréje."""
    if not I18N or not isinstance(s, str) or not s:
        return s
    t = I18N.get(s)
    if t is not None:
        return t
    if _I18N_RE is None:
        return s
    return _I18N_RE.sub(lambda m_: _I18N_FRAG.get(m_.group(0), m_.group(0)), s)


def load_language(code: str) -> bool:
    global I18N, _I18N_RE, _I18N_FRAG
    I18N, _I18N_RE, _I18N_FRAG = {}, None, {}
    if not code or code == "hu":
        return True
    f = LANG_DIR / f"{code}.json"
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return False
    I18N = {k: v for k, v in data.items() if isinstance(v, str) and v and not k.startswith("__")}
    I18N.update({k: v for k, v in data.items() if k.startswith("__") and isinstance(v, str)})
    frags = sorted((k for k in I18N if len(k) >= 4 and not k.startswith("__") and "\n" not in k.strip()),
                   key=len, reverse=True)
    pats = []
    for k in frags:
        pat = re.escape(k)
        if k[0].isalnum():
            pat = r"(?<!\w)" + pat
        if k[-1].isalnum():
            pat = pat + r"(?!\w)"
        pats.append(pat)
    _I18N_FRAG = {k: I18N[k] for k in frags}
    if pats:
        _I18N_RE = re.compile("|".join(pats))
    return True


def translate_tree(root):
    """A már létrehozott (konstruktorban kapott) feliratok fordítása egy ablakrészen."""
    if not I18N or root is None:
        return
    from PySide6.QtWidgets import QAbstractButton, QTabWidget as _QTW
    widgets = [root] + root.findChildren(QWidget)
    for w_ in widgets:
        if isinstance(w_, QLabel):
            if w_.text():
                _ORIG["QLabel.setText"](w_, tr(w_.text()))
        elif isinstance(w_, QAbstractButton):
            if w_.text():
                _ORIG["QAbstractButton.setText"](w_, tr(w_.text()))
        elif isinstance(w_, QGroupBox):
            _ORIG["QGroupBox.setTitle"](w_, tr(w_.title()))
        elif isinstance(w_, (QLineEdit, QPlainTextEdit)):
            if w_.placeholderText():
                w_.setPlaceholderText(w_.placeholderText())
        elif isinstance(w_, _QTW):
            for i in range(w_.count()):
                _ORIG["QTabWidget.setTabText"](w_, i, tr(w_.tabText(i)))
        if w_.toolTip() and "<qt>" not in w_.toolTip():
            _ORIG["QWidget.setToolTip"](w_, tr(w_.toolTip()))
        if w_.isWindow() and w_.windowTitle():
            _ORIG["QWidget.setWindowTitle"](w_, tr(w_.windowTitle()))


_ORIG: dict = {}


def install_translation_hooks():
    """Qt függvények becsomagolása, hogy a szövegek fordítva jelenjenek meg."""
    if not I18N or _ORIG:
        return
    from PySide6.QtWidgets import (QAbstractButton, QMenu as _QMenu, QDialog as _QDialog, QFileDialog as _QFD,
                                   QStatusBar as _QSB, QTextBrowser as _QTB)

    def wrap(cls, name, fn):
        _ORIG[f"{cls.__name__}.{name}"] = getattr(cls, name)
        setattr(cls, name, fn)

    O = _ORIG
    wrap(QLabel, "setText", lambda self, t: O["QLabel.setText"](self, tr(t)))
    wrap(QAbstractButton, "setText", lambda self, t: O["QAbstractButton.setText"](self, tr(t)))
    wrap(QGroupBox, "setTitle", lambda self, t: O["QGroupBox.setTitle"](self, tr(t)))
    wrap(QWidget, "setToolTip", lambda self, t: O["QWidget.setToolTip"](self, tr(t)))
    wrap(QWidget, "setWindowTitle", lambda self, t: O["QWidget.setWindowTitle"](self, tr(t)))
    wrap(QLineEdit, "setPlaceholderText", lambda self, t: O["QLineEdit.setPlaceholderText"](self, tr(t)))
    wrap(QPlainTextEdit, "setPlaceholderText", lambda self, t: O["QPlainTextEdit.setPlaceholderText"](self, tr(t)))
    wrap(QPlainTextEdit, "appendPlainText", lambda self, t: O["QPlainTextEdit.appendPlainText"](self, tr(t)))
    wrap(_QSB, "showMessage", lambda self, t, *a: O["QStatusBar.showMessage"](self, tr(t), *a))

    def _add_item(self, *a):
        a = list(a)
        if a and isinstance(a[0], str) and (len(a) > 1 or a[0].endswith("…")):
            a[0] = tr(a[0])
        return O["QComboBox.addItem"](self, *a)
    wrap(QComboBox, "addItem", _add_item)
    wrap(QTabWidget, "addTab", lambda self, w_, t: O["QTabWidget.addTab"](self, w_, tr(t)))
    wrap(QTabWidget, "setTabText", lambda self, i, t: O["QTabWidget.setTabText"](self, i, tr(t)))
    wrap(QTabWidget, "setTabToolTip", lambda self, i, t: O["QTabWidget.setTabToolTip"](self, i, tr(t)))
    wrap(QTableWidget, "setHorizontalHeaderLabels",
         lambda self, ls: O["QTableWidget.setHorizontalHeaderLabels"](self, [tr(x) for x in ls]))

    def _add_action(self, *a):
        a = list(a)
        for i, x in enumerate(a):
            if isinstance(x, str):
                a[i] = tr(x)
                break
        return O["QMenu.addAction"](self, *a)
    wrap(_QMenu, "addAction", _add_action)
    wrap(_QMenu, "exec", lambda self, *a: (translate_tree(self), O["QMenu.exec"](self, *a))[1])
    wrap(_QDialog, "exec", lambda self: (translate_tree(self), O["QDialog.exec"](self))[1])
    wrap(_QTB, "setHtml", lambda self, h: O["QTextBrowser.setHtml"](
        self, I18N.get("__HELP_HTML__", h) if h == HELP_HTML else tr(h)))

    # QMessageBox: statikus függvények + példány
    for nm in ("information", "warning", "question", "critical"):
        f0 = getattr(QMessageBox, nm)
        O[f"QMessageBox.{nm}"] = f0

        def mk(f0):
            def fn(parent, title, text, *a, **k):
                return f0(parent, tr(title), tr(text), *a, **k)
            return staticmethod(fn)
        setattr(QMessageBox, nm, mk(f0))
    wrap(QMessageBox, "setText", lambda self, t: O["QMessageBox.setText"](self, tr(t)))
    wrap(QMessageBox, "setInformativeText", lambda self, t: O["QMessageBox.setInformativeText"](self, tr(t)))

    def _mb_add_button(self, *a):
        a = list(a)
        if a and isinstance(a[0], str):
            a[0] = tr(a[0])
        return O["QMessageBox.addButton"](self, *a)
    wrap(QMessageBox, "addButton", _mb_add_button)

    # QInputDialog: a választott elem EREDETI (magyar) szövegét adjuk vissza, mert a program ehhez hasonlít
    gi, gt = QInputDialog.getItem, QInputDialog.getText
    O["QInputDialog.getItem"], O["QInputDialog.getText"] = gi, gt

    def _get_item(parent, title, label, items, *a, **k):
        items = list(items)
        tx = [tr(x) for x in items]
        res, ok = gi(parent, tr(title), tr(label), tx, *a, **k)
        return (items[tx.index(res)] if res in tx else res), ok
    QInputDialog.getItem = staticmethod(_get_item)
    QInputDialog.getText = staticmethod(lambda parent, title, label, *a, **k: gt(parent, tr(title), tr(label), *a, **k))
    for nm in ("getOpenFileName", "getSaveFileName", "getOpenFileNames"):
        f0 = getattr(_QFD, nm)
        O[f"QFileDialog.{nm}"] = f0

        def mk2(f0):
            def fn(parent=None, caption="", directory="", filter="", *a, **k):
                return f0(parent, tr(caption), directory, tr(filter), *a, **k)
            return staticmethod(fn)
        setattr(_QFD, nm, mk2(f0))
    gd = _QFD.getExistingDirectory
    O["QFileDialog.getExistingDirectory"] = gd
    _QFD.getExistingDirectory = staticmethod(lambda parent=None, caption="", *a, **k: gd(parent, tr(caption), *a, **k))


class TooltipFilter(QObject):
    """Globális ki/be kapcsoló a súgó buborékokhoz."""

    def __init__(self):
        super().__init__()
        self.enabled = True

    def eventFilter(self, obj, ev):
        if not self.enabled and ev.type() == QEvent.ToolTip:
            return True
        return False


TOOLTIP_FILTER: "TooltipFilter | None" = None


def install_tip_filter(root):
    """A súgó-kapcsoló szűrőjét a saját elemekre tesszük (az egész appra tenni a webes kottanézettel ütközik)."""
    if TOOLTIP_FILTER is None or root is None:
        return
    try:
        from PySide6.QtWebEngineWidgets import QWebEngineView as _WEV
    except Exception:
        _WEV = None
    for w in [root] + root.findChildren(QWidget):
        if _WEV is not None:
            p = w
            skip = False
            while p is not None:
                if isinstance(p, _WEV):
                    skip = True
                    break
                p = p.parentWidget()
            if skip:
                continue
        if not w.property("_tipf"):
            w.installEventFilter(TOOLTIP_FILTER)
            w.setProperty("_tipf", True)


# ===========================================================================
# DAW - TÖBBSÁVOS SZERKESZTŐ: adatmodell, renderelés, valós idejű keverő
# ===========================================================================
try:
    import numpy as np
    HAS_NUMPY = True
except Exception:  # pragma: no cover
    np = None
    HAS_NUMPY = False
try:
    import pedalboard as pb
    from pedalboard.io import AudioFile as PBAudioFile
    HAS_PEDALBOARD = True
except Exception:  # pragma: no cover
    pb = None
    PBAudioFile = None
    HAS_PEDALBOARD = False
try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except Exception:  # pragma: no cover
    sd = None
    HAS_SOUNDDEVICE = False

import copy
import uuid
from concurrent.futures import ThreadPoolExecutor

DAW_SR = 44100
DAW_DIR = APP_DIR / "daw_projektek"
TRACK_COLORS = ["#8b5cf6", "#22d3ee", "#f59e0b", "#4ade80", "#f472b6", "#60a5fa", "#f87171", "#a3e635",
                "#fb923c", "#2dd4bf"]
STEM_NAMES = {"vocals": "Ének", "drums": "Dob", "bass": "Basszus", "other": "Egyéb hangszerek",
              "instrumental": "Kíséret", "piano": "Zongora", "guitar": "Gitár"}
STEM_INSTRUMENTS = {"vocals": "voice", "drums": "drums", "bass": "electric_bass,acoustic_bass"}

# Effekt definíciók: kulcs -> (magyar név, pedalboard osztály, [(param, felirat, min, max, lépés, alap)])
FX_DEFS = {
    "gain": ("Erősítés", "Gain", [("gain_db", "Erősítés (dB)", -40, 24, 0.5, 0.0)]),
    "highpass": ("Mélyvágó (HPF)", "HighpassFilter", [("cutoff_frequency_hz", "Vágási frekvencia (Hz)", 20, 2000, 5, 100.0)]),
    "lowpass": ("Magasvágó (LPF)", "LowpassFilter", [("cutoff_frequency_hz", "Vágási frekvencia (Hz)", 200, 20000, 50, 12000.0)]),
    "lowshelf": ("EQ mély polc", "LowShelfFilter", [("cutoff_frequency_hz", "Frekvencia (Hz)", 20, 1000, 5, 150.0),
                                                     ("gain_db", "Erősítés (dB)", -24, 24, 0.5, 0.0), ("q", "Q", 0.1, 4, 0.05, 0.707)]),
    "peak": ("EQ csúcs (sáv)", "PeakFilter", [("cutoff_frequency_hz", "Frekvencia (Hz)", 20, 20000, 10, 1000.0),
                                              ("gain_db", "Erősítés (dB)", -24, 24, 0.5, 0.0), ("q", "Q (szélesség)", 0.1, 10, 0.05, 1.0)]),
    "highshelf": ("EQ magas polc", "HighShelfFilter", [("cutoff_frequency_hz", "Frekvencia (Hz)", 1000, 20000, 50, 8000.0),
                                                       ("gain_db", "Erősítés (dB)", -24, 24, 0.5, 0.0), ("q", "Q", 0.1, 4, 0.05, 0.707)]),
    "compressor": ("Kompresszor", "Compressor", [("threshold_db", "Küszöb (dB)", -60, 0, 0.5, -18.0), ("ratio", "Arány", 1, 20, 0.5, 3.0),
                                                ("attack_ms", "Felfutás (ms)", 0.1, 200, 0.5, 10.0), ("release_ms", "Lecsengés (ms)", 5, 2000, 5, 150.0)]),
    "limiter": ("Limiter", "Limiter", [("threshold_db", "Küszöb (dB)", -30, 0, 0.5, -1.0), ("release_ms", "Lecsengés (ms)", 5, 1000, 5, 100.0)]),
    "gate": ("Zajkapu", "NoiseGate", [("threshold_db", "Küszöb (dB)", -100, 0, 1, -50.0), ("ratio", "Arány", 1, 50, 1, 10.0),
                                      ("attack_ms", "Felfutás (ms)", 0.1, 100, 0.5, 1.0), ("release_ms", "Lecsengés (ms)", 5, 1000, 5, 100.0)]),
    "reverb": ("Zengető", "Reverb", [("room_size", "Teremméret", 0, 1, 0.05, 0.5), ("damping", "Csillapítás", 0, 1, 0.05, 0.5),
                                     ("wet_level", "Zengés szint", 0, 1, 0.02, 0.25), ("dry_level", "Eredeti szint", 0, 1, 0.02, 0.8),
                                     ("width", "Szélesség", 0, 1, 0.05, 1.0)]),
    "delay": ("Visszhang (delay)", "Delay", [("delay_seconds", "Késleltetés (mp)", 0.01, 2, 0.01, 0.35), ("feedback", "Visszacsatolás", 0, 0.95, 0.05, 0.3),
                                             ("mix", "Keverés", 0, 1, 0.05, 0.25)]),
    "chorus": ("Kórus effekt", "Chorus", [("rate_hz", "Sebesség (Hz)", 0.05, 10, 0.05, 1.0), ("depth", "Mélység", 0, 1, 0.05, 0.25),
                                          ("centre_delay_ms", "Közép késl. (ms)", 1, 50, 0.5, 7.0), ("feedback", "Visszacsatolás", 0, 0.95, 0.05, 0.0),
                                          ("mix", "Keverés", 0, 1, 0.05, 0.5)]),
    "phaser": ("Fázistoló", "Phaser", [("rate_hz", "Sebesség (Hz)", 0.05, 10, 0.05, 1.0), ("depth", "Mélység", 0, 1, 0.05, 0.5),
                                       ("centre_frequency_hz", "Közép frekvencia (Hz)", 100, 10000, 50, 1300.0),
                                       ("feedback", "Visszacsatolás", 0, 0.95, 0.05, 0.0), ("mix", "Keverés", 0, 1, 0.05, 0.5)]),
    "distortion": ("Torzító", "Distortion", [("drive_db", "Meghajtás (dB)", 0, 60, 0.5, 15.0)]),
    "bitcrush": ("Bitcrusher", "Bitcrush", [("bit_depth", "Bitmélység", 2, 16, 0.5, 8.0)]),
    "clipping": ("Vágó (clipper)", "Clipping", [("threshold_db", "Küszöb (dB)", -30, 0, 0.5, -6.0)]),
    "pitch": ("Hangmagasság", "PitchShift", [("semitones", "Félhang", -12, 12, 0.5, 0.0)]),
}

FX_TIPS = {
    "gain": "Egyszerű hangerőnövelés vagy -csökkentés decibelben.",
    "highpass": "Levágja a mélyeket a frekvencia alatt. Énekre 80-120 Hz: eltünteti a dübörgést.",
    "lowpass": "Levágja a magasakat a frekvencia fölött. Tompább, 'messzebbi' hangot ad.",
    "lowshelf": "A mély tartomány egészét emeli vagy vágja.",
    "peak": "Egy frekvenciasávot emel vagy vág. Énekre 3-5 kHz emelés: tisztább, érthetőbb.",
    "highshelf": "A magas tartomány egészét emeli (fényesebb) vagy vágja (sötétebb).",
    "compressor": "Kiegyenlíti a hangerő-ingadozást: a halk részek előjönnek, a hangosak visszahúzódnak.",
    "limiter": "Nem engedi a jelet a küszöb fölé - a végső keverésre téve megakadályozza a torzulást.",
    "gate": "Elnémítja a küszöb alatti halk zajokat (pl. sávbontás utáni maradványok).",
    "reverb": "Teret ad a hangnak: kis teremtől nagy katedrálisig.",
    "delay": "Ismétlődő visszhang.",
    "chorus": "Több hangszer/hang egyszerre szólásának hatása - szélesebb, lebegőbb hang.",
    "phaser": "Söprő, 'örvénylő' hangszín-moduláció.",
    "distortion": "Torzítás - gitárra, basszusra, kemény hangzáshoz.",
    "bitcrush": "Szándékos 'digitális' lebutítás - retró, lo-fi hatás.",
    "clipping": "Keményen levágja a csúcsokat - agresszív hangzás.",
    "pitch": "Hangmagasság eltolása félhangokban (a tempó nem változik).",
}

FX_PRESETS = {
    "Ének - tiszta, közeli": [("highpass", {"cutoff_frequency_hz": 100.0}), ("compressor", {"threshold_db": -20.0, "ratio": 3.0}),
                              ("peak", {"cutoff_frequency_hz": 4000.0, "gain_db": 3.0, "q": 1.0}), ("reverb", {"room_size": 0.3, "wet_level": 0.15})],
    "Ének - nagy terem": [("highpass", {"cutoff_frequency_hz": 100.0}), ("compressor", {"threshold_db": -18.0, "ratio": 2.5}),
                          ("reverb", {"room_size": 0.85, "wet_level": 0.35, "dry_level": 0.75})],
    "Dob - ütős": [("compressor", {"threshold_db": -15.0, "ratio": 4.0, "attack_ms": 5.0}), ("lowshelf", {"cutoff_frequency_hz": 100.0, "gain_db": 3.0})],
    "Basszus - kerek": [("lowpass", {"cutoff_frequency_hz": 5000.0}), ("compressor", {"threshold_db": -20.0, "ratio": 4.0})],
    "Lo-fi rádió": [("highpass", {"cutoff_frequency_hz": 300.0}), ("lowpass", {"cutoff_frequency_hz": 3500.0}), ("bitcrush", {"bit_depth": 10.0})],
    "Mester - fényes, hangos": [("highshelf", {"cutoff_frequency_hz": 9000.0, "gain_db": 2.0}),
                                ("compressor", {"threshold_db": -12.0, "ratio": 2.0, "attack_ms": 20.0}), ("limiter", {"threshold_db": -1.0})],
}


def db2lin(db: float) -> float:
    return float(10 ** (db / 20.0)) if db > -90 else 0.0


def new_id() -> str:
    return uuid.uuid4().hex[:10]


def fx_instance(fx: dict):
    """Egy effekt leíróból pedalboard plugin."""
    name, cls, params = FX_DEFS[fx["type"]]
    obj = getattr(pb, cls)()
    for key, _lbl, lo, hi, _st, default in params:
        v = fx.get("params", {}).get(key, default)
        setattr(obj, key, float(min(max(v, lo), hi)))
    return obj


def make_board(chain: list[dict]):
    plugins = [fx_instance(f) for f in chain if f.get("enabled", True) and f.get("type") in FX_DEFS]
    return pb.Pedalboard(plugins) if plugins else None


def new_fx(kind: str, params: dict | None = None) -> dict:
    defaults = {p[0]: p[5] for p in FX_DEFS[kind][2]}
    defaults.update(params or {})
    return {"id": new_id(), "type": kind, "enabled": True, "params": defaults}


def new_track(name: str, color: str | None = None) -> dict:
    return {"id": new_id(), "name": name, "color": color or random.choice(TRACK_COLORS), "volume_db": 0.0,
            "pan": 0.0, "mute": False, "solo": False, "fx": [], "clips": [], "automation": [], "auto_on": False}


def new_clip(src: str, start: float, length: float, name: str = "") -> dict:
    return {"id": new_id(), "src": src, "name": name or Path(src).stem, "start": start, "offset": 0.0,
            "length": length, "gain_db": 0.0, "fade_in": 0.0, "fade_out": 0.0, "muted": False,
            "stretch": 1.0, "pitch": 0.0, "reverse": False}


def new_project() -> dict:
    return {"version": 1, "name": "Új projekt", "bpm": 120.0, "beats_per_bar": 4, "tracks": [],
            "master_fx": [], "master_db": 0.0, "loop": [0.0, 8.0], "loop_on": False}


def read_audio(path: str):
    """Hangfájl -> (2, N) float32, 44.1 kHz."""
    with PBAudioFile(str(path)).resampled_to(DAW_SR) as f:
        data = f.read(f.frames)
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[0] == 1:
        data = np.vstack([data, data])
    elif data.shape[0] > 2:
        data = data[:2]
    return np.ascontiguousarray(data)


def write_audio(path: str, data, quality=None):
    path = str(path)
    kw = {}
    if path.lower().endswith(".mp3"):
        kw["quality"] = quality or "320k"
    with PBAudioFile(path, "w", DAW_SR, data.shape[0], **kw) as f:
        f.write(np.ascontiguousarray(data, dtype=np.float32))


def estimate_bpm(data) -> float:
    """Durva tempóbecslés onset-autokorrelációval (60-180 BPM)."""
    mono = data.mean(axis=0) if data.ndim == 2 else data
    mono = mono[: DAW_SR * 90]
    hop = 512
    n = len(mono) // hop
    if n < 64:
        return 120.0
    frames = mono[: n * hop].reshape(n, hop)
    energy = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    onset = np.maximum(0, np.diff(np.log(energy + 1e-6)))
    onset -= onset.mean()
    ac = np.correlate(onset, onset, mode="full")[len(onset) - 1:]
    fps = DAW_SR / hop
    lo, hi = int(fps * 60 / 180), int(fps * 60 / 60)
    seg = ac[lo:hi]
    if len(seg) == 0:
        return 120.0
    lag = lo + int(np.argmax(seg))
    bpm = 60.0 * fps / lag
    while bpm < 70:
        bpm *= 2
    while bpm > 170:
        bpm /= 2
    return round(bpm, 1)


class SourceCache:
    """Forrásfájlok és tempó/hangnem-módosított változataik gyorsítótára."""

    def __init__(self):
        self.lock = threading.Lock()
        self.raw: dict[str, object] = {}
        self.proc: dict[tuple, object] = {}
        self.peaks: dict[tuple, object] = {}

    def get_raw(self, src: str):
        with self.lock:
            if src in self.raw:
                return self.raw[src]
        data = read_audio(src)
        with self.lock:
            self.raw[src] = data
        return data

    def get(self, src: str, stretch: float = 1.0, pitch: float = 0.0, reverse: bool = False):
        key = (src, round(stretch, 4), round(pitch, 3), bool(reverse))
        with self.lock:
            if key in self.proc:
                return self.proc[key]
        data = self.get_raw(src)
        if abs(stretch - 1.0) > 1e-4 or abs(pitch) > 1e-3:
            data = pb.time_stretch(data, DAW_SR, stretch_factor=float(stretch),
                                   pitch_shift_in_semitones=float(pitch))
            data = np.ascontiguousarray(data, dtype=np.float32)
        if reverse:
            data = np.ascontiguousarray(data[:, ::-1])
        with self.lock:
            self.proc[key] = data
        return data

    def get_peaks(self, src, stretch=1.0, pitch=0.0, reverse=False, block=256):
        key = (src, round(stretch, 4), round(pitch, 3), bool(reverse))
        with self.lock:
            if key in self.peaks:
                return self.peaks[key]
        data = self.get(src, stretch, pitch, reverse)
        mono = np.abs(data).max(axis=0)
        n = len(mono) // block
        pk = mono[: n * block].reshape(n, block).max(axis=1) if n else np.zeros(1, np.float32)
        with self.lock:
            self.peaks[key] = pk
        return pk

    def length(self, src, stretch=1.0, pitch=0.0, reverse=False) -> float:
        return self.get(src, stretch, pitch, reverse).shape[1] / DAW_SR


def clip_render_key(c: dict) -> tuple:
    return tuple(c[k] for k in ("src", "start", "offset", "length", "gain_db", "fade_in", "fade_out", "muted",
                                "stretch", "pitch", "reverse"))


def track_render_key(t: dict) -> str:
    return json.dumps([[clip_render_key(c) for c in t["clips"]], t["fx"]], sort_keys=True, default=str)


def render_clip_into(buf, clip: dict, cache: SourceCache):
    if clip.get("muted"):
        return
    src = cache.get(clip["src"], clip.get("stretch", 1.0), clip.get("pitch", 0.0), clip.get("reverse", False))
    o = int(round(clip["offset"] * DAW_SR))
    n = int(round(clip["length"] * DAW_SR))
    seg = src[:, o:o + n]
    n = seg.shape[1]
    if n <= 0:
        return
    seg = seg * db2lin(clip.get("gain_db", 0.0))
    fi = min(n, int(clip.get("fade_in", 0) * DAW_SR))
    fo = min(n, int(clip.get("fade_out", 0) * DAW_SR))
    if fi > 0:
        seg[:, :fi] *= np.linspace(0, 1, fi, dtype=np.float32)
    if fo > 0:
        seg[:, n - fo:] *= np.linspace(1, 0, fo, dtype=np.float32)
    s = int(round(clip["start"] * DAW_SR))
    if s < 0:
        seg = seg[:, -s:]
        s = 0
    end = min(buf.shape[1], s + seg.shape[1])
    if end > s:
        buf[:, s:end] += seg[:, : end - s]


def project_length(project: dict) -> float:
    end = 0.0
    for t in project["tracks"]:
        for c in t["clips"]:
            end = max(end, c["start"] + c["length"])
    return end


def render_track(track: dict, cache: SourceCache, total_len: float):
    n = int((total_len + 1.0) * DAW_SR)
    buf = np.zeros((2, n), dtype=np.float32)
    for c in track["clips"]:
        render_clip_into(buf, c, cache)
    board = make_board(track["fx"])
    if board is not None:
        # a zengés/visszhang farka is beleférjen
        tail = np.zeros((2, DAW_SR * 3), dtype=np.float32)
        out = board(np.concatenate([buf, tail], axis=1), DAW_SR)
        buf = np.ascontiguousarray(out[:, :n], dtype=np.float32)
    return buf


def pan_gains(pan: float) -> tuple[float, float]:
    """Állandó teljesítményű panoráma, -1 (bal) .. +1 (jobb)."""
    a = (pan + 1) * math.pi / 4
    return math.cos(a) * math.sqrt(2), math.sin(a) * math.sqrt(2)


def automation_gain(points: list, t0: float, n: int):
    """Hangerő-automáció (dB pontok) -> lineáris erősítés tömb."""
    if not points:
        return None
    ts = np.array([p[0] for p in points], dtype=np.float64)
    ds = np.array([p[1] for p in points], dtype=np.float64)
    tt = t0 + np.arange(n) / DAW_SR
    db = np.interp(tt, ts, ds)
    return np.where(db <= -60, 0.0, 10 ** (db / 20.0)).astype(np.float32)


class MixState:
    """A hangszál által olvasott, atomikusan lecserélt keverési állapot."""

    def __init__(self):
        self.tracks: list[dict] = []   # {id, buf, gl, gr, active, auto}
        self.master_gain = 1.0
        self.master_board = None
        self.loop = None


class DawEngine(QObject):
    rendered = Signal(str)          # sáv id
    render_failed = Signal(str, str)
    state_changed = Signal()

    def __init__(self):
        super().__init__()
        self.cache = SourceCache()
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.buffers: dict[str, object] = {}
        self.buf_keys: dict[str, str] = {}
        self.pending: dict[str, str] = {}
        self.mix = MixState()
        self.pos = 0                    # minták
        self.playing = False
        self.stream = None
        self.peaks: dict[str, float] = {}
        self.master_peak = (0.0, 0.0)
        self.total_len = 0.0
        self.lock = threading.Lock()
        self.error = ""

    # ---------------------------------------------------------- render
    def sync(self, project: dict):
        """A projekt változása után: hiányzó/elavult sáv-renderek indítása, keverési állapot frissítése."""
        self.total_len = max(project_length(project), 1.0)
        need_len = self.total_len
        for t in project["tracks"]:
            key = track_render_key(t) + f"|{need_len:.2f}"
            if self.buf_keys.get(t["id"]) != key and self.pending.get(t["id"]) != key:
                self.pending[t["id"]] = key
                snap = copy.deepcopy(t)
                self.pool.submit(self._render_job, snap, key, need_len)
        alive = {t["id"] for t in project["tracks"]}
        for tid in list(self.buffers):
            if tid not in alive:
                self.buffers.pop(tid, None)
                self.buf_keys.pop(tid, None)
        self.update_mix(project)

    def _render_job(self, track, key, total_len):
        try:
            buf = render_track(track, self.cache, total_len)
            if self.pending.get(track["id"]) != key:
                return
            self.buffers[track["id"]] = buf
            self.buf_keys[track["id"]] = key
            self.pending.pop(track["id"], None)
            self._rebuild_buffers()
            self.rendered.emit(track["id"])
        except Exception as e:
            self.pending.pop(track["id"], None)
            self.render_failed.emit(track["id"], str(e))

    def is_rendering(self) -> bool:
        return bool(self.pending)

    def update_mix(self, project: dict):
        self._project = project
        self._rebuild_buffers()

    def _rebuild_buffers(self):
        project = getattr(self, "_project", None)
        if project is None:
            return
        any_solo = any(t["solo"] for t in project["tracks"])
        tracks = []
        for t in project["tracks"]:
            buf = self.buffers.get(t["id"])
            if buf is None:
                continue
            gl, gr = pan_gains(t["pan"])
            g = db2lin(t["volume_db"])
            active = not t["mute"] and (t["solo"] or not any_solo)
            tracks.append({"id": t["id"], "buf": buf, "gl": gl * g, "gr": gr * g, "active": active,
                           "auto": sorted(t["automation"]) if t.get("auto_on") and t["automation"] else None})
        m = MixState()
        m.tracks = tracks
        m.master_gain = db2lin(project["master_db"])
        try:
            m.master_board = make_board(project["master_fx"])
        except Exception:
            m.master_board = None
        lp = project.get("loop")
        m.loop = (int(lp[0] * DAW_SR), int(lp[1] * DAW_SR)) if project.get("loop_on") and lp and lp[1] > lp[0] else None
        self.mix = m

    # ---------------------------------------------------------- keverés
    def mix_block(self, start: int, n: int, mix: MixState | None = None, realtime=True):
        mix = mix or self.mix
        out = np.zeros((2, n), dtype=np.float32)
        for tr in mix.tracks:
            buf = tr["buf"]
            seg = buf[:, start:start + n]
            m = seg.shape[1]
            if m <= 0:
                if realtime:
                    self.peaks[tr["id"]] = 0.0
                continue
            if not tr["active"]:
                if realtime:
                    self.peaks[tr["id"]] = 0.0
                continue
            l = seg[0] * tr["gl"]
            r = seg[1] * tr["gr"]
            if tr["auto"]:
                ag = automation_gain(tr["auto"], start / DAW_SR, m)
                l = l * ag
                r = r * ag
            out[0, :m] += l
            out[1, :m] += r
            if realtime:
                self.peaks[tr["id"]] = float(max(np.abs(l).max(initial=0), np.abs(r).max(initial=0)))
        if mix.master_board is not None:
            out = mix.master_board(out, DAW_SR, reset=False)
        out *= mix.master_gain
        return out

    def _callback(self, outdata, frames, time_info, status):
        try:
            if not self.playing:
                outdata[:] = 0
                return
            mix = self.mix
            pos = self.pos
            block = self.mix_block(pos, frames, mix)
            pos += frames
            if mix.loop and pos >= mix.loop[1]:
                pos = mix.loop[0]
                if mix.master_board is not None:
                    mix.master_board.reset()
            self.pos = pos
            np.clip(block, -1.0, 1.0, out=block)
            outdata[:] = block.T
            self.master_peak = (float(np.abs(block[0]).max(initial=0)), float(np.abs(block[1]).max(initial=0)))
            if not mix.loop and pos > (self.total_len + 3) * DAW_SR:
                self.playing = False
        except Exception as e:  # a hangszálban sosem dobunk
            self.error = str(e)
            outdata[:] = 0

    def ensure_stream(self) -> str:
        if not HAS_SOUNDDEVICE:
            return "A 'sounddevice' csomag hiányzik (pip install sounddevice)."
        if self.stream is None:
            try:
                self.stream = sd.OutputStream(samplerate=DAW_SR, channels=2, dtype="float32", blocksize=1024,
                                              latency="low", callback=self._callback)
                self.stream.start()
            except Exception as e:
                self.stream = None
                return f"Nem sikerült megnyitni a hangkimenetet: {e}"
        return ""

    def play(self) -> str:
        err = self.ensure_stream()
        if err:
            return err
        if self.mix.master_board is not None:
            self.mix.master_board.reset()
        self.playing = True
        return ""

    def pause(self):
        self.playing = False

    def seek(self, seconds: float):
        self.pos = max(0, int(seconds * DAW_SR))
        if self.mix.master_board is not None:
            self.mix.master_board.reset()

    def position(self) -> float:
        return self.pos / DAW_SR

    def close(self):
        self.playing = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        self.pool.shutdown(wait=False, cancel_futures=True)

    # ---------------------------------------------------------- export
    def render_offline(self, project: dict, start: float = 0.0, end: float | None = None, only_track: str | None = None,
                       progress=None):
        """Teljes keverés (vagy egy sáv) kiszámítása fájlba íráshoz."""
        total = max(project_length(project), 0.1)
        for t in project["tracks"]:
            key = track_render_key(t) + f"|{max(total, 1.0):.2f}"
            if self.buf_keys.get(t["id"]) != key:
                self.buffers[t["id"]] = render_track(t, self.cache, max(total, 1.0))
                self.buf_keys[t["id"]] = key
        end = end if end is not None else total + 2.0
        s0, s1 = int(start * DAW_SR), int(end * DAW_SR)
        any_solo = any(t["solo"] for t in project["tracks"])
        m = MixState()
        for t in project["tracks"]:
            if only_track and t["id"] != only_track:
                continue
            gl, gr = pan_gains(t["pan"])
            g = db2lin(t["volume_db"])
            active = True if only_track else (not t["mute"] and (t["solo"] or not any_solo))
            m.tracks.append({"id": t["id"], "buf": self.buffers[t["id"]], "gl": gl * g, "gr": gr * g, "active": active,
                             "auto": sorted(t["automation"]) if t.get("auto_on") and t["automation"] else None})
        m.master_gain = 1.0 if only_track else db2lin(project["master_db"])
        m.master_board = None if only_track else make_board(project["master_fx"])
        chunks = []
        step = DAW_SR * 5
        p = s0
        while p < s1:
            n = min(step, s1 - p)
            chunks.append(self.mix_block(p, n, m, realtime=False))
            p += n
            if progress:
                progress((p - s0) / max(1, s1 - s0))
        out = np.concatenate(chunks, axis=1) if chunks else np.zeros((2, 1), np.float32)
        return np.clip(out, -1.0, 1.0)


# ---------------------------------------------------------------------------
# DAW felület
# ---------------------------------------------------------------------------
from PySide6.QtCore import QPointF, QRectF, QLineF
from PySide6.QtGui import QBrush, QPainter, QPainterPath, QPen, QPixmap, QCursor
from PySide6.QtWidgets import QColorDialog, QDialog, QDialogButtonBox, QFrame, QListWidget, QListWidgetItem, QMenu, \
    QScrollBar, QSizePolicy

HEADER_W = 300
TRACK_H = 104
RULER_H = 30


def fmt_time(t: float) -> str:
    t = max(0.0, t)
    return f"{int(t // 60)}:{t % 60:05.2f}"


class TrackHeader(QFrame):
    def __init__(self, panel, track: dict):
        super().__init__()
        self.panel = panel
        self.tid = track["id"]
        self.setFixedHeight(TRACK_H)
        self.setFixedWidth(HEADER_W)
        self.setObjectName("trackheader")
        g = QGridLayout(self)
        g.setContentsMargins(8, 5, 6, 5)
        g.setHorizontalSpacing(4)
        g.setVerticalSpacing(3)
        self.btn_color = QPushButton()
        self.btn_color.setFixedSize(14, 26)
        self.btn_color.setStyleSheet(f"background:{track['color']}; border:none; border-radius:3px;")
        self.btn_color.setToolTip(tip_html("Sáv színe|Kattintásra új színt választhatsz."))
        self.btn_color.clicked.connect(self._pick_color)
        self.ed_name = QLineEdit(track["name"])
        self.ed_name.setCursorPosition(0)
        self.ed_name.setToolTip(tip_html("Sáv neve|Átírható. A MIDI-készítés a névből (ének, dob, basszus) "
                                         "tippeli ki a hangszert."))
        self.ed_name.editingFinished.connect(lambda: panel.set_track_prop(self.tid, "name", self.ed_name.text().strip() or "Sáv"))
        row = QHBoxLayout()
        row.setSpacing(3)
        row.addWidget(self.btn_color)
        row.addWidget(self.ed_name, 1)
        self.btn_m = self._tog("M", track["mute"], "Némítás|A sáv nem szól.", "mute")
        self.btn_s = self._tog("S", track["solo"], "Szóló|Csak a szólóra kapcsolt sávok szólnak.", "solo")
        self.btn_a = self._tog("A", track.get("auto_on", False),
                               "Hangerő-automáció|Bekapcsolva a sávon látszik és szerkeszthető a hangerőgörbe: "
                               "kattintás = új pont, húzás = mozgatás, jobb klikk a ponton = törlés. Ilyenkor a "
                               "klipek nem mozgathatók ezen a sávon.", "auto_on")
        for b in (self.btn_m, self.btn_s, self.btn_a):
            row.addWidget(b)
        self.btn_fx = QPushButton(self._fx_text(track))
        self.btn_fx.setFixedWidth(46)
        self.btn_fx.setToolTip(tip_html("Effektek|A sáv effektlánca: EQ, kompresszor, zengető, visszhang stb. "
                                        "A szám az aktív effektek számát mutatja."))
        self.btn_fx.clicked.connect(lambda: panel.open_fx(self.tid))
        row.addWidget(self.btn_fx)
        self.btn_menu = QPushButton("⋮")
        self.btn_menu.setFixedWidth(24)
        self.btn_menu.setToolTip(tip_html("Sáv menü|Átnevezés, duplikálás, mozgatás, export, MIDI, törlés."))
        self.btn_menu.clicked.connect(lambda: panel.track_menu(self.tid, self.btn_menu))
        row.addWidget(self.btn_menu)
        g.addLayout(row, 0, 0, 1, 3)

        lbl_v = QLabel("Hangerő")
        lbl_v.setObjectName("dim")
        self.sl_vol = QSlider(Qt.Horizontal)
        self.sl_vol.setRange(-600, 120)
        self.sl_vol.setValue(int(track["volume_db"] * 10))
        self.lbl_vol = QLabel(f"{track['volume_db']:+.1f} dB")
        self.lbl_vol.setFixedWidth(62)
        tipv = tip_html("Sáv hangereje|-60 dB (néma) és +12 dB között. Dupla kattintás a feliratra: 0 dB.")
        for w in (lbl_v, self.sl_vol, self.lbl_vol):
            w.setToolTip(tipv)
        self.sl_vol.valueChanged.connect(self._vol_changed)
        self.sl_vol.sliderReleased.connect(lambda: panel.commit_mix())
        self.lbl_vol.mouseDoubleClickEvent = lambda e: self.sl_vol.setValue(0) or panel.commit_mix()
        g.addWidget(lbl_v, 1, 0)
        g.addWidget(self.sl_vol, 1, 1)
        g.addWidget(self.lbl_vol, 1, 2)
        lbl_p = QLabel("Panoráma")
        lbl_p.setObjectName("dim")
        self.sl_pan = QSlider(Qt.Horizontal)
        self.sl_pan.setRange(-100, 100)
        self.sl_pan.setValue(int(track["pan"] * 100))
        self.lbl_pan = QLabel(self._pan_text(track["pan"]))
        self.lbl_pan.setFixedWidth(62)
        tipp = tip_html("Panoráma|A sáv helye a sztereó térben: bal - közép - jobb. Dupla kattintás a feliratra: közép.")
        for w in (lbl_p, self.sl_pan, self.lbl_pan):
            w.setToolTip(tipp)
        self.sl_pan.valueChanged.connect(self._pan_changed)
        self.sl_pan.sliderReleased.connect(lambda: panel.commit_mix())
        self.lbl_pan.mouseDoubleClickEvent = lambda e: self.sl_pan.setValue(0) or panel.commit_mix()
        g.addWidget(lbl_p, 2, 0)
        g.addWidget(self.sl_pan, 2, 1)
        g.addWidget(self.lbl_pan, 2, 2)
        self.meter = QProgressBar()
        self.meter.setRange(0, 100)
        self.meter.setTextVisible(False)
        self.meter.setFixedHeight(6)
        self.meter.setObjectName("meter")
        self.meter.setToolTip(tip_html("Szintmérő|A sáv pillanatnyi hangereje lejátszás közben."))
        g.addWidget(self.meter, 3, 0, 1, 3)

    def _fx_text(self, track):
        n = sum(1 for f in track["fx"] if f.get("enabled", True))
        return f"FX {n}" if n else "FX"

    def _tog(self, txt, on, tip, prop):
        b = QPushButton(txt)
        b.setCheckable(True)
        b.setChecked(bool(on))
        b.setFixedWidth(28)
        b.setObjectName("tog_" + txt)
        b.setToolTip(tip_html(tip))
        b.toggled.connect(lambda v: self.panel.set_track_prop(self.tid, prop, v))
        return b

    @staticmethod
    def _pan_text(p):
        if abs(p) < 0.01:
            return "közép"
        return f"B {abs(p) * 100:.0f}" if p < 0 else f"J {p * 100:.0f}"

    def _vol_changed(self, v):
        db = v / 10.0
        self.lbl_vol.setText(f"{db:+.1f} dB")
        self.panel.set_track_prop(self.tid, "volume_db", db, commit=False)

    def _pan_changed(self, v):
        self.lbl_pan.setText(self._pan_text(v / 100))
        self.panel.set_track_prop(self.tid, "pan", v / 100, commit=False)

    def _pick_color(self):
        c = QColorDialog.getColor(QColor(self.panel.track(self.tid)["color"]), self, "Sáv színe")
        if c.isValid():
            self.panel.set_track_prop(self.tid, "color", c.name())


class Ruler(QWidget):
    def __init__(self, panel):
        super().__init__()
        self.panel = panel
        self.setFixedHeight(RULER_H)
        self.setMouseTracking(True)
        self._drag = None
        self.setToolTip(tip_html("Idővonal vonalzó|Kattintás: a lejátszófej odaugrik. Húzás: hurok (ismétlési) "
                                 "tartomány kijelölése - a 🔁 gombbal kapcsolod be. Fent ütemszám, lent perc:mp."))

    def paintEvent(self, ev):
        p = QPainter(self)
        P = self.panel
        p.fillRect(self.rect(), QColor("#1b1c22"))
        pps = P.pps
        t0 = P.x_off
        t1 = t0 + self.width() / pps
        prj = P.project
        # hurok
        lp = prj.get("loop") or [0, 0]
        if lp[1] > lp[0]:
            x0, x1 = (lp[0] - t0) * pps, (lp[1] - t0) * pps
            p.fillRect(QRectF(x0, 0, x1 - x0, self.height()), QColor(106, 77, 255, 110 if prj.get("loop_on") else 40))
        beat = 60.0 / max(20.0, prj["bpm"])
        bar = beat * prj["beats_per_bar"]
        # ütem rács
        step_bars = 1
        while bar * step_bars * pps < 60:
            step_bars *= 2
        p.setPen(QColor("#9a9bab"))
        f = p.font()
        f.setPointSize(8)
        p.setFont(f)
        b = int(t0 / bar)
        while b * bar <= t1:
            x = (b * bar - t0) * pps
            if b % step_bars == 0:
                p.drawLine(QPointF(x, 0), QPointF(x, 10))
                p.drawText(QPointF(x + 3, 10), str(b + 1))
            b += 1
        # idő felirat
        sec_step = 1
        for s in (1, 2, 5, 10, 15, 30, 60, 120):
            if s * pps >= 70:
                sec_step = s
                break
        s = int(t0 / sec_step) * sec_step
        p.setPen(QColor("#6f7080"))
        while s <= t1:
            x = (s - t0) * pps
            p.drawLine(QPointF(x, self.height() - 6), QPointF(x, self.height()))
            p.drawText(QPointF(x + 3, self.height() - 4), f"{int(s // 60)}:{int(s % 60):02d}")
            s += sec_step
        # lejátszófej
        x = (P.engine.position() - t0) * pps
        p.setPen(QPen(QColor("#facc15"), 2))
        p.drawLine(QPointF(x, 0), QPointF(x, self.height()))
        p.end()

    def mousePressEvent(self, ev):
        t = self.panel.x_to_t(ev.position().x())
        self._drag = (t, ev.position().x())

    def mouseMoveEvent(self, ev):
        if self._drag and abs(ev.position().x() - self._drag[1]) > 4:
            t0 = self.panel.snap(self._drag[0])
            t1 = self.panel.snap(self.panel.x_to_t(ev.position().x()))
            self.panel.project["loop"] = [min(t0, t1), max(t0, t1)]
            self.update()

    def mouseReleaseEvent(self, ev):
        if self._drag and abs(ev.position().x() - self._drag[1]) <= 4:
            self.panel.seek(self.panel.snap(self._drag[0]) if self.panel.snap_on() else self._drag[0])
        else:
            self.panel.engine.update_mix(self.panel.project)
            self.panel.autosave()
        self._drag = None


class TimelineCanvas(QWidget):
    EDGE = 7

    def __init__(self, panel):
        super().__init__()
        self.panel = panel
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._cache = None
        self._drag = None
        self.setToolTip("")

    def invalidate(self):
        self._cache = None
        self.update()

    def resizeEvent(self, ev):
        self._cache = None
        super().resizeEvent(ev)

    # ---------------------------------------------------- geometria
    def clip_rect(self, ti: int, c: dict) -> QRectF:
        P = self.panel
        x0 = (c["start"] - P.x_off) * P.pps
        w = c["length"] * P.pps
        return QRectF(x0, ti * TRACK_H + 4, w, TRACK_H - 8)

    def hit(self, pos):
        P = self.panel
        ti = int(pos.y() // TRACK_H)
        if ti < 0 or ti >= len(P.project["tracks"]):
            return None, None, None
        tr = P.project["tracks"][ti]
        for c in reversed(tr["clips"]):
            r = self.clip_rect(ti, c)
            if r.adjusted(-2, 0, 2, 0).contains(pos):
                x = pos.x()
                fi_x = r.left() + c["fade_in"] * P.pps
                fo_x = r.right() - c["fade_out"] * P.pps
                if pos.y() < r.top() + 14 and abs(x - fi_x) < 9:
                    return tr, c, "fade_in"
                if pos.y() < r.top() + 14 and abs(x - fo_x) < 9:
                    return tr, c, "fade_out"
                if x - r.left() < self.EDGE:
                    return tr, c, "left"
                if r.right() - x < self.EDGE:
                    return tr, c, "right"
                return tr, c, "move"
        return tr, None, None

    # ---------------------------------------------------- rajzolás
    def _render_static(self):
        P = self.panel
        pm = QPixmap(self.size() * self.devicePixelRatioF())
        pm.setDevicePixelRatio(self.devicePixelRatioF())
        pm.fill(QColor("#111217"))
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing, True)
        W = self.width()
        t0 = P.x_off
        t1 = t0 + W / P.pps
        prj = P.project
        beat = 60.0 / max(20.0, prj["bpm"])
        bar = beat * prj["beats_per_bar"]
        for ti, tr in enumerate(prj["tracks"]):
            y = ti * TRACK_H
            p.fillRect(QRectF(0, y, W, TRACK_H), QColor("#15161c" if ti % 2 else "#121318"))
            p.setPen(QColor("#22242c"))
            p.drawLine(QPointF(0, y + TRACK_H - 0.5), QPointF(W, y + TRACK_H - 0.5))
        # rács
        if beat * P.pps > 12:
            b = int(t0 / beat)
            while b * beat <= t1:
                x = (b * beat - t0) * P.pps
                is_bar = b % prj["beats_per_bar"] == 0
                p.setPen(QColor("#2b2d38" if is_bar else "#1c1d24"))
                p.drawLine(QPointF(x, 0), QPointF(x, self.height()))
                b += 1
        else:
            b = int(t0 / bar)
            while b * bar <= t1:
                x = (b * bar - t0) * P.pps
                p.setPen(QColor("#24262f"))
                p.drawLine(QPointF(x, 0), QPointF(x, self.height()))
                b += 1
        # klipek
        sel = P.selected
        for ti, tr in enumerate(prj["tracks"]):
            col = QColor(tr["color"])
            for c in tr["clips"]:
                r = self.clip_rect(ti, c)
                if r.right() < 0 or r.left() > W:
                    continue
                is_sel = sel and sel[1] == c["id"]
                bg = QColor(col)
                bg.setAlpha(70 if not c.get("muted") else 25)
                p.setPen(QPen(QColor("#ffffff") if is_sel else col.darker(110), 2 if is_sel else 1))
                p.setBrush(bg)
                p.drawRoundedRect(r, 5, 5)
                self._draw_wave(p, r, c, col if not c.get("muted") else QColor("#555"))
                # áttűnések
                p.setPen(QPen(QColor(255, 255, 255, 150), 1))
                if c["fade_in"] > 0:
                    p.drawLine(QPointF(r.left(), r.bottom()), QPointF(r.left() + c["fade_in"] * P.pps, r.top()))
                if c["fade_out"] > 0:
                    p.drawLine(QPointF(r.right() - c["fade_out"] * P.pps, r.top()), QPointF(r.right(), r.bottom()))
                p.setBrush(QColor("#ffffff"))
                p.setPen(Qt.NoPen)
                p.drawEllipse(QPointF(r.left() + c["fade_in"] * P.pps, r.top() + 5), 3.5, 3.5)
                p.drawEllipse(QPointF(r.right() - c["fade_out"] * P.pps, r.top() + 5), 3.5, 3.5)
                # felirat
                p.setPen(QColor("#ffffff"))
                f = p.font()
                f.setPointSize(8)
                f.setBold(True)
                p.setFont(f)
                label = c["name"]
                extra = []
                if abs(c.get("stretch", 1) - 1) > 1e-3:
                    extra.append(f"{c['stretch'] * 100:.0f}%")
                if abs(c.get("pitch", 0)) > 1e-3:
                    extra.append(f"{c['pitch']:+g} fh")
                if c.get("reverse"):
                    extra.append("⇄")
                if c.get("muted"):
                    extra.append("néma")
                if extra:
                    label += "  [" + ", ".join(extra) + "]"
                p.drawText(QRectF(r.left() + 12, r.top() + 2, max(10, r.width() - 24), 14), Qt.AlignLeft | Qt.AlignVCenter, label)
            # automáció
            if tr.get("auto_on"):
                self._draw_automation(p, ti, tr)
        p.end()
        self._cache = pm

    def _draw_wave(self, p, r: QRectF, c: dict, col: QColor):
        P = self.panel
        pk = P.engine_peaks(c)
        if pk is None:
            p.setPen(QColor(255, 255, 255, 90))
            p.drawText(r, Qt.AlignCenter, tr("betöltés…"))
            return
        W = self.width()
        x_a = max(0, int(r.left()))
        x_b = min(W, int(r.right()))
        if x_b <= x_a:
            return
        xs = np.arange(x_a, x_b + 1, dtype=np.float64)
        t = P.x_off + xs / P.pps
        src_t = c["offset"] + (t - c["start"])
        bins = np.clip((src_t * DAW_SR / 256).astype(np.int64), 0, len(pk) - 1)
        if len(bins) < 2:
            return
        lo = bins[:-1]
        hi = np.maximum(bins[1:], lo + 1)
        cum = np.maximum.reduceat(pk, np.clip(lo, 0, len(pk) - 1)) if len(pk) > 1 else pk[lo]
        amp = np.clip(cum * db2lin(c.get("gain_db", 0.0)), 0, 1.0)
        mid = r.center().y() + 6
        hh = (r.height() - 20) / 2
        wc = QColor(col).lighter(150)
        wc.setAlpha(200)
        p.setPen(QPen(wc, 1))
        lines = [QLineF(float(x), mid - a * hh, float(x), mid + a * hh) for x, a in zip(xs[:-1], amp)]
        p.drawLines(lines)

    def _auto_y(self, ti, db):
        top = ti * TRACK_H + 8
        h = TRACK_H - 16
        v = (min(6.0, max(-40.0, db)) + 40.0) / 46.0
        return top + (1 - v) * h

    def _auto_db(self, ti, y):
        top = ti * TRACK_H + 8
        h = TRACK_H - 16
        v = 1 - (y - top) / h
        return round(max(-40.0, min(6.0, v * 46.0 - 40.0)), 1)

    def _draw_automation(self, p, ti, tr):
        P = self.panel
        pts = sorted(tr["automation"])
        p.setPen(QPen(QColor("#facc15"), 2))
        if not pts:
            y = self._auto_y(ti, 0.0)
            p.setPen(QPen(QColor(250, 204, 21, 120), 1, Qt.DashLine))
            p.drawLine(QPointF(0, y), QPointF(self.width(), y))
            return
        path = QPainterPath()
        first = True
        xs = [(-1.0, pts[0][1])] + [((t - P.x_off) * P.pps, d) for t, d in pts] + [(self.width() + 1.0, pts[-1][1])]
        for x, d in xs:
            y = self._auto_y(ti, d)
            if first:
                path.moveTo(x, y)
                first = False
            else:
                path.lineTo(x, y)
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)
        p.setBrush(QColor("#facc15"))
        for t, d in pts:
            p.drawEllipse(QPointF((t - P.x_off) * P.pps, self._auto_y(ti, d)), 4, 4)

    def paintEvent(self, ev):
        if self._cache is None or self._cache.size() != self.size() * self.devicePixelRatioF():
            self._render_static()
        p = QPainter(self)
        p.drawPixmap(0, 0, self._cache)
        P = self.panel
        lp = P.project.get("loop") or [0, 0]
        if P.project.get("loop_on") and lp[1] > lp[0]:
            x0, x1 = (lp[0] - P.x_off) * P.pps, (lp[1] - P.x_off) * P.pps
            p.fillRect(QRectF(x0, 0, x1 - x0, self.height()), QColor(106, 77, 255, 28))
        x = (P.engine.position() - P.x_off) * P.pps
        p.setPen(QPen(QColor("#facc15"), 2))
        p.drawLine(QPointF(x, 0), QPointF(x, self.height()))
        if self._drag and self._drag.get("ghost_track") is not None:
            p.setPen(QPen(QColor("#ffffff"), 1, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawRect(QRectF(0, self._drag["ghost_track"] * TRACK_H + 1, self.width() - 1, TRACK_H - 2))
        p.end()

    # ---------------------------------------------------- egér
    def mousePressEvent(self, ev):
        self.setFocus()
        P = self.panel
        pos = ev.position()
        ti = int(pos.y() // TRACK_H)
        tracks = P.project["tracks"]
        if 0 <= ti < len(tracks) and tracks[ti].get("auto_on"):
            self._auto_press(ev, ti, tracks[ti])
            return
        tr, c, zone = self.hit(pos)
        if ev.button() == Qt.RightButton:
            if c:
                P.select(tr["id"], c["id"])
                P.clip_menu(tr["id"], c["id"], ev.globalPosition().toPoint())
            elif tr:
                P.lane_menu(tr["id"], P.x_to_t(pos.x()), ev.globalPosition().toPoint())
            return
        if c is None:
            P.select(tr["id"] if tr else None, None)
            P.seek(P.snap(P.x_to_t(pos.x())) if P.snap_on() else P.x_to_t(pos.x()))
            return
        P.select(tr["id"], c["id"])
        self._drag = {"zone": zone, "tid": tr["id"], "cid": c["id"], "x0": pos.x(), "orig": dict(c),
                      "before": P.snapshot(), "ti": ti, "ghost_track": None, "moved": False}

    def mouseMoveEvent(self, ev):
        P = self.panel
        pos = ev.position()
        if self._drag and "auto" in self._drag:
            self._auto_move(ev)
            return
        if not self._drag:
            ti = int(pos.y() // TRACK_H)
            tracks = P.project["tracks"]
            if 0 <= ti < len(tracks) and tracks[ti].get("auto_on"):
                self.setCursor(Qt.CrossCursor)
                return
            _tr, c, zone = self.hit(pos)
            self.setCursor({"left": Qt.SizeHorCursor, "right": Qt.SizeHorCursor, "fade_in": Qt.PointingHandCursor,
                            "fade_out": Qt.PointingHandCursor, "move": Qt.OpenHandCursor}.get(zone, Qt.ArrowCursor))
            return
        d = self._drag
        dt = (pos.x() - d["x0"]) / P.pps
        if abs(pos.x() - d["x0"]) > 2:
            d["moved"] = True
        c = P.clip(d["tid"], d["cid"])
        o = d["orig"]
        if c is None:
            return
        src_len = P.src_len(c)
        if d["zone"] == "move":
            ns = max(0.0, o["start"] + dt)
            c["start"] = P.snap(ns) if P.snap_on() else ns
            gt = int(pos.y() // TRACK_H)
            d["ghost_track"] = gt if (0 <= gt < len(P.project["tracks"]) and gt != d["ti"]) else None
            self.setCursor(Qt.ClosedHandCursor)
        elif d["zone"] == "left":
            ns = o["start"] + dt
            ns = P.snap(ns) if P.snap_on() else ns
            delta = ns - o["start"]
            delta = max(-o["offset"], min(delta, o["length"] - 0.05))
            c["start"] = o["start"] + delta
            c["offset"] = o["offset"] + delta
            c["length"] = o["length"] - delta
        elif d["zone"] == "right":
            ne = o["start"] + o["length"] + dt
            ne = P.snap(ne) if P.snap_on() else ne
            c["length"] = max(0.05, min(ne - o["start"], src_len - o["offset"]))
        elif d["zone"] == "fade_in":
            c["fade_in"] = round(max(0.0, min(o["fade_in"] + dt, c["length"] - c["fade_out"])), 3)
        elif d["zone"] == "fade_out":
            c["fade_out"] = round(max(0.0, min(o["fade_out"] - dt, c["length"] - c["fade_in"])), 3)
        self.invalidate()
        P.status(f"{c['name']}: kezdés {fmt_time(c['start'])} · hossz {c['length']:.2f} mp · "
                 f"beúsztatás {c['fade_in']:.2f} · kiúsztatás {c['fade_out']:.2f}")

    def mouseReleaseEvent(self, ev):
        P = self.panel
        d = self._drag
        self._drag = None
        if not d:
            return
        if "auto" in d:
            P.push_undo(d["before"])
            P.changed()
            return
        if not d["moved"]:
            return
        if d["zone"] == "move" and d["ghost_track"] is not None:
            P.move_clip_to_track(d["tid"], d["cid"], P.project["tracks"][d["ghost_track"]]["id"], record=False)
        P.push_undo(d["before"])
        P.changed()

    def mouseDoubleClickEvent(self, ev):
        tr, c, _ = self.hit(ev.position())
        if c:
            self.panel.clip_dialog(tr["id"], c["id"])

    def wheelEvent(self, ev):
        P = self.panel
        dy = ev.angleDelta().y()
        if ev.modifiers() & Qt.ControlModifier:
            t = P.x_to_t(ev.position().x())
            P.set_zoom(P.pps * (1.25 if dy > 0 else 0.8), anchor_t=t, anchor_x=ev.position().x())
            ev.accept()
        elif ev.modifiers() & Qt.ShiftModifier:
            P.hbar.setValue(P.hbar.value() - int(dy / 120 * 100 * 120 / P.pps))
            ev.accept()
        else:
            ev.ignore()

    def keyPressEvent(self, ev):
        P = self.panel
        k = ev.key()
        if k == Qt.Key_Space:
            P.toggle_play()
        elif k in (Qt.Key_Delete, Qt.Key_Backspace):
            P.delete_selected()
        elif k == Qt.Key_S and not ev.modifiers():
            P.split_selected()
        elif k == Qt.Key_Home:
            P.seek(0)
        elif k == Qt.Key_D and ev.modifiers() & Qt.ControlModifier:
            P.duplicate_selected()
        elif k == Qt.Key_C and ev.modifiers() & Qt.ControlModifier:
            P.copy_selected()
        elif k == Qt.Key_V and ev.modifiers() & Qt.ControlModifier:
            P.paste()
        elif k in (Qt.Key_Plus, Qt.Key_Equal):
            P.set_zoom(P.pps * 1.25)
        elif k == Qt.Key_Minus:
            P.set_zoom(P.pps * 0.8)
        else:
            super().keyPressEvent(ev)

    # ---------------------------------------------------- automáció
    def _auto_press(self, ev, ti, tr):
        P = self.panel
        pos = ev.position()
        pts = tr["automation"]
        near = None
        for i, (t, d) in enumerate(pts):
            x = (t - P.x_off) * P.pps
            y = self._auto_y(ti, d)
            if abs(x - pos.x()) < 8 and abs(y - pos.y()) < 8:
                near = i
                break
        before = P.snapshot()
        if ev.button() == Qt.RightButton:
            if near is not None:
                pts.pop(near)
                P.push_undo(before)
                P.changed()
            return
        if near is None:
            t = max(0.0, P.x_to_t(pos.x()))
            pts.append([round(t, 3), self._auto_db(ti, pos.y())])
            pts.sort()
            near = next(i for i, p_ in enumerate(pts) if abs(p_[0] - round(t, 3)) < 1e-6)
        self._drag = {"auto": True, "ti": ti, "tid": tr["id"], "idx": near, "before": before}
        self.invalidate()

    def _auto_move(self, ev):
        P = self.panel
        d = self._drag
        tr = P.track(d["tid"])
        pos = ev.position()
        pt = tr["automation"][d["idx"]]
        pt[0] = round(max(0.0, P.x_to_t(pos.x())), 3)
        pt[1] = self._auto_db(d["ti"], pos.y())
        P.status(f"Automáció pont: {fmt_time(pt[0])} · {pt[1]:+.1f} dB")
        self.invalidate()


class ParamSlider(QWidget):
    changed = Signal(str, float)

    def __init__(self, key, label, lo, hi, step, value):
        super().__init__()
        self.key, self.lo, self.hi, self.step = key, lo, hi, step
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.sl = QSlider(Qt.Horizontal)
        self.n = int(round((hi - lo) / step))
        self.sl.setRange(0, self.n)
        self.sp = QDoubleSpinBox()
        self.sp.setRange(lo, hi)
        self.sp.setSingleStep(step)
        self.sp.setDecimals(3 if step < 0.01 else 2)
        self.sp.setFixedWidth(100)
        self._set(value)
        self.sl.valueChanged.connect(lambda v: self._from_slider(v))
        self.sp.valueChanged.connect(lambda v: self._from_spin(v))
        lay.addWidget(self.sl, 1)
        lay.addWidget(self.sp)

    def _set(self, v):
        self.sl.blockSignals(True)
        self.sp.blockSignals(True)
        self.sl.setValue(int(round((v - self.lo) / self.step)))
        self.sp.setValue(v)
        self.sl.blockSignals(False)
        self.sp.blockSignals(False)

    def _from_slider(self, i):
        v = self.lo + i * self.step
        self.sp.blockSignals(True)
        self.sp.setValue(v)
        self.sp.blockSignals(False)
        self.changed.emit(self.key, float(v))

    def _from_spin(self, v):
        self.sl.blockSignals(True)
        self.sl.setValue(int(round((v - self.lo) / self.step)))
        self.sl.blockSignals(False)
        self.changed.emit(self.key, float(v))


class FxDialog(QDialog):
    """Effektlánc szerkesztő (sávra vagy a mesterre). Változások azonnal hallhatók."""

    def __init__(self, parent, title: str, chain: list, on_change):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(820, 520)
        self.chain = chain
        self.on_change = on_change
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.on_change)
        lay = QHBoxLayout(self)
        left = QVBoxLayout()
        self.lst = QListWidget()
        self.lst.setToolTip(tip_html("Effektlánc|Fentről lefelé sorban dolgozzák fel a hangot. A pipával "
                                     "ideiglenesen kikapcsolhatod az effektet. Kattints egyre a beállításaihoz."))
        self.lst.currentRowChanged.connect(self._show_params)
        self.lst.itemChanged.connect(self._item_toggled)
        left.addWidget(self.lst, 1)
        r = QHBoxLayout()
        self.cb_add = QComboBox()
        self.cb_add.addItem("Effekt hozzáadása…", None)
        for k, (name, _c, _p) in FX_DEFS.items():
            self.cb_add.addItem(name, k)
            self.cb_add.setItemData(self.cb_add.count() - 1, FX_TIPS.get(k, ""), Qt.ToolTipRole)
        self.cb_add.setToolTip(tip_html("Új effekt|Válassz egyet a listából: a lánc végére kerül."))
        self.cb_add.activated.connect(self._add)
        r.addWidget(self.cb_add, 1)
        left.addLayout(r)
        r = QHBoxLayout()
        for txt, fn, tip in (("▲", lambda: self._move(-1), "Feljebb|Előrébb kerül a láncban."),
                             ("▼", lambda: self._move(1), "Lejjebb|Hátrébb kerül a láncban."),
                             ("Törlés", self._remove, "Törlés|A kijelölt effekt eltávolítása.")):
            b = QPushButton(txt)
            b.setToolTip(tip_html(tip))
            b.clicked.connect(fn)
            r.addWidget(b)
        left.addLayout(r)
        r = QHBoxLayout()
        self.cb_preset = QComboBox()
        self.cb_preset.addItem("Kész effektlánc betöltése…", None)
        for k in FX_PRESETS:
            self.cb_preset.addItem(k, k)
        self.cb_preset.setToolTip(tip_html("Kész láncok|Bevált beállítások énekre, dobra, basszusra, mesterre. "
                                           "Betöltéskor lecseréli a jelenlegi láncot (visszavonható)."))
        self.cb_preset.activated.connect(self._preset)
        r.addWidget(self.cb_preset, 1)
        left.addLayout(r)
        lay.addLayout(left, 2)
        right = QVBoxLayout()
        self.lbl_fx = QLabel("Válassz egy effektet a bal oldali listából.")
        self.lbl_fx.setObjectName("h1")
        self.lbl_desc = QLabel("")
        self.lbl_desc.setWordWrap(True)
        self.lbl_desc.setObjectName("dim")
        right.addWidget(self.lbl_fx)
        right.addWidget(self.lbl_desc)
        self.form_host = QWidget()
        self.form = QFormLayout(self.form_host)
        right.addWidget(self.form_host)
        right.addStretch()
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.button(QDialogButtonBox.Close).setText("Bezárás")
        bb.rejected.connect(self.accept)
        right.addWidget(bb)
        lay.addLayout(right, 3)
        self._refresh()
        if self.chain:
            self.lst.setCurrentRow(0)

    def _refresh(self, keep=None):
        self.lst.blockSignals(True)
        self.lst.clear()
        for f in self.chain:
            it = QListWidgetItem(FX_DEFS[f["type"]][0])
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if f.get("enabled", True) else Qt.Unchecked)
            it.setToolTip(FX_TIPS.get(f["type"], ""))
            self.lst.addItem(it)
        self.lst.blockSignals(False)
        if keep is not None and 0 <= keep < len(self.chain):
            self.lst.setCurrentRow(keep)
        elif not self.chain:
            self._show_params(-1)

    def _changed(self):
        self._timer.start(250)

    def _item_toggled(self, it):
        i = self.lst.row(it)
        self.chain[i]["enabled"] = it.checkState() == Qt.Checked
        self._changed()

    def _add(self, idx):
        k = self.cb_add.itemData(idx)
        self.cb_add.setCurrentIndex(0)
        if not k:
            return
        self.chain.append(new_fx(k))
        self._refresh(len(self.chain) - 1)
        self._changed()

    def _remove(self):
        i = self.lst.currentRow()
        if i >= 0:
            self.chain.pop(i)
            self._refresh(min(i, len(self.chain) - 1))
            self._changed()

    def _move(self, d):
        i = self.lst.currentRow()
        j = i + d
        if 0 <= i < len(self.chain) and 0 <= j < len(self.chain):
            self.chain[i], self.chain[j] = self.chain[j], self.chain[i]
            self._refresh(j)
            self._changed()

    def _preset(self, idx):
        k = self.cb_preset.itemData(idx)
        self.cb_preset.setCurrentIndex(0)
        if not k:
            return
        self.chain[:] = [new_fx(kind, params) for kind, params in FX_PRESETS[k]]
        self._refresh(0)
        self._changed()

    def _show_params(self, i):
        while self.form.rowCount():
            self.form.removeRow(0)
        if i < 0 or i >= len(self.chain):
            self.lbl_fx.setText("Nincs kijelölt effekt")
            self.lbl_desc.setText("Adj hozzá egyet a bal alsó listából, vagy tölts be egy kész láncot.")
            return
        f = self.chain[i]
        name, _cls, params = FX_DEFS[f["type"]]
        self.lbl_fx.setText(name)
        self.lbl_desc.setText(FX_TIPS.get(f["type"], ""))
        for key, lbl, lo, hi, step, default in params:
            w = ParamSlider(key, lbl, lo, hi, step, f["params"].get(key, default))
            w.setToolTip(tip_html(f"{lbl}|Tartomány: {lo} – {hi}. Alapérték: {default}."))
            w.changed.connect(lambda k, v, f=f: (f["params"].__setitem__(k, v), self._changed()))
            self.form.addRow(lbl + ":", w)


class ClipDialog(QDialog):
    def __init__(self, parent, panel, clip: dict):
        super().__init__(parent)
        self.setWindowTitle("Klip tulajdonságai")
        self.panel = panel
        self.clip = clip
        f = QFormLayout(self)
        self.ed_name = QLineEdit(clip["name"])
        self.sp_start = self._dsp(0, 36000, 0.01, clip["start"], 3)
        self.sp_offset = self._dsp(0, 36000, 0.01, clip["offset"], 3)
        self.sp_len = self._dsp(0.05, 36000, 0.01, clip["length"], 3)
        self.sp_gain = self._dsp(-40, 24, 0.5, clip["gain_db"], 1)
        self.sp_fi = self._dsp(0, 600, 0.05, clip["fade_in"], 2)
        self.sp_fo = self._dsp(0, 600, 0.05, clip["fade_out"], 2)
        self.sp_tempo = self._dsp(25, 400, 1, clip.get("stretch", 1.0) * 100, 1)
        self.sp_pitch = self._dsp(-24, 24, 0.5, clip.get("pitch", 0.0), 1)
        self.chk_rev = QCheckBox("Visszafelé lejátszás")
        self.chk_rev.setChecked(bool(clip.get("reverse")))
        self.chk_mute = QCheckBox("Klip némítása")
        self.chk_mute.setChecked(bool(clip.get("muted")))
        rows = [
            ("Név:", self.ed_name, "Név|A klip felirata az idővonalon."),
            ("Kezdés (mp):", self.sp_start, "Kezdés|Hol kezdődik a klip az idővonalon."),
            ("Eltolás a forrásban (mp):", self.sp_offset, "Eltolás|A forrásfájl hányadik másodpercétől szól (a bal szél levágása)."),
            ("Hossz (mp):", self.sp_len, "Hossz|Mennyi szól a klipből."),
            ("Hangerő (dB):", self.sp_gain, "Klip hangerő|A klip saját erősítése, a sáv hangerejétől függetlenül."),
            ("Beúsztatás (mp):", self.sp_fi, "Beúsztatás|Ennyi idő alatt hangosodik fel a klip eleje."),
            ("Kiúsztatás (mp):", self.sp_fo, "Kiúsztatás|Ennyi idő alatt halkul el a klip vége."),
            ("Tempó (%):", self.sp_tempo, "Tempó|100% = eredeti. Nagyobb = gyorsabb (rövidebb), kisebb = lassabb. "
                                          "A hangmagasság nem változik."),
            ("Hangmagasság (félhang):", self.sp_pitch, "Hangmagasság|Transzponálás félhangokban (+12 = egy oktávval feljebb). "
                                                       "A tempó nem változik."),
        ]
        for lbl, w, tip in rows:
            w.setToolTip(tip_html(tip))
            f.addRow(lbl, w)
        self.chk_rev.setToolTip(tip_html("Visszafelé|A klip hangja visszafelé szól."))
        self.chk_mute.setToolTip(tip_html("Némítás|A klip megmarad, de nem szól."))
        f.addRow("", self.chk_rev)
        f.addRow("", self.chk_mute)
        b = QPushButton("Tempó felismerése és igazítás a projekt tempójához")
        b.setToolTip(tip_html("Tempó igazítás|Megbecsüli a klip tempóját (BPM), és úgy állítja a Tempó %-ot, "
                              "hogy a projekt tempójához illeszkedjen. A becslés tájékoztató jellegű."))
        b.clicked.connect(self._match_tempo)
        f.addRow("", b)
        self.lbl_info = QLabel("")
        self.lbl_info.setObjectName("dim")
        f.addRow("", self.lbl_info)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Alkalmaz")
        bb.button(QDialogButtonBox.Cancel).setText("Mégse")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        f.addRow(bb)

    @staticmethod
    def _dsp(lo, hi, step, val, dec):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setDecimals(dec)
        s.setValue(val)
        return s

    def _match_tempo(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            raw = self.panel.engine.cache.get_raw(self.clip["src"])
            bpm = estimate_bpm(raw)
        finally:
            QApplication.restoreOverrideCursor()
        target = self.panel.project["bpm"]
        self.sp_tempo.setValue(round(target / bpm * 100, 1))
        self.lbl_info.setText(f"Becsült tempó: {bpm} BPM → projekt: {target} BPM")

    def apply(self):
        c = self.clip
        old_stretch = c.get("stretch", 1.0)
        new_stretch = self.sp_tempo.value() / 100.0
        c["name"] = self.ed_name.text().strip() or c["name"]
        c["start"] = self.sp_start.value()
        c["offset"] = self.sp_offset.value()
        c["length"] = self.sp_len.value()
        if abs(new_stretch - old_stretch) > 1e-4:
            ratio = old_stretch / new_stretch
            c["offset"] *= ratio
            c["length"] *= ratio
        c["stretch"] = new_stretch
        c["gain_db"] = self.sp_gain.value()
        c["pitch"] = self.sp_pitch.value()
        c["reverse"] = self.chk_rev.isChecked()
        c["muted"] = self.chk_mute.isChecked()
        c["fade_in"] = min(self.sp_fi.value(), c["length"])
        c["fade_out"] = min(self.sp_fo.value(), c["length"] - c["fade_in"])


class DawPanel(QWidget):
    """A DAW fül: projekt, sávok, idővonal, keverés, export."""

    def __init__(self, main):
        super().__init__()
        self.main = main
        self.project = new_project()
        self.engine = DawEngine()
        self.engine.rendered.connect(lambda tid: (self.canvas.invalidate(), self._render_status()))
        self.engine.render_failed.connect(lambda tid, m: self.status(f"Renderelési hiba: {m}"))
        self.pps = 40.0
        self.x_off = 0.0
        self.selected = None          # (track_id, clip_id)
        self.clipboard = None
        self.undo_stack: list[str] = []
        self.redo_stack: list[str] = []
        self.project_path: Path | None = None
        self.headers: dict[str, TrackHeader] = {}
        self._peak_jobs: set = set()
        DAW_DIR.mkdir(exist_ok=True)
        self._build()
        last = DAW_DIR / "_utolso.y2daw"
        if last.is_file():
            try:
                self.project = json.loads(last.read_text(encoding="utf-8"))
            except Exception:
                pass
        self.rebuild()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(40)

    # ---------------------------------------------------------- felépítés
    def _build(self):
        v = QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(4)
        # 1. eszközsor: projekt
        r1 = QHBoxLayout()
        self.btn = {}

        def btn(key, txt, fn, tip, lay=r1, obj=None):
            b = QPushButton(txt)
            b.clicked.connect(fn)
            b.setToolTip(tip_html(tip))
            if obj:
                b.setObjectName(obj)
            lay.addWidget(b)
            self.btn[key] = b
            return b
        btn("new", "🧹 Tiszta lap", self.new_project, "Tiszta lap|Minden sáv és klip eltávolítása a DAW-ból (a "
            "hangfájlok megmaradnak a lemezen). Ctrl+Z-vel visszahozható.")
        btn("open", "DAW-fájl…", self.open_project, "DAW-fájl megnyitása|Csak a sávszerkesztő tartalmát tölti be egy "
            ".y2daw fájlból. A teljes munkát (minden fül) a felső Projekt sávval mentsd és nyisd meg.")
        btn("save", "DAW-fájl mentése", self.save_project, "DAW-fájl mentése|Csak a sávszerkesztő tartalmát menti "
            ".y2daw fájlba. A teljes munkát (minden fül) a felső Projekt sáv 💾 gombja menti.")
        btn("saveas", "…másként", lambda: self.save_project(True), "DAW-fájl mentése másként|Csak a sávok, új néven.")
        r1.addSpacing(12)
        btn("import", "＋ Hangfájl…", self.import_files,
            "Hangfájl importálása|WAV, MP3, FLAC, OGG fájl(ok) betöltése új sáv(ok)ra, a lejátszófej helyére.")
        btn("lib", "＋ Könyvtárból", self.import_from_library,
            "Könyvtárból|A Könyvtár fülön kijelölt generált dal betöltése új sávra.")
        btn("sep", "✂ Sávokra bontás…", self.separate_file,
            "Sávokra bontás|Egy dalt 4 sávra bont (ének, dob, basszus, egyéb) a HTDemucs modellel, és mindegyiket "
            "külön sávra teszi. Egy klipre jobb klikkel is elérhető.", obj="primary")
        btn("track", "＋ Üres sáv", self.add_empty_track, "Üres sáv|Új, üres sáv hozzáadása.")
        r1.addSpacing(12)
        btn("undo", "↶", self.undo, "Visszavonás (Ctrl+Z)|Az utolsó szerkesztés visszavonása.")
        btn("redo", "↷", self.redo, "Újra (Ctrl+Y)|A visszavont lépés megismétlése.")
        r1.addStretch()
        btn("export", "⤓ Keverés exportálása…", self.export_mix,
            "Keverés exportálása|A teljes dal (vagy a hurok tartomány) mentése WAV vagy MP3 fájlba, minden effekttel, "
            "és felvétele a Könyvtárba.", obj="primary")
        btn("stems", "⤓ Sávok exportálása…", self.export_stems,
            "Sávok exportálása|Minden sáv külön fájlba, a saját effektjeivel és hangerejével.")
        v.addLayout(r1)
        # 2. eszközsor: transzport
        r2 = QHBoxLayout()
        btn("home", "⏮", lambda: self.seek(0), "Elejére (Home)|A lejátszófej a dal elejére ugrik.", lay=r2)
        self.btn_play = btn("play", "▶", self.toggle_play, "Lejátszás / szünet (Szóköz)|A teljes keverés lejátszása "
                            "a lejátszófejtől.", lay=r2, obj="primary")
        btn("stop", "■", self.stop, "Leállítás|Megáll, és visszaugrik oda, ahonnan indult.", lay=r2)
        self.btn_loop = btn("loop", "🔁", self.toggle_loop, "Hurok|Bekapcsolva a vonalzón kijelölt tartományt ismétli. "
                            "A tartományt a vonalzón húzással jelölöd ki.", lay=r2)
        self.btn_loop.setCheckable(True)
        for k in ("home", "play", "stop", "loop"):
            self.btn[k].setFixedSize(46, 34)
        self.lbl_pos = QLabel("0:00.00")
        self.lbl_pos.setObjectName("h1")
        self.lbl_pos.setMinimumWidth(170)
        self.lbl_pos.setToolTip(tip_html("Pozíció|Perc:másodperc és ütem.ütemrész."))
        r2.addWidget(self.lbl_pos)
        r2.addWidget(QLabel("Tempó:"))
        self.sp_bpm = QDoubleSpinBox()
        self.sp_bpm.setRange(30, 300)
        self.sp_bpm.setDecimals(1)
        self.sp_bpm.setSuffix(" BPM")
        self.sp_bpm.setToolTip(tip_html("Projekt tempó|A rács és a vonalzó ütemei ehhez igazodnak. A klipeket "
                                        "a klip tulajdonságaiban lehet ehhez igazítani."))
        self.sp_bpm.valueChanged.connect(self._bpm_changed)
        r2.addWidget(self.sp_bpm)
        btn("bpm", "Felismerés", self.detect_bpm, "Tempó felismerése|A kijelölt (vagy az első) klip tempóját "
            "megbecsüli, és beállítja projekt tempónak.", lay=r2)
        r2.addWidget(QLabel("Rács:"))
        self.cb_snap = QComboBox()
        for txt, val in (("ki", 0), ("ütem", 4), ("1/2 ütem", 2), ("negyed", 1), ("nyolcad", 0.5), ("tizenhatod", 0.25)):
            self.cb_snap.addItem(txt, val)
        self.cb_snap.setCurrentIndex(3)
        self.cb_snap.setToolTip(tip_html("Rácshoz igazítás|Mozgatáskor és vágáskor a klipek a kiválasztott "
                                         "rácsosztáshoz ugranak. 'ki' = szabad mozgatás. Alt nélkül is mindig él."))
        r2.addWidget(self.cb_snap)
        btn("zout", "−", lambda: self.set_zoom(self.pps * 0.8), "Kicsinyítés (−, Ctrl+görgő)|Többet látsz az idővonalból.", lay=r2)
        btn("zin", "+", lambda: self.set_zoom(self.pps * 1.25), "Nagyítás (+, Ctrl+görgő)|Részletesebb nézet a pontos vágáshoz.", lay=r2)
        btn("zfit", "Mind", self.zoom_fit, "Teljes nézet|Az egész projekt elfér a képernyőn.", lay=r2)
        r2.addStretch()
        r2.addWidget(QLabel("Mester:"))
        self.sl_master = QSlider(Qt.Horizontal)
        self.sl_master.setRange(-400, 120)
        self.sl_master.setFixedWidth(130)
        self.sl_master.setToolTip(tip_html("Mester hangerő|A teljes keverés hangereje (-40 .. +12 dB)."))
        self.sl_master.valueChanged.connect(self._master_changed)
        self.sl_master.sliderReleased.connect(self.autosave)
        r2.addWidget(self.sl_master)
        self.lbl_master = QLabel("0.0 dB")
        self.lbl_master.setFixedWidth(60)
        r2.addWidget(self.lbl_master)
        self.btn_mfx = btn("mfx", "Mester FX", lambda: self.open_fx(None),
                           "Mester effektek|A teljes keverésre ható effektlánc (pl. EQ, kompresszor, limiter). "
                           "Valós időben hallható.", lay=r2)
        mv = QVBoxLayout()
        mv.setSpacing(2)
        self.meter_l = QProgressBar()
        self.meter_r = QProgressBar()
        for m in (self.meter_l, self.meter_r):
            m.setRange(0, 100)
            m.setTextVisible(False)
            m.setFixedSize(110, 7)
            m.setObjectName("meter")
            m.setToolTip(tip_html("Mester szintmérő|Bal és jobb csatorna. Ha gyakran a végén jár, csökkentsd a "
                                  "hangerőt vagy tegyél Limitert a mesterre."))
            mv.addWidget(m)
        r2.addLayout(mv)
        v.addLayout(r2)
        # vonalzó
        rr = QHBoxLayout()
        rr.setSpacing(0)
        corner = QLabel("  Sávok")
        corner.setFixedWidth(HEADER_W)
        corner.setObjectName("dim")
        rr.addWidget(corner)
        self.ruler = Ruler(self)
        rr.addWidget(self.ruler, 1)
        v.addLayout(rr)
        # sávok + vászon
        self.body = QWidget()
        bl = QHBoxLayout(self.body)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(0)
        self.hdr_host = QWidget()
        self.hdr_lay = QVBoxLayout(self.hdr_host)
        self.hdr_lay.setContentsMargins(0, 0, 0, 0)
        self.hdr_lay.setSpacing(0)
        self.hdr_host.setFixedWidth(HEADER_W)
        bl.addWidget(self.hdr_host, 0, Qt.AlignTop)
        self.canvas = TimelineCanvas(self)
        self.canvas.setToolTip(tip_html(
            "Idővonal|Klip mozgatása: húzás (másik sávra is). Vágás széleken: a klip bal/jobb szélét húzd. "
            "Beúsztatás/kiúsztatás: a klip tetején a fehér pöttyöket húzd. Dupla kattintás: klip tulajdonságai "
            "(tempó, hangmagasság). Jobb klikk: menü (vágás, sávokra bontás, MIDI). Kattintás üres helyre: "
            "lejátszófej. Billentyűk: Szóköz, S = vágás, Del = törlés, Ctrl+D = duplikálás, Ctrl+C/V, Ctrl+görgő = nagyítás, "
            "Shift+görgő = vízszintes görgetés."))
        bl.addWidget(self.canvas, 1, Qt.AlignTop)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.body)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        v.addWidget(self.scroll, 1)
        self.empty_hint = QLabel("Még nincs sáv. Kezdd a fenti gombokkal: ＋ Hangfájl…, ＋ Könyvtárból, vagy ✂ Sávokra bontás…")
        self.empty_hint.setAlignment(Qt.AlignCenter)
        self.empty_hint.setObjectName("dim")
        v.addWidget(self.empty_hint)
        hr = QHBoxLayout()
        sp = QLabel("")
        sp.setFixedWidth(HEADER_W)
        hr.addWidget(sp)
        self.hbar = QScrollBar(Qt.Horizontal)
        self.hbar.valueChanged.connect(self._hscroll)
        self.hbar.setToolTip(tip_html("Vízszintes görgetés|Mozgás az idővonalon (Shift+görgővel is)."))
        hr.addWidget(self.hbar, 1)
        v.addLayout(hr)
        self.lbl_status = QLabel("")
        self.lbl_status.setObjectName("dim")
        self.lbl_status.setToolTip(tip_html("Állapot|Mi történik épp: számolás (effektek, tempó), kijelölt klip "
                                            "adatai, sávbontás és MIDI állapota."))
        v.addWidget(self.lbl_status)
        for seq, fn in (("Ctrl+Z", self.undo), ("Ctrl+Y", self.redo), ("Ctrl+Shift+Z", self.redo)):
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)

    # ---------------------------------------------------------- segéd
    def status(self, txt: str):
        self.lbl_status.setText(txt)

    def _render_status(self):
        if self.engine.is_rendering():
            self.status(f"Számolás… ({len(self.engine.pending)} sáv frissül - effektek, tempó, hangmagasság)")
        else:
            self.status("Kész.")

    def track(self, tid):
        return next((t for t in self.project["tracks"] if t["id"] == tid), None)

    def clip(self, tid, cid):
        t = self.track(tid)
        return next((c for c in t["clips"] if c["id"] == cid), None) if t else None

    def src_len(self, c):
        try:
            return self.engine.cache.length(c["src"], c.get("stretch", 1.0), c.get("pitch", 0.0), c.get("reverse", False))
        except Exception:
            return c["offset"] + c["length"]

    def engine_peaks(self, c):
        key = (c["src"], round(c.get("stretch", 1.0), 4), round(c.get("pitch", 0.0), 3), bool(c.get("reverse")))
        pk = self.engine.cache.peaks.get(key)
        if pk is not None:
            return pk
        if key not in self._peak_jobs:
            self._peak_jobs.add(key)

            def job():
                try:
                    self.engine.cache.get_peaks(*key)
                finally:
                    self._peak_jobs.discard(key)
                    self.engine.rendered.emit("")
            self.engine.pool.submit(job)
        return None

    def x_to_t(self, x: float) -> float:
        return self.x_off + x / self.pps

    def snap_on(self) -> bool:
        return bool(self.cb_snap.currentData())

    def snap(self, t: float) -> float:
        div = self.cb_snap.currentData()
        if not div:
            return t
        g = 60.0 / self.project["bpm"] * div
        return max(0.0, round(t / g) * g)

    def snapshot(self) -> str:
        return json.dumps(self.project, ensure_ascii=False)

    def push_undo(self, before: str):
        if self.undo_stack and self.undo_stack[-1] == before:
            return
        self.undo_stack.append(before)
        if len(self.undo_stack) > 200:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def record(self):
        """Módosítás előtt hívd: visszavonási pont."""
        self.push_undo(self.snapshot())

    def changed(self, rebuild_headers=False):
        if rebuild_headers:
            self.rebuild()
        else:
            self._update_geometry()
            self.canvas.invalidate()
            self.ruler.update()
            self.engine.sync(self.project)
            self._render_status()
        self.autosave()

    def autosave(self):
        try:
            (DAW_DIR / "_utolso.y2daw").write_text(json.dumps(self.project, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def rebuild(self):
        for h in self.headers.values():
            h.setParent(None)
            h.deleteLater()
        self.headers.clear()
        for t in self.project["tracks"]:
            h = TrackHeader(self, t)
            self.headers[t["id"]] = h
            self.hdr_lay.addWidget(h)
            install_tip_filter(h)
        self.sp_bpm.blockSignals(True)
        self.sp_bpm.setValue(self.project["bpm"])
        self.sp_bpm.blockSignals(False)
        self.sl_master.blockSignals(True)
        self.sl_master.setValue(int(self.project["master_db"] * 10))
        self.sl_master.blockSignals(False)
        self.lbl_master.setText(f"{self.project['master_db']:+.1f} dB")
        self.btn_loop.blockSignals(True)
        self.btn_loop.setChecked(bool(self.project.get("loop_on")))
        self.btn_loop.blockSignals(False)
        n = sum(1 for f in self.project["master_fx"] if f.get("enabled", True))
        self.btn_mfx.setText(f"Mester FX {n}" if n else "Mester FX")
        self._update_geometry()
        self.canvas.invalidate()
        self.ruler.update()
        self.engine.sync(self.project)
        self._render_status()
        if hasattr(self.main, "lbl_daw_title"):
            pass

    def _update_geometry(self):
        n = len(self.project["tracks"])
        h = max(1, n * TRACK_H)
        self.canvas.setFixedHeight(h)
        self.hdr_host.setFixedHeight(h)
        self.empty_hint.setVisible(n == 0)
        total = project_length(self.project) + 60
        self.hbar.blockSignals(True)
        self.hbar.setRange(0, int(total * 100))
        self.hbar.setPageStep(int(self.canvas.width() / self.pps * 100))
        self.hbar.setSingleStep(max(1, int(50 / self.pps * 100)))
        self.hbar.setValue(int(self.x_off * 100))
        self.hbar.blockSignals(False)

    def _hscroll(self, v):
        self.x_off = v / 100.0
        self.canvas.invalidate()
        self.ruler.update()

    def set_zoom(self, pps, anchor_t=None, anchor_x=None):
        pps = max(2.0, min(800.0, pps))
        if anchor_t is None:
            anchor_t = self.engine.position()
            anchor_x = (anchor_t - self.x_off) * self.pps
        self.pps = pps
        self.x_off = max(0.0, anchor_t - anchor_x / pps)
        self._update_geometry()
        self.canvas.invalidate()
        self.ruler.update()

    def zoom_fit(self):
        L = max(10.0, project_length(self.project) + 2)
        self.pps = max(2.0, self.canvas.width() / L)
        self.x_off = 0.0
        self._update_geometry()
        self.canvas.invalidate()
        self.ruler.update()

    def _tick(self):
        pos = self.engine.position()
        bpm = self.project["bpm"]
        beat = pos / (60.0 / bpm)
        bpb = self.project["beats_per_bar"]
        self.lbl_pos.setText(f"{fmt_time(pos)}   {int(beat // bpb) + 1}.{int(beat % bpb) + 1}")
        if self.engine.playing:
            self.btn_play.setText("⏸")
            vis = self.canvas.width() / self.pps
            if pos > self.x_off + vis * 0.92 or pos < self.x_off:
                self.x_off = max(0.0, pos - vis * 0.1)
                self._update_geometry()
                self.canvas.invalidate()
            self.canvas.update()
            self.ruler.update()
            for tid, h in self.headers.items():
                h.meter.setValue(int(min(1.0, self.engine.peaks.get(tid, 0.0)) * 100))
            l, r = self.engine.master_peak
            self.meter_l.setValue(int(min(1.0, l) * 100))
            self.meter_r.setValue(int(min(1.0, r) * 100))
        else:
            if self.btn_play.text() != "▶":
                self.btn_play.setText("▶")
                for h in self.headers.values():
                    h.meter.setValue(0)
                self.meter_l.setValue(0)
                self.meter_r.setValue(0)
                self.canvas.update()
                self.ruler.update()
        if self.engine.error:
            self.status("Hanghiba: " + self.engine.error)
            self.engine.error = ""

    # ---------------------------------------------------------- transzport
    def toggle_play(self):
        if self.engine.playing:
            self.engine.pause()
            return
        if not self.project["tracks"]:
            return
        if self.main.player and self.main.player.playbackState() == QMediaPlayer.PlayingState:
            self.main.player.pause()
        self._play_from = self.engine.position()
        err = self.engine.play()
        if err:
            QMessageBox.warning(self, "Lejátszás", err)

    def stop(self):
        self.engine.pause()
        self.seek(getattr(self, "_play_from", 0.0))

    def seek(self, t: float):
        self.engine.seek(t)
        self.canvas.update()
        self.ruler.update()

    def toggle_loop(self, on):
        self.project["loop_on"] = bool(on)
        lp = self.project.get("loop") or [0, 0]
        if on and lp[1] - lp[0] < 0.1:
            beat = 60 / self.project["bpm"]
            s = self.snap(self.engine.position())
            self.project["loop"] = [s, s + beat * self.project["beats_per_bar"] * 4]
        self.engine.update_mix(self.project)
        self.canvas.update()
        self.ruler.update()
        self.autosave()

    def _bpm_changed(self, v):
        self.project["bpm"] = float(v)
        self.canvas.invalidate()
        self.ruler.update()
        self.autosave()

    def _master_changed(self, v):
        self.project["master_db"] = v / 10.0
        self.lbl_master.setText(f"{v / 10:+.1f} dB")
        self.engine.update_mix(self.project)

    def detect_bpm(self):
        c = None
        if self.selected and self.selected[1]:
            c = self.clip(*self.selected)
        if c is None:
            for t in self.project["tracks"]:
                if t["clips"]:
                    c = t["clips"][0]
                    break
        if c is None:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            bpm = estimate_bpm(self.engine.cache.get_raw(c["src"]))
        finally:
            QApplication.restoreOverrideCursor()
        self.record()
        self.sp_bpm.setValue(bpm)
        self.status(f"Becsült tempó: {bpm} BPM ({c['name']}) - ha a fele vagy duplája a valódinak, írd át kézzel.")

    # ---------------------------------------------------------- sáv műveletek
    def set_track_prop(self, tid, prop, val, commit=True):
        t = self.track(tid)
        if t is None or t.get(prop) == val:
            return
        if commit:
            self.record()
        t[prop] = val
        if prop in ("volume_db", "pan", "mute", "solo"):
            self.engine.update_mix(self.project)
            if commit:
                self.autosave()
            return
        if prop == "color":
            self.changed(rebuild_headers=True)
            return
        if prop == "auto_on":
            self.engine.update_mix(self.project)
            self.canvas.invalidate()
            self.autosave()
            return
        self.changed()

    def commit_mix(self):
        self.autosave()

    def add_track(self, name, color=None, index=None) -> dict:
        t = new_track(name, color or TRACK_COLORS[len(self.project["tracks"]) % len(TRACK_COLORS)])
        if index is None:
            self.project["tracks"].append(t)
        else:
            self.project["tracks"].insert(index, t)
        return t

    def add_empty_track(self):
        self.record()
        self.add_track(f"Sáv {len(self.project['tracks']) + 1}")
        self.changed(rebuild_headers=True)

    def track_menu(self, tid, anchor):
        t = self.track(tid)
        m = QMenu(self)
        idx = self.project["tracks"].index(t)
        m.addAction("Duplikálás", lambda: self._dup_track(tid))
        m.addAction("Feljebb", lambda: self._move_track(tid, -1)).setEnabled(idx > 0)
        m.addAction("Lejjebb", lambda: self._move_track(tid, 1)).setEnabled(idx < len(self.project["tracks"]) - 1)
        m.addSeparator()
        m.addAction("Sáv exportálása…", lambda: self.export_stems(only=tid))
        m.addAction("MIDI kotta a sávból…", lambda: self.make_midi(tid, None))
        m.addAction("Hangerő-automáció törlése", lambda: self._clear_auto(tid)).setEnabled(bool(t["automation"]))
        m.addSeparator()
        m.addAction("Sáv törlése", lambda: self._delete_track(tid))
        m.exec(anchor.mapToGlobal(anchor.rect().bottomLeft()))

    def _dup_track(self, tid):
        self.record()
        t = copy.deepcopy(self.track(tid))
        t["id"] = new_id()
        t["name"] += " (másolat)"
        for c in t["clips"]:
            c["id"] = new_id()
        self.project["tracks"].insert(self.project["tracks"].index(self.track(tid)) + 1, t)
        self.changed(rebuild_headers=True)

    def _move_track(self, tid, d):
        self.record()
        ts = self.project["tracks"]
        i = ts.index(self.track(tid))
        ts[i], ts[i + d] = ts[i + d], ts[i]
        self.changed(rebuild_headers=True)

    def _clear_auto(self, tid):
        self.record()
        self.track(tid)["automation"] = []
        self.changed()

    def _delete_track(self, tid):
        if QMessageBox.question(self, "Sáv törlése", f"Törlöd a(z) '{self.track(tid)['name']}' sávot?") != QMessageBox.Yes:
            return
        self.record()
        self.project["tracks"] = [t for t in self.project["tracks"] if t["id"] != tid]
        if self.selected and self.selected[0] == tid:
            self.selected = None
        self.changed(rebuild_headers=True)

    # ---------------------------------------------------------- klip műveletek
    def select(self, tid, cid):
        self.selected = (tid, cid) if tid else None
        self.canvas.invalidate()
        if cid:
            c = self.clip(tid, cid)
            self.status(f"Kijelölve: {c['name']} · kezdés {fmt_time(c['start'])} · hossz {c['length']:.2f} mp  "
                        f"(dupla kattintás: tulajdonságok, jobb klikk: menü)")

    def clip_menu(self, tid, cid, gpos):
        c = self.clip(tid, cid)
        m = QMenu(self)
        m.addAction("Vágás a lejátszófejnél  (S)", self.split_selected)
        m.addAction("Duplikálás  (Ctrl+D)", self.duplicate_selected)
        m.addAction("Másolás  (Ctrl+C)", self.copy_selected)
        m.addAction("Beillesztés a lejátszófejhez  (Ctrl+V)", self.paste).setEnabled(self.clipboard is not None)
        m.addAction("Némítás ki" if c.get("muted") else "Némítás", lambda: self._clip_prop(tid, cid, "muted", not c.get("muted")))
        m.addAction("Normalizálás (-1 dB csúcs)", lambda: self.normalize(tid, cid))
        m.addAction("Áthelyezés új sávra", lambda: self._to_new_track(tid, cid))
        m.addAction("Tulajdonságok… (tempó, hangmagasság)", lambda: self.clip_dialog(tid, cid))
        m.addSeparator()
        m.addAction("✂ Sávokra bontás (ének, dob, basszus, egyéb)", lambda: self.separate_clip(tid, cid))
        m.addAction("♫ MIDI kotta készítése…", lambda: self.make_midi(tid, cid))
        m.addAction("Klip exportálása…", lambda: self.export_clip(tid, cid))
        m.addAction("Forrásfájl mappája", lambda: open_in_explorer(Path(c["src"])))
        m.addSeparator()
        m.addAction("Törlés  (Del)", self.delete_selected)
        m.exec(gpos)

    def lane_menu(self, tid, t, gpos):
        m = QMenu(self)
        m.addAction("Lejátszófej ide", lambda: self.seek(self.snap(t)))
        m.addAction("Beillesztés ide", lambda: self.paste(at=self.snap(t), tid=tid)).setEnabled(self.clipboard is not None)
        m.addAction("Hangfájl importálása erre a sávra…", lambda: self.import_files(tid=tid, at=self.snap(t)))
        m.exec(gpos)

    def _clip_prop(self, tid, cid, k, v):
        self.record()
        self.clip(tid, cid)[k] = v
        self.changed()

    def clip_dialog(self, tid, cid):
        c = self.clip(tid, cid)
        before = self.snapshot()
        d = ClipDialog(self, self, c)
        install_tip_filter(d)
        if d.exec() == QDialog.Accepted:
            d.apply()
            self.push_undo(before)
            self.changed()

    def split_selected(self):
        if not self.selected or not self.selected[1]:
            return
        tid, cid = self.selected
        c = self.clip(tid, cid)
        t = self.engine.position()
        if not (c["start"] + 0.01 < t < c["start"] + c["length"] - 0.01):
            self.status("A lejátszófej nincs a kijelölt klipen belül - előbb kattints oda, ahol vágni szeretnél.")
            return
        self.record()
        cut = t - c["start"]
        c2 = copy.deepcopy(c)
        c2["id"] = new_id()
        c2["start"] = t
        c2["offset"] = c["offset"] + cut
        c2["length"] = c["length"] - cut
        c2["fade_in"] = 0.0
        c["length"] = cut
        c["fade_out"] = 0.0
        tr = self.track(tid)
        tr["clips"].insert(tr["clips"].index(c) + 1, c2)
        self.changed()

    def delete_selected(self):
        if not self.selected or not self.selected[1]:
            return
        tid, cid = self.selected
        self.record()
        tr = self.track(tid)
        tr["clips"] = [c for c in tr["clips"] if c["id"] != cid]
        self.selected = None
        self.changed()

    def duplicate_selected(self):
        if not self.selected or not self.selected[1]:
            return
        tid, cid = self.selected
        c = self.clip(tid, cid)
        self.record()
        c2 = copy.deepcopy(c)
        c2["id"] = new_id()
        c2["start"] = c["start"] + c["length"]
        self.track(tid)["clips"].append(c2)
        self.selected = (tid, c2["id"])
        self.changed()

    def copy_selected(self):
        if self.selected and self.selected[1]:
            self.clipboard = copy.deepcopy(self.clip(*self.selected))
            self.status("Klip másolva - Ctrl+V a lejátszófej helyére illeszti.")

    def paste(self, at=None, tid=None):
        if self.clipboard is None:
            return
        tid = tid or (self.selected[0] if self.selected else (self.project["tracks"][0]["id"] if self.project["tracks"] else None))
        if tid is None:
            return
        self.record()
        c = copy.deepcopy(self.clipboard)
        c["id"] = new_id()
        c["start"] = at if at is not None else self.engine.position()
        self.track(tid)["clips"].append(c)
        self.selected = (tid, c["id"])
        self.changed()

    def normalize(self, tid, cid):
        c = self.clip(tid, cid)
        src = self.engine.cache.get(c["src"], c.get("stretch", 1.0), c.get("pitch", 0.0), c.get("reverse", False))
        o, n = int(c["offset"] * DAW_SR), int(c["length"] * DAW_SR)
        peak = float(np.abs(src[:, o:o + n]).max(initial=0))
        if peak <= 1e-6:
            return
        self.record()
        c["gain_db"] = round(-1.0 - 20 * math.log10(peak), 2)
        self.changed()
        self.status(f"Normalizálva: klip hangerő {c['gain_db']:+.2f} dB")

    def _to_new_track(self, tid, cid):
        self.record()
        c = self.clip(tid, cid)
        t = self.add_track(c["name"], index=self.project["tracks"].index(self.track(tid)) + 1)
        self.track(tid)["clips"].remove(c)
        t["clips"].append(c)
        self.changed(rebuild_headers=True)

    def move_clip_to_track(self, tid, cid, new_tid, record=True):
        if tid == new_tid:
            return
        if record:
            self.record()
        c = self.clip(tid, cid)
        self.track(tid)["clips"].remove(c)
        self.track(new_tid)["clips"].append(c)
        self.selected = (new_tid, cid)

    # ---------------------------------------------------------- import
    def add_file_as_track(self, path: str, name: str | None = None, at: float = 0.0, tid: str | None = None,
                          color=None, index=None) -> dict | None:
        try:
            L = self.engine.cache.length(path)
        except Exception as e:
            QMessageBox.warning(self, "Importálás", f"Nem sikerült beolvasni:\n{path}\n\n{e}")
            return None
        t = self.track(tid) if tid else self.add_track(name or Path(path).stem, color, index)
        t["clips"].append(new_clip(path, at, L, name or Path(path).stem))
        return t

    def import_files(self, tid=None, at=None):
        files, _ = QFileDialog.getOpenFileNames(self, "Hangfájl(ok) importálása", self.main.cfg["output_dir"],
                                                "Hang (*.wav *.mp3 *.flac *.ogg *.aiff *.m4a);;Minden (*)")
        if not files:
            return
        self.record()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for f in files:
                self.add_file_as_track(f, at=self.engine.position() if at is None else at, tid=tid)
        finally:
            QApplication.restoreOverrideCursor()
        self.changed(rebuild_headers=True)

    def import_from_library(self, t: dict | None = None):
        t = t or self.main._sel_track()
        if not t:
            QMessageBox.information(self, "DAW", "Előbb jelölj ki egy számot a Könyvtár fülön.")
            return
        if not Path(t["wav_path"]).is_file():
            QMessageBox.warning(self, "DAW", "A szám hangfájlja nem található.")
            return
        self.record()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.add_file_as_track(t["wav_path"], name=t.get("title"), at=0.0)
            if len(self.project["tracks"]) == 1:
                self.project["name"] = t.get("title") or "Projekt"
        finally:
            QApplication.restoreOverrideCursor()
        self.changed(rebuild_headers=True)
        self.zoom_fit()

    # ---------------------------------------------------------- sávbontás, MIDI
    def separate_file(self, path: str | None = None, at: float = 0.0, after_index=None, mute_tid=None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Melyik dalt bontsuk sávokra?", self.main.cfg["output_dir"],
                                                  "Hang (*.wav *.mp3 *.flac *.ogg *.m4a);;Minden (*)")
            if not path:
                return
        self.main.request_separation(path, {"at": at, "after_index": after_index, "mute_tid": mute_tid})
        self.status("Sávokra bontás a feladatsorban… (pár perc lehet; a YuE2 addig kiürül a videókártyáról)")

    def separate_clip(self, tid, cid):
        c = self.clip(tid, cid)
        idx = self.project["tracks"].index(self.track(tid))
        if abs(c.get("stretch", 1) - 1) > 1e-3 or abs(c.get("pitch", 0)) > 1e-3 or c.get("reverse") or c["offset"] > 0.01:
            # a módosított részletet bontjuk
            src = self.engine.cache.get(c["src"], c.get("stretch", 1.0), c.get("pitch", 0.0), c.get("reverse", False))
            o, n = int(c["offset"] * DAW_SR), int(c["length"] * DAW_SR)
            tmp = DAW_DIR / "temp" / f"klip_{c['id']}.wav"
            tmp.parent.mkdir(exist_ok=True)
            write_audio(tmp, src[:, o:o + n])
            self.separate_file(str(tmp), at=c["start"], after_index=idx, mute_tid=tid)
        else:
            self.separate_file(c["src"], at=c["start"], after_index=idx, mute_tid=tid)

    def on_separated(self, stems: dict, ctx: dict):
        self.record()
        idx = ctx.get("after_index")
        idx = (idx + 1) if idx is not None else len(self.project["tracks"])
        order = ["vocals", "drums", "bass", "other"] + [k for k in stems if k not in ("vocals", "drums", "bass", "other")]
        colors = {"vocals": "#f472b6", "drums": "#f59e0b", "bass": "#22d3ee", "other": "#4ade80"}
        added = 0
        for k in order:
            if k not in stems:
                continue
            t = self.add_file_as_track(stems[k], name=tr(STEM_NAMES.get(k, k)), at=ctx.get("at", 0.0),
                                       color=colors.get(k), index=idx + added)
            if t:
                added += 1
        if ctx.get("mute_tid") and self.track(ctx["mute_tid"]):
            self.track(ctx["mute_tid"])["mute"] = True
        self.changed(rebuild_headers=True)
        if added and len(self.project["tracks"]) == added:
            self.zoom_fit()
        self.status(f"Sávokra bontás kész: {added} új sáv." + (" Az eredeti sávot elnémítottam." if ctx.get("mute_tid") else ""))

    def make_midi(self, tid, cid):
        t = self.track(tid)
        items = [("Automatikus (a sáv neve alapján)", None), ("Ének / dallam", "voice"), ("Dob", "drums"),
                 ("Basszus", "electric_bass,acoustic_bass"), ("Zongora", "acoustic_piano,electric_piano"),
                 ("Gitár", "acoustic_guitar,clean_electric_guitar,distorted_electric_guitar"),
                 ("Vonósok", "violin,viola,cello,string_ensemble"), ("Bármi (nincs megkötés)", "")]
        names = [i[0] for i in items]
        choice, ok = QInputDialog.getItem(self, "MIDI kotta", "Milyen hangszer hangjait keresse a MuScriptor?", names, 0, False)
        if not ok:
            return
        instr = dict(items)[choice]
        if instr is None:
            nm = t["name"].lower()
            instr = ("voice" if any(w in nm for w in ("ének", "vocal", "hang")) else
                     "drums" if "dob" in nm or "drum" in nm else
                     "electric_bass,acoustic_bass" if "basszus" in nm or "bass" in nm else "")
        # a renderelt (effektek nélküli) anyag mono WAV-ként
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            buf = np.zeros((2, int((project_length(self.project) + 1) * DAW_SR)), np.float32)
            clips = [self.clip(tid, cid)] if cid else t["clips"]
            for c in clips:
                render_clip_into(buf, c, self.engine.cache)
            if cid:
                c = clips[0]
                s, n = int(c["start"] * DAW_SR), int(c["length"] * DAW_SR)
                buf = buf[:, s:s + n]
            mono = buf.mean(axis=0, keepdims=True)
            tmp = DAW_DIR / "temp" / f"midi_{new_id()}.wav"
            tmp.parent.mkdir(exist_ok=True)
            write_audio(tmp, mono)
        finally:
            QApplication.restoreOverrideCursor()
        base = safe_name((self.clip(tid, cid)["name"] if cid else t["name"]) + "_" + self.project.get("name", ""), 60)
        self.main.request_midi(str(tmp), instr, base)
        self.status("MIDI készítése a feladatsorban…")

    # ---------------------------------------------------------- FX
    def open_fx(self, tid):
        if tid is None:
            chain = self.project["master_fx"]
            title = "Mester effektek"
        else:
            chain = self.track(tid)["fx"]
            title = f"Effektek – {self.track(tid)['name']}"
        before = self.snapshot()

        def on_change():
            if tid is None:
                self.engine.update_mix(self.project)
            else:
                self.engine.sync(self.project)
                self._render_status()
            self.autosave()
        d = FxDialog(self, title, chain, on_change)
        install_tip_filter(d)
        d.exec()
        if self.snapshot() != before:
            self.push_undo(before)
        on_change()
        self.rebuild()

    # ---------------------------------------------------------- undo
    def undo(self):
        if not self.undo_stack:
            return
        self.redo_stack.append(self.snapshot())
        self.project = json.loads(self.undo_stack.pop())
        self.selected = None
        self.rebuild()
        self.autosave()

    def redo(self):
        if not self.redo_stack:
            return
        self.undo_stack.append(self.snapshot())
        self.project = json.loads(self.redo_stack.pop())
        self.selected = None
        self.rebuild()
        self.autosave()

    # ---------------------------------------------------------- projekt fájl
    def new_project(self, ask: bool = True):
        if ask and self.project["tracks"] and QMessageBox.question(
                self, "Tiszta lap", "Kiürítsem a sávszerkesztőt? (Ctrl+Z-vel visszahozható.)") != QMessageBox.Yes:
            return
        self.record()
        self.engine.pause()
        self.engine.seek(0)
        self.project = new_project()
        self.project_path = None
        self.rebuild()
        self.autosave()

    def open_project(self):
        f, _ = QFileDialog.getOpenFileName(self, "Projekt megnyitása", str(DAW_DIR), "YuE2 DAW projekt (*.y2daw)")
        if not f:
            return
        self.record()
        self.engine.pause()
        self.project = json.loads(Path(f).read_text(encoding="utf-8"))
        self.project_path = Path(f)
        self.rebuild()
        self.zoom_fit()
        self.autosave()

    def save_project(self, save_as=False):
        if self.project_path is None or save_as:
            f, _ = QFileDialog.getSaveFileName(self, "Projekt mentése",
                                               str(DAW_DIR / f"{safe_name(self.project.get('name', 'projekt'))}.y2daw"),
                                               "YuE2 DAW projekt (*.y2daw)")
            if not f:
                return
            self.project_path = Path(f)
            self.project["name"] = self.project_path.stem
        self.project_path.write_text(json.dumps(self.project, ensure_ascii=False, indent=1), encoding="utf-8")
        self.status(f"Mentve: {self.project_path}")

    # ---------------------------------------------------------- export
    def _export_dialog(self):
        box = QMessageBox(self)
        box.setWindowTitle("Exportálás")
        box.setText("Mit exportáljunk?")
        b_all = box.addButton("Teljes dal", QMessageBox.AcceptRole)
        b_loop = box.addButton("Csak a hurok tartomány", QMessageBox.AcceptRole)
        box.addButton("Mégse", QMessageBox.RejectRole)
        lp = self.project.get("loop") or [0, 0]
        b_loop.setEnabled(lp[1] > lp[0])
        box.exec()
        if box.clickedButton() is b_all:
            return 0.0, None
        if box.clickedButton() is b_loop:
            return lp[0], lp[1]
        return None

    def _progress(self, title):
        from PySide6.QtWidgets import QProgressDialog
        pd = QProgressDialog(title, None, 0, 100, self)
        pd.setWindowModality(Qt.WindowModal)
        pd.setMinimumDuration(0)
        pd.setValue(0)

        def cb(x):
            pd.setValue(int(x * 100))
            QApplication.processEvents()
        return pd, cb

    def export_mix(self):
        if not self.project["tracks"]:
            return
        rng = self._export_dialog()
        if rng is None:
            return
        out_dir = Path(self.main.cfg["output_dir"]) / dt.date.today().isoformat()
        out_dir.mkdir(parents=True, exist_ok=True)
        f, _ = QFileDialog.getSaveFileName(self, "Keverés mentése",
                                           str(out_dir / f"{safe_name(self.project.get('name', 'keveres'))}_keveres.wav"),
                                           "WAV (*.wav);;MP3 (*.mp3);;FLAC (*.flac)")
        if not f:
            return
        self.engine.pause()
        pd, cb = self._progress("Keverés számolása…")
        try:
            data = self.engine.render_offline(self.project, rng[0], rng[1], progress=cb)
            write_audio(f, data)
        except Exception as e:
            pd.close()
            QMessageBox.warning(self, "Export", f"Hiba: {e}")
            return
        pd.close()
        dur = data.shape[1] / DAW_SR
        if QMessageBox.question(self, "Export kész", f"Elkészült:\n{f}\n\nFelvegyem a Könyvtárba is?") == QMessageBox.Yes:
            self.main.db.add(created=dt.datetime.now().isoformat(timespec="seconds"),
                             title=f"{self.project.get('name', 'Keverés')} (DAW keverés)", style="DAW keverés",
                             lyrics="", seed=0, cot="", model="DAW",
                             params=json.dumps({"daw_project": self.project}, ensure_ascii=False),
                             wav_path=f, abc_path="", duration=dur, gen_secs=0,
                             project=getattr(self.main, "project_name", ""))
            self.main.refresh_library()
        self.status(f"Exportálva: {f}")

    def export_stems(self, only=None):
        if not self.project["tracks"]:
            return
        d = QFileDialog.getExistingDirectory(self, "Hová mentsük a sávokat?", self.main.cfg["output_dir"])
        if not d:
            return
        self.engine.pause()
        tracks = [t for t in self.project["tracks"] if (only is None or t["id"] == only)]
        pd, cb = self._progress("Sávok számolása…")
        try:
            for i, t in enumerate(tracks):
                data = self.engine.render_offline(self.project, only_track=t["id"],
                                                  progress=lambda x, i=i: cb((i + x) / len(tracks)))
                write_audio(Path(d) / f"{i + 1:02d}_{safe_name(t['name'])}.wav", data)
        except Exception as e:
            pd.close()
            QMessageBox.warning(self, "Export", f"Hiba: {e}")
            return
        pd.close()
        self.status(f"{len(tracks)} sáv exportálva ide: {d}")
        open_in_explorer(Path(d))

    def export_clip(self, tid, cid):
        c = self.clip(tid, cid)
        f, _ = QFileDialog.getSaveFileName(self, "Klip mentése", str(Path(self.main.cfg["output_dir"]) / f"{safe_name(c['name'])}.wav"),
                                           "WAV (*.wav);;MP3 (*.mp3)")
        if not f:
            return
        buf = np.zeros((2, int((c["start"] + c["length"] + 1) * DAW_SR)), np.float32)
        render_clip_into(buf, c, self.engine.cache)
        s = int(c["start"] * DAW_SR)
        write_audio(f, buf[:, s:s + int(c["length"] * DAW_SR)])
        self.status(f"Klip exportálva: {f}")

    def shutdown(self):
        self.autosave()
        self.engine.close()


def daw_missing_widget() -> QWidget:
    w = QWidget()
    l = QVBoxLayout(w)
    miss = [n for n, ok in (("numpy", HAS_NUMPY), ("pedalboard", HAS_PEDALBOARD), ("sounddevice", HAS_SOUNDDEVICE)) if not ok]
    lbl = QLabel("<h2>A DAW-hoz még hiányzik néhány Python csomag</h2>"
                 f"<p>Hiányzik: <b>{', '.join(miss)}</b></p>"
                 "<p>Telepítés: zárd be az appot, és futtasd le parancssorban:</p>"
                 f"<pre>python -m pip install {' '.join(miss)}</pre>"
                 "<p>(Az új YuE2_Studio_inditas.bat ezt magától elvégzi.)</p>")
    lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
    lbl.setWordWrap(True)
    l.addWidget(lbl)
    l.addStretch()
    return w


# ===========================================================================
# KOTTA - ABC / MIDI megjelenítése kottaként, nyomtatás, PDF
# ===========================================================================
try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings
    from PySide6.QtGui import QPageLayout, QPageSize
    from PySide6.QtCore import QMarginsF
    HAS_WEBENGINE = True
except Exception:  # pragma: no cover
    HAS_WEBENGINE = False

ABCJS_VERSION = "6.4.4"
ABCJS_SHA256 = "1f060d3568b8f49c0f22b7eaa98f8eb9e34a0e83c2dc14bc730d572a320e4c69"
ABCJS_URLS = [
    f"https://cdn.jsdelivr.net/npm/abcjs@{ABCJS_VERSION}/dist/abcjs-basic-min.js",
    f"https://unpkg.com/abcjs@{ABCJS_VERSION}/dist/abcjs-basic-min.js",
]
SCORE_LIB_DIR = APP_DIR / "kotta_lib"
SCORE_DIR = APP_DIR / "kottak"

# --------------------------------------------------------------- MIDI olvasó
def _read_varlen(data: bytes, i: int):
    v = 0
    while True:
        b = data[i]
        i += 1
        v = (v << 7) | (b & 0x7F)
        if not b & 0x80:
            return v, i


def parse_midi(path: str) -> dict:
    """Egyszerű SMF olvasó: hangok, tempó, ütemmutató, előjegyzés."""
    data = Path(path).read_bytes()
    if data[:4] != b"MThd":
        raise ValueError("Ez nem szabványos MIDI fájl.")
    hlen = int.from_bytes(data[4:8], "big")
    ntrks = int.from_bytes(data[10:12], "big")
    div = int.from_bytes(data[12:14], "big")
    if div & 0x8000:
        raise ValueError("SMPTE időzítésű MIDI nem támogatott.")
    pos = 8 + hlen
    notes, tempo, timesig, keysig, names = [], None, None, None, []
    for tr in range(ntrks):
        if data[pos:pos + 4] != b"MTrk":
            break
        tlen = int.from_bytes(data[pos + 4:pos + 8], "big")
        i, end = pos + 8, pos + 8 + tlen
        pos = end
        tick, status = 0, 0
        on = {}
        while i < end:
            delta, i = _read_varlen(data, i)
            tick += delta
            b = data[i]
            if b & 0x80:
                status = b
                i += 1
            if status == 0xFF:
                mtype = data[i]
                ln, i = _read_varlen(data, i + 1)
                md = data[i:i + ln]
                i += ln
                if mtype == 0x51 and tempo is None:
                    tempo = int.from_bytes(md, "big")
                elif mtype == 0x58 and timesig is None and ln >= 2:
                    timesig = (md[0], 2 ** md[1])
                elif mtype == 0x59 and keysig is None and ln >= 2:
                    keysig = (int.from_bytes(md[:1], "big", signed=True), md[1])
                elif mtype == 0x03:
                    names.append(md.decode("latin-1", "replace"))
                continue
            if status in (0xF0, 0xF7):
                ln, i = _read_varlen(data, i)
                i += ln
                continue
            hi = status & 0xF0
            ch = status & 0x0F
            if hi in (0xC0, 0xD0):
                i += 1
                continue
            d1, d2 = data[i], data[i + 1]
            i += 2
            if hi == 0x90 and d2 > 0:
                on.setdefault((ch, d1), []).append((tick, d2))
            elif hi == 0x80 or (hi == 0x90 and d2 == 0):
                lst = on.get((ch, d1))
                if lst:
                    st, vel = lst.pop(0)
                    notes.append({"start": st, "end": tick, "pitch": d1, "vel": vel, "ch": ch, "track": tr})
    return {"div": div, "notes": notes, "tempo_us": tempo or 500000, "timesig": timesig or (4, 4),
            "keysig": keysig, "names": names}


# --------------------------------------------------------------- ABC író
_MAJOR_KEYS = {0: "C", 1: "G", 2: "D", 3: "A", 4: "E", 5: "B", 6: "F#", 7: "C#",
               -1: "F", -2: "Bb", -3: "Eb", -4: "Ab", -5: "Db", -6: "Gb", -7: "Cb"}
_MINOR_KEYS = {0: "Am", 1: "Em", 2: "Bm", 3: "F#m", 4: "C#m", 5: "G#m", 6: "D#m", 7: "A#m",
               -1: "Dm", -2: "Gm", -3: "Cm", -4: "Fm", -5: "Bbm", -6: "Ebm", -7: "Abm"}
_SHARP_ORDER = "FCGDAEB"
_LETTERS_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def estimate_key_full(pitches, durations=None) -> tuple[int, bool]:
    """Krumhansl-Kessler profilok (dúr + moll): (előjegyzés -6..+6, moll-e)."""
    maj = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    mnr = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    hist = [0.0] * 12
    for i, p in enumerate(pitches):
        hist[p % 12] += durations[i] if durations else 1.0
    if not any(hist):
        return 0, False
    mh = sum(hist) / 12

    def corr(prof, tonic):
        mp = sum(prof) / 12
        xs = [hist[(tonic + k) % 12] - mh for k in range(12)]
        ys = [v - mp for v in prof]
        num = sum(a * b for a, b in zip(xs, ys))
        den = (sum(a * a for a in xs) * sum(b * b for b in ys)) ** 0.5 or 1.0
        return num / den
    best = (-2.0, 0, False)
    for tonic in range(12):
        fifths_major = ((tonic * 7) % 12)
        fifths_major = fifths_major - 12 if fifths_major > 6 else fifths_major
        c = corr(maj, tonic)
        if c > best[0]:
            best = (c, fifths_major, False)
        rel = (tonic + 3) % 12
        fm = (rel * 7) % 12
        fm = fm - 12 if fm > 6 else fm
        c = corr(mnr, tonic)
        if c > best[0]:
            best = (c, fm, True)
    return best[1], best[2]


def estimate_key(pitches) -> int:
    return estimate_key_full(pitches)[0]


def _key_alter(fifths: int) -> dict:
    alt = {l: 0 for l in "CDEFGAB"}
    if fifths > 0:
        for l in _SHARP_ORDER[:fifths]:
            alt[l] = 1
    elif fifths < 0:
        for l in "BEADGCF"[:-fifths]:
            alt[l] = -1
    return alt


def _spell(pitch: int, fifths: int):
    """MIDI hangmagasság -> (betű, módosítás, oktáv) a hangnemhez illő írásmóddal."""
    pc = pitch % 12
    octave = pitch // 12 - 1
    key = _key_alter(fifths)
    for l, base in _LETTERS_PC.items():
        if (base + key[l]) % 12 == pc:
            return l, key[l], octave + (-1 if base + key[l] >= 12 else (1 if base + key[l] < 0 else 0))
    prefer_flat = fifths < 0
    for l, base in _LETTERS_PC.items():
        if base == pc:
            return l, 0, octave
    if prefer_flat:
        for l, base in _LETTERS_PC.items():
            if (base - 1) % 12 == pc:
                return l, -1, octave + (1 if base - 1 < 0 else 0)
    for l, base in _LETTERS_PC.items():
        if (base + 1) % 12 == pc:
            return l, 1, octave
    return "C", 0, octave


def _abc_pitch(letter, octave):
    if octave >= 5:
        return letter.lower() + "'" * (octave - 5)
    return letter + "," * max(0, 4 - octave)


_DUR_PARTS = [16, 12, 8, 6, 4, 3, 2, 1]


def _split_dur(n: int):
    out = []
    while n > 0:
        for p in _DUR_PARTS:
            if p <= n:
                out.append(p)
                n -= p
                break
    return out


def _dur_str(n: int, unit: int) -> str:
    """n darab egység (1 egység = 1/16 vagy 1/8 hang a 'unit' szerint)."""
    if n == 1:
        return ""
    return str(n)


def _voice_events(notes, grid_ticks, total_q, chords=True):
    """Hangok -> egymást követő (kezdés, hossz, [hangmagasságok]) események, szünetekkel."""
    by_start = {}
    for n in notes:
        s = int(round(n["start"] / grid_ticks))
        d = max(1, int(round((n["end"] - n["start"]) / grid_ticks)))
        by_start.setdefault(s, []).append((n["pitch"], d))
    starts = sorted(by_start)
    ev = []
    for i, s in enumerate(starts):
        group = by_start[s]
        nxt = starts[i + 1] if i + 1 < len(starts) else None
        d = max(dd for _p, dd in group)
        if nxt is not None:
            d = min(d, nxt - s)
        pitches = sorted({p for p, _d in group})
        if not chords:
            pitches = [pitches[-1]]
        ev.append((s, max(1, d), pitches))
    return ev


def _render_voice(events, bar_len, total_bars, fifths, bars_per_line=4):
    """Események -> ABC sorok ütemvonalakkal, kötésekkel, ütemenkénti módosítójelekkel."""
    key = _key_alter(fifths)
    out_bars = []
    pos = 0
    ev_i = 0
    # elemek: (hossz, hangok|None)
    items = []
    for s, d, ps in events:
        if s > pos:
            items.append((s - pos, None))
        items.append((d, ps))
        pos = s + d
    end = total_bars * bar_len
    if pos < end:
        items.append((end - pos, None))
    bar_tokens, cur = [], []
    room = bar_len
    state = {}

    def fmt_notes(ps, n, tie):
        toks = []
        for p in ps:
            l, alt, octv = _spell(p, fifths)
            k = (l, octv)
            curalt = state.get(k, key[l])
            acc = ""
            if alt != curalt:
                acc = {1: "^", -1: "_", 0: "="}[alt]
                state[k] = alt
            toks.append(acc + _abc_pitch(l, octv))
        ds = _dur_str(n, 1)
        if len(toks) == 1:
            return toks[0] + ds + ("-" if tie else "")
        return "[" + "".join(toks) + "]" + ds + ("-" if tie else "")

    for length, ps in items:
        remaining = length
        while remaining > 0:
            take = min(remaining, room)
            parts = _split_dur(take)
            for j, part in enumerate(parts):
                last_piece = (j == len(parts) - 1) and (remaining - take == 0)
                if ps is None:
                    cur.append("z" + _dur_str(part, 1))
                else:
                    cur.append(fmt_notes(ps, part, tie=not last_piece))
            remaining -= take
            room -= take
            if room == 0:
                bar_tokens.append(" ".join(cur))
                cur = []
                room = bar_len
                state = {}
    if cur:
        bar_tokens.append(" ".join(cur))
    lines = []
    for i in range(0, len(bar_tokens), bars_per_line):
        lines.append(" | ".join(bar_tokens[i:i + bars_per_line]) + " |")
    if lines:
        lines[-1] = lines[-1][:-1] + "|]"
    return lines


def skyline(notes, tol=0.0):
    """Dallamkivonás: egy hang csak akkor marad, ha kezdetekor nem szól fölötte magasabb hang."""
    ns = sorted(notes, key=lambda n: (n["start"], -n["pitch"]))
    keep = []
    active = []
    for n in ns:
        active = [a for a in active if a["end"] - tol > n["start"]]
        if any(a["pitch"] > n["pitch"] for a in active):
            continue
        keep.append(n)
        active.append(n)
    return keep


def midi_to_abc(path: str, mode: str = "auto", grid: int = 16, title: str = "", key_override: int | None = None,
                bpm_override: float | None = None, bars_per_line: int = 4) -> str:
    """MIDI -> ABC. mode: 'melody' (egy sor, dallam), 'piano' (két sor, akkordok), 'auto'."""
    m = parse_midi(path)
    notes = [n for n in m["notes"] if n["ch"] != 9]
    if not notes:
        if m["notes"]:
            raise ValueError("Ebben a MIDI-ben csak dob van - dobkotta most nem készíthető.")
        raise ValueError("A MIDI fájlban nincsenek hangok.")
    num, den = m["timesig"]
    div = m["div"]
    grid_ticks = div * 4 / grid           # grid=16 -> 16-od hang
    unit_per_quarter = grid // 4
    bar_len = int(num * (grid / den))
    pitches = [n["pitch"] for n in notes]
    est_fifths, est_minor = estimate_key_full(pitches, [n["end"] - n["start"] for n in notes])
    if key_override is not None:
        fifths, minor = key_override, est_minor and key_override == est_fifths
    elif m["keysig"] is not None and m["keysig"][0] != 0:
        fifths, minor = m["keysig"][0], bool(m["keysig"][1])
    else:
        fifths, minor = est_fifths, est_minor
    keyname = (_MINOR_KEYS if minor else _MAJOR_KEYS).get(fifths, "C")
    bpm = bpm_override or round(60_000_000 / m["tempo_us"])
    last_q = max(int(round(n["end"] / grid_ticks)) for n in notes)
    total_bars = max(1, -(-last_q // bar_len))
    if mode == "auto":
        lo = sum(1 for p in pitches if p < 55)
        mode = "piano" if lo > len(pitches) * 0.15 and len(set(n["start"] for n in notes)) < len(notes) * 0.9 else "melody"
    head = ["X:1", f"T:{title or Path(path).stem}", f"M:{num}/{den}", f"L:1/{grid}",
            f"Q:1/4={int(bpm)}", "%%score {RH LH}" if mode == "piano" else "", f"K:{keyname}"]
    head = [h for h in head if h]
    body = []
    if mode == "piano":
        rh = [n for n in notes if n["pitch"] >= 60]
        lh = [n for n in notes if n["pitch"] < 60]
        body.append('V:RH clef=treble name="Jobb kéz"')
        body += _render_voice(_voice_events(rh, grid_ticks, last_q), bar_len, total_bars, fifths, bars_per_line) if rh else \
            [" | ".join(["z" + str(bar_len)] * total_bars) + " |]"]
        body.append('V:LH clef=bass name="Bal kéz"')
        body += _render_voice(_voice_events(lh, grid_ticks, last_q), bar_len, total_bars, fifths, bars_per_line) if lh else \
            [" | ".join(["z" + str(bar_len)] * total_bars) + " |]"]
    else:
        mel = skyline(notes, tol=grid_ticks * 0.5)
        mp = [n["pitch"] for n in mel]
        clef = "bass" if sum(mp) / len(mp) < 55 else "treble"
        body.append(f"V:1 clef={clef}")
        body += _render_voice(_voice_events(mel, grid_ticks, last_q, chords=False), bar_len, total_bars, fifths,
                              bars_per_line)
    return "\n".join(head + body) + "\n"


def abc_filter_voices(abc: str, keep: str) -> str:
    """YuE2 ABC: csak a megadott szólam (pl. 'Vocal' / 'Ins') megtartása."""
    if not keep:
        return abc
    lines = abc.replace("\r\n", "\n").split("\n")
    out, skip = [], False
    for ln in lines:
        mm = re.match(r"^V:\s*(\S+)", ln)
        if mm:
            skip = mm.group(1).lower() != keep.lower()
            if "=" in ln and skip:      # fejléc szólam-definíció: kihagyjuk
                continue
        if not skip:
            out.append(ln)
    return "\n".join(out)


def abc_strip_chords(abc: str) -> str:
    out = []
    for ln in abc.split("\n"):
        if re.match(r"^[A-Za-z]:", ln) or ln.startswith("%"):
            out.append(ln)
        else:
            out.append(re.sub(r'"[^"]*"', "", ln))
    return "\n".join(out)


def abc_set_header(abc: str, field: str, value: str) -> str:
    if not value:
        return abc
    lines = abc.split("\n")
    for i, ln in enumerate(lines):
        if ln.startswith(field + ":"):
            lines[i] = f"{field}:{value}"
            return "\n".join(lines)
    # X: után szúrjuk be
    for i, ln in enumerate(lines):
        if ln.startswith("X:"):
            lines.insert(i + 1, f"{field}:{value}")
            return "\n".join(lines)
    return f"{field}:{value}\n" + abc


SCORE_HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<style>
 html,body{margin:0;background:#e9e9ee;font-family:'Segoe UI',sans-serif}
 #paper{background:#fff;margin:18px auto;padding:24px 28px;box-shadow:0 2px 14px rgba(0,0,0,.25);max-width:900px}
 #msg{color:#b91c1c;font-size:13px;max-width:900px;margin:8px auto}
 .abcjs-note_selected{fill:#6a4dff !important}
 svg{display:block}
 @media print{ html,body{background:#fff !important} #paper{box-shadow:none;margin:0;padding:0;max-width:none} #msg{display:none} }
</style>
<script src="abcjs-basic-min.js"></script>
</head><body><div id="msg"></div><div id="paper"></div>
<script>
var visualObj=null, synthControl=null, timer=null, currentAbc="", lastOpts={};
function render(abc, opts){
  currentAbc=abc; lastOpts=opts||{};
  var o={add_classes:true, responsive:"resize", staffwidth:lastOpts.staffwidth||760,
         visualTranspose:lastOpts.transpose||0, scale:lastOpts.scale||1.0,
         paddingtop:10, paddingbottom:10, paddingleft:10, paddingright:10,
         format:{titlefont:"Georgia 22", subtitlefont:"Georgia 15", composerfont:"Georgia 13",
                 vocalfont:"Georgia 13", annotationfont:"Georgia 12", gchordfont:"Georgia 13"}};
  if(typeof ABCJS==="undefined"){document.getElementById("msg").innerText="A kottamegjelenítő (abcjs) nem töltődött be.";return "noabcjs";}
  var res=ABCJS.renderAbc("paper", abc, o);
  visualObj=res[0];
  var w=(visualObj&&visualObj.warnings)?visualObj.warnings:[];
  document.getElementById("msg").innerText = w.length? ("Figyelmeztetés: "+w.slice(0,3).join(" · ").replace(/<[^>]+>/g,"")) : "";
  return "ok";
}
function clearSel(){document.querySelectorAll(".abcjs-note_selected").forEach(function(e){e.classList.remove("abcjs-note_selected");});}
function play(){
  if(!visualObj){return "nincs";}
  stop();
  if(!ABCJS.synth.supportsAudio()){return "Ez a nézet nem támogatja a hanglejátszást.";}
  synthControl=new ABCJS.synth.CreateSynth();
  var ac=new (window.AudioContext||window.webkitAudioContext)();
  return synthControl.init({audioContext:ac, visualObj:visualObj, options:{midiTranspose:lastOpts.transpose||0}})
   .then(function(){return synthControl.prime();})
   .then(function(){
      timer=new ABCJS.TimingCallbacks(visualObj,{eventCallback:function(ev){
         clearSel(); if(!ev){return;} (ev.elements||[]).forEach(function(g){g.forEach(function(el){el.classList.add("abcjs-note_selected");});});
         if(ev.elements&&ev.elements[0]&&ev.elements[0][0]){ev.elements[0][0].scrollIntoView({block:"center",behavior:"smooth"});}
      }});
      timer.start(); synthControl.start(); return "ok";})
   .catch(function(e){return "Lejátszási hiba (internet kell a hangszerhangokhoz): "+e;});
}
function stop(){ try{ if(synthControl){synthControl.stop();} if(timer){timer.stop();} }catch(e){} clearSel(); return "ok"; }
function midiB64(){
  try{
    var m=ABCJS.synth.getMidiFile(currentAbc,{midiOutputType:"binary", midiTranspose:lastOpts.transpose||0});
    var u=m[0]; var s=""; for(var i=0;i<u.length;i++){s+=String.fromCharCode(u[i]);} return btoa(s);
  }catch(e){return "ERR:"+e;}
}
</script></body></html>"""


def find_musescore() -> str | None:
    cands = []
    for base in (os.environ.get("ProgramFiles", r"C:\Program Files"), os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")):
        for v in ("MuseScore 4\\bin\\MuseScore4.exe", "MuseScore 3\\bin\\MuseScore3.exe", "MuseScore Studio 4\\bin\\MuseScore4.exe"):
            cands.append(Path(base) / v)
    for c in cands:
        if c.is_file():
            return str(c)
    for n in ("MuseScore4", "mscore", "musescore", "MuseScore3"):
        w = shutil.which(n)
        if w:
            return w
    return None


class ScorePanel(QWidget):
    """Kotta fül: ABC vagy MIDI forrás, megjelenítés, nyomtatás, PDF, MIDI, lejátszás."""

    def __init__(self, main):
        super().__init__()
        self.main = main
        self.source_abc = ""          # a forrás ABC (MIDI-ből átalakítva is)
        self.source_kind = ""         # abc / midi
        self.midi_path = ""
        self._ready = False
        self._pending = None
        SCORE_DIR.mkdir(exist_ok=True)
        root = QHBoxLayout(self)
        # ---- bal oldali vezérlők
        left = QWidget()
        left.setFixedWidth(360)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        self.btn_clear = clear_button(lambda: self.clear(ask=True), "A Kotta fül kiürítése: forrás, cím, szerző, "
                                      "beállítások. A mentett PDF-ek és fájlok megmaradnak.")
        lv.addWidget(self.btn_clear, 0, Qt.AlignRight)
        g = QGroupBox("1. Honnan jöjjön a kotta?")
        gl = QVBoxLayout(g)
        for txt, fn, tip in (
            ("Az ABC-kotta fül tartalma", self.from_abc_tab, "ABC-kotta fülről|Az ABC-kotta fülön lévő zenei tervet "
             "(pl. YuE2 generált vagy SheetSage2-vel kinyert) rajzolja ki."),
            ("Könyvtár: kijelölt dal terve", self.from_library, "Könyvtárból|A Könyvtár fülön kijelölt dalhoz mentett "
             "ABC tervet rajzolja ki (csak melody/full módú dalnál van)."),
            ("MIDI fájl megnyitása…", self.open_midi, "MIDI fájl|Egy .mid fájlból készít kottát (pl. a DAW-ban "
             "MuScriptorral készítettből). A ritmust a rácshoz igazítja."),
            ("Legutóbbi DAW MIDI", self.last_daw_midi, "Legutóbbi MIDI|A DAW-ban legutóbb készített MIDI kotta "
             "betöltése."),
            ("ABC fájl megnyitása…", self.open_abc, "ABC fájl|Egy .abc kottafájl megnyitása."),
        ):
            b = QPushButton(txt)
            b.setToolTip(tip_html(tip))
            b.clicked.connect(fn)
            gl.addWidget(b)
        self.lbl_src = QLabel("Még nincs betöltve kotta.")
        self.lbl_src.setWordWrap(True)
        self.lbl_src.setObjectName("dim")
        self.lbl_src.setToolTip(tip_html("Forrás|Honnan származik az épp látható kotta."))
        gl.addWidget(self.lbl_src)
        lv.addWidget(g)

        self.g_midi = QGroupBox("MIDI átalakítás")
        mf = QFormLayout(self.g_midi)
        self.cb_mode = QComboBox()
        for t_, d_ in (("Automatikus", "auto"), ("Dallam (egy sor)", "melody"), ("Zongora (két sor, akkordok)", "piano")):
            self.cb_mode.addItem(t_, d_)
        self.cb_mode.setToolTip(tip_html("Kotta típusa|Dallam: egyetlen ötvonalas sor, mindig a legfelső hang "
                                         "(énekhez ideális). Zongora: violin- és basszuskulcs, akkordokkal."))
        self.cb_grid = QComboBox()
        self.cb_grid.addItem("Tizenhatod (pontosabb)", 16)
        self.cb_grid.addItem("Nyolcad (egyszerűbb, olvashatóbb)", 8)
        self.cb_grid.setToolTip(tip_html("Ritmus rács|A felismert hangokat ehhez a legkisebb értékhez igazítja. "
                                         "Nyolcaddal egyszerűbb, könnyebben olvasható kotta lesz."))
        self.cb_key = QComboBox()
        self.cb_key.addItem("Automatikus felismerés", None)
        for f_ in range(-6, 7):
            self.cb_key.addItem(f"{_MAJOR_KEYS[f_]}-dúr / {_MINOR_KEYS[f_]}-moll  ({abs(f_)} {'#' if f_ > 0 else ('b' if f_ < 0 else '')})", f_)
        self.cb_key.setToolTip(tip_html("Hangnem|Az előjegyzés (kereszt/bé) a kotta elején. Automatikusan a "
                                        "hangok alapján becsüli; ha furcsa, válaszd ki kézzel."))
        self.sp_bpl = QSpinBox()
        self.sp_bpl.setRange(1, 8)
        self.sp_bpl.setValue(4)
        self.sp_bpl.setToolTip(tip_html("Ütem soronként|Hány ütem kerüljön egy kottasorba."))
        for w_ in (self.cb_mode, self.cb_grid, self.cb_key):
            w_.currentIndexChanged.connect(self._reconvert)
        self.sp_bpl.valueChanged.connect(self._reconvert)
        mf.addRow("Típus:", self.cb_mode)
        mf.addRow("Rács:", self.cb_grid)
        mf.addRow("Hangnem:", self.cb_key)
        mf.addRow("Ütem/sor:", self.sp_bpl)
        self.g_midi.setVisible(False)
        lv.addWidget(self.g_midi)

        g2 = QGroupBox("2. Megjelenés")
        f2 = QFormLayout(g2)
        self.ed_title = QLineEdit()
        self.ed_title.setToolTip(tip_html("Cím|A kotta tetején megjelenő cím."))
        self.ed_composer = QLineEdit()
        self.ed_composer.setToolTip(tip_html("Szerző|Jobb oldalt a cím alatt jelenik meg (pl. 'Zene: YuE2 · Szöveg: Sanyi')."))
        self.cb_voice = QComboBox()
        for t_, d_ in (("Minden szólam", ""), ("Csak ének (Vocal)", "Vocal"), ("Csak hangszer (Ins)", "Ins")):
            self.cb_voice.addItem(t_, d_)
        self.cb_voice.setToolTip(tip_html("Szólamok|A YuE2 tervében külön van ének (Vocal) és hangszer (Ins) szólam. "
                                          "Itt kiválaszthatod, melyik látsszon."))
        self.chk_chords = QCheckBox("Akkordjelek mutatása")
        self.chk_chords.setChecked(True)
        self.chk_chords.setToolTip(tip_html("Akkordjelek|A hangjegyek fölötti akkordnevek (pl. Am, G7) - gitárosoknak hasznos."))
        self.sp_tr = QSpinBox()
        self.sp_tr.setRange(-12, 12)
        self.sp_tr.setToolTip(tip_html("Transzponálás|Félhangokkal feljebb vagy lejjebb tolja a kottát (és a "
                                       "lejátszást). Pl. +2 = egy egész hanggal feljebb, ha kényelmesebb énekelni."))
        self.sp_scale = QDoubleSpinBox()
        self.sp_scale.setRange(0.5, 2.5)
        self.sp_scale.setSingleStep(0.1)
        self.sp_scale.setValue(1.0)
        self.sp_scale.setToolTip(tip_html("Méret|A kotta nagyítása a képernyőn és nyomtatásban."))
        for w_ in (self.ed_title, self.ed_composer):
            w_.editingFinished.connect(self.refresh)
        for w_ in (self.cb_voice,):
            w_.currentIndexChanged.connect(self.refresh)
        self.chk_chords.toggled.connect(self.refresh)
        self.sp_tr.valueChanged.connect(self.refresh)
        self.sp_scale.valueChanged.connect(self.refresh)
        f2.addRow("Cím:", self.ed_title)
        f2.addRow("Szerző:", self.ed_composer)
        f2.addRow("Szólam:", self.cb_voice)
        f2.addRow("", self.chk_chords)
        f2.addRow("Transzponálás:", self.sp_tr)
        f2.addRow("Méret:", self.sp_scale)
        lv.addWidget(g2)

        g3 = QGroupBox("3. Mit kezdjünk vele?")
        g3l = QGridLayout(g3)
        acts = [
            ("▶ Lejátszás", self.play, "Lejátszás|Lejátssza a kottát, és közben lila színnel mutatja, hol tart. "
             "A hangszerhangokat első alkalommal az internetről tölti le."),
            ("■ Leállítás", self.stop, "Leállítás|Megállítja a kotta lejátszását."),
            ("🖨 Nyomtatás…", self.print_score, "Nyomtatás|A kotta kinyomtatása a választott nyomtatón (A4)."),
            ("PDF mentése…", self.save_pdf, "PDF|A kotta mentése PDF fájlba - elküldhető, később nyomtatható."),
            ("MIDI mentése…", self.save_midi, "MIDI mentése|A kotta mentése MIDI fájlba (bármely zeneprogram megnyitja)."),
            ("ABC mentése…", self.save_abc, "ABC mentése|A kotta szöveges ABC formában."),
            ("MuseScore-ban", self.open_musescore, "MuseScore|Megnyitja a kottát az ingyenes MuseScore "
             "kottaszerkesztőben (ha telepítve van) - ott tovább szerkeszthető, kiadvány-minőségben nyomtatható."),
            ("Kód szerkesztése", self.toggle_code, "ABC kód|Megmutatja a kotta szöveges forrását: kézzel javítható, "
             "a változás azonnal látszik."),
        ]
        for i, (txt, fn, tip) in enumerate(acts):
            b = QPushButton(txt)
            b.setToolTip(tip_html(tip))
            b.clicked.connect(fn)
            if txt.startswith("🖨"):
                b.setObjectName("primary")
            g3l.addWidget(b, i // 2, i % 2)
        lv.addWidget(g3)
        self.ed_code = QPlainTextEdit()
        self.ed_code.setVisible(False)
        self.ed_code.setToolTip(tip_html("ABC forrás|Itt javíthatod a kottát szövegesen. Pár pillanattal a gépelés "
                                         "után frissül a kép."))
        self._code_timer = QTimer(self)
        self._code_timer.setSingleShot(True)
        self._code_timer.timeout.connect(self._code_changed)
        self.ed_code.textChanged.connect(lambda: self._code_timer.start(700))
        lv.addWidget(self.ed_code, 1)
        lv.addStretch()
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(left)
        sc.setFixedWidth(380)
        root.addWidget(sc)
        # ---- jobb: kotta nézet
        right = QVBoxLayout()
        self.lbl_lib = QLabel("")
        self.lbl_lib.setObjectName("dim")
        self.lbl_lib.setToolTip(tip_html("Állapot|A kottamegjelenítő betöltésének állapota."))
        right.addWidget(self.lbl_lib)
        if HAS_WEBENGINE:
            self.view = QWebEngineView()
            self.view.setToolTip("")
            st_ = self.view.settings()
            st_.setAttribute(QWebEngineSettings.LocalContentCanAccessFileUrls, True)
            st_.setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)
            st_.setAttribute(QWebEngineSettings.PlaybackRequiresUserGesture, False)
            self.view.loadFinished.connect(self._loaded)
            right.addWidget(self.view, 1)
        else:
            self.view = None
            lb = QLabel("A kottanézethez a Qt WebEngine kell (a teljes PySide6 csomag része): "
                        "python -m pip install --upgrade PySide6")
            lb.setWordWrap(True)
            right.addWidget(lb, 1)
        root.addLayout(right, 1)
        self._init_page()

    # ---------------------------------------------------------- könyvtár
    def _lib_ok(self) -> bool:
        f = SCORE_LIB_DIR / "abcjs-basic-min.js"
        if f.is_file():
            try:
                return hashlib.sha256(f.read_bytes()).hexdigest() == ABCJS_SHA256
            except Exception:
                return False
        return False

    def _init_page(self):
        if self.view is None:
            return
        SCORE_LIB_DIR.mkdir(exist_ok=True)
        if not self._lib_ok():
            self.lbl_lib.setText("A kottamegjelenítő első használatkor letöltődik (~0,5 MB)…")
            threading.Thread(target=self._download_lib, daemon=True).start()
            return
        self._load_html()

    def _download_lib(self):
        err = ""
        for url in ABCJS_URLS:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": f"{APP_NAME}/{APP_VERSION}"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    data = r.read()
                if hashlib.sha256(data).hexdigest() != ABCJS_SHA256:
                    err = "ellenőrzőösszeg eltérés"
                    continue
                (SCORE_LIB_DIR / "abcjs-basic-min.js").write_bytes(data)
                QTimer.singleShot(0, self._load_html)
                return
            except Exception as e:
                err = str(e)
        QTimer.singleShot(0, lambda: self.lbl_lib.setText(
            f"Nem sikerült letölteni a kottamegjelenítőt ({err}). Ellenőrizd az internetet, majd indítsd újra az appot."))

    def _load_html(self):
        (SCORE_LIB_DIR / "kotta.html").write_text(SCORE_HTML, encoding="utf-8")
        self.lbl_lib.setText("")
        self.view.load(QUrl.fromLocalFile(str(SCORE_LIB_DIR / "kotta.html")))

    def _loaded(self, ok):
        self._ready = ok
        if self._pending is not None:
            abc = self._pending
            self._pending = None
            self._render_js(abc)
        elif self.source_abc:
            self.refresh()

    # ---------------------------------------------------------- projekt
    def get_state(self) -> dict:
        return {"kind": self.source_kind, "abc": self.source_abc, "midi": self.midi_path, "label": self.lbl_src.text(),
                "title": self.ed_title.text(), "composer": self.ed_composer.text(),
                "voice": self.cb_voice.currentIndex(), "chords": self.chk_chords.isChecked(),
                "tr": self.sp_tr.value(), "scale": self.sp_scale.value(), "mode": self.cb_mode.currentIndex(),
                "grid": self.cb_grid.currentIndex(), "key": self.cb_key.currentIndex(), "bpl": self.sp_bpl.value()}

    def set_state(self, st: dict):
        ws = (self.ed_title, self.ed_composer, self.cb_voice, self.chk_chords, self.sp_tr, self.sp_scale,
              self.cb_mode, self.cb_grid, self.cb_key, self.sp_bpl)
        for w_ in ws:
            w_.blockSignals(True)
        try:
            self.ed_title.setText(st.get("title", ""))
            self.ed_composer.setText(st.get("composer", ""))
            self.cb_voice.setCurrentIndex(int(st.get("voice", 0)))
            self.chk_chords.setChecked(bool(st.get("chords", True)))
            self.sp_tr.setValue(int(st.get("tr", 0)))
            self.sp_scale.setValue(float(st.get("scale", 1.0)))
            self.cb_mode.setCurrentIndex(int(st.get("mode", 0)))
            self.cb_grid.setCurrentIndex(int(st.get("grid", 0)))
            self.cb_key.setCurrentIndex(int(st.get("key", 0)))
            self.sp_bpl.setValue(int(st.get("bpl", 4)))
        finally:
            for w_ in ws:
                w_.blockSignals(False)
        self.source_kind = st.get("kind", "")
        self.source_abc = st.get("abc", "")
        self.midi_path = st.get("midi", "")
        self.g_midi.setVisible(self.source_kind == "midi")
        self.lbl_src.setText(st.get("label") or "Még nincs betöltve kotta.")
        if self.ed_code.isVisible():
            self.ed_code.blockSignals(True)
            self.ed_code.setPlainText(self.current_abc())
            self.ed_code.blockSignals(False)
        if self.source_abc:
            self.refresh()
        elif self.view is not None and self._ready:
            self.view.page().runJavaScript("document.getElementById('paper').innerHTML='';"
                                           "document.getElementById('msg').innerText='';")
        else:
            self._pending = None

    def has_content(self) -> bool:
        return bool(self.source_abc)

    def clear(self, ask: bool = False):
        if ask and self.has_content() and QMessageBox.question(
                self, "Tiszta lap", "Kiürítsem a Kotta fület?") != QMessageBox.Yes:
            return
        self.set_state({})

    # ---------------------------------------------------------- források
    def set_abc(self, abc: str, label: str, title: str = ""):
        self.source_kind = "abc"
        self.source_abc = abc
        self.g_midi.setVisible(False)
        self.lbl_src.setText(label)
        t = re.search(r"^T:\s*(.+)$", abc, re.M)
        self.ed_title.setText(title or (t.group(1).strip() if t else ""))
        self.refresh()

    def from_abc_tab(self):
        abc = self.main.ed_abc.toPlainText().strip()
        if not abc:
            QMessageBox.information(self, "Kotta", "Az ABC-kotta fül üres. Generálj melody/full módban, vagy nyerj ki "
                                                   "egy kottát hangfájlból (ABC-kotta fül).")
            return
        self.set_abc(abc, "Forrás: ABC-kotta fül")

    def from_library(self):
        t = self.main._sel_track()
        if not t:
            QMessageBox.information(self, "Kotta", "Előbb jelölj ki egy dalt a Könyvtár fülön.")
            return
        if not (t.get("abc_path") and Path(t["abc_path"]).is_file()):
            QMessageBox.information(self, "Kotta", "Ehhez a dalhoz nincs mentett zenei terv (off módban készült?). "
                                                   "Tipp: bontsd sávokra a DAW-ban, és az Ének sávból készíts MIDI-t.")
            return
        self.set_abc(Path(t["abc_path"]).read_text(encoding="utf-8"), f"Forrás: {t.get('title')}", t.get("title") or "")

    def open_abc(self):
        f, _ = QFileDialog.getOpenFileName(self, "ABC kotta", str(SCORE_DIR), "ABC (*.abc *.txt)")
        if f:
            self.set_abc(Path(f).read_text(encoding="utf-8", errors="replace"), f"Forrás: {Path(f).name}")

    def open_midi(self, path: str | None = None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "MIDI fájl", str(DAW_DIR / "midi") if (DAW_DIR / "midi").is_dir()
                                                  else str(APP_DIR), "MIDI (*.mid *.midi)")
            if not path:
                return
        self.midi_path = path
        self.source_kind = "midi"
        self.g_midi.setVisible(True)
        self.ed_title.setText(Path(path).stem.split("_", 2)[-1])
        self.lbl_src.setText(f"Forrás: {Path(path).name} (MIDI)")
        self._reconvert()

    def last_daw_midi(self):
        d = DAW_DIR / "midi"
        mids = sorted(d.glob("*.mid"), key=lambda p: p.stat().st_mtime) if d.is_dir() else []
        if not mids:
            QMessageBox.information(self, "Kotta", "Még nem készült MIDI a DAW-ban. Ott: jobb klikk egy klipen → "
                                                   "♫ MIDI kotta készítése…")
            return
        self.open_midi(str(mids[-1]))

    def _reconvert(self):
        if self.source_kind != "midi" or not self.midi_path:
            return
        try:
            abc = midi_to_abc(self.midi_path, mode=self.cb_mode.currentData(), grid=self.cb_grid.currentData(),
                              title=self.ed_title.text(), key_override=self.cb_key.currentData(),
                              bars_per_line=self.sp_bpl.value())
        except Exception as e:
            QMessageBox.warning(self, "MIDI", f"Nem sikerült kottává alakítani:\n{e}")
            return
        self.source_abc = abc
        self.refresh()

    # ---------------------------------------------------------- megjelenítés
    def current_abc(self) -> str:
        abc = self.source_abc
        if not abc:
            return ""
        if self.cb_voice.currentData():
            abc = abc_filter_voices(abc, self.cb_voice.currentData())
        if not self.chk_chords.isChecked():
            abc = abc_strip_chords(abc)
        abc = abc_set_header(abc, "T", self.ed_title.text().strip())
        abc = abc_set_header(abc, "C", self.ed_composer.text().strip())
        return abc

    def refresh(self):
        abc = self.current_abc()
        if not abc:
            return
        if self.ed_code.isVisible() and self.ed_code.toPlainText() != abc:
            self.ed_code.blockSignals(True)
            self.ed_code.setPlainText(abc)
            self.ed_code.blockSignals(False)
        self._render_js(abc)

    def _render_js(self, abc: str):
        if self.view is None:
            return
        if not self._ready:
            self._pending = abc
            return
        opts = {"transpose": self.sp_tr.value(), "scale": self.sp_scale.value(), "staffwidth": 760}
        self.view.page().runJavaScript(f"render({json.dumps(abc)}, {json.dumps(opts)})")

    def toggle_code(self):
        on = not self.ed_code.isVisible()
        self.ed_code.setVisible(on)
        if on:
            self.ed_code.blockSignals(True)
            self.ed_code.setPlainText(self.current_abc())
            self.ed_code.blockSignals(False)

    def _code_changed(self):
        txt = self.ed_code.toPlainText()
        if txt.strip():
            self.source_kind = "abc"
            self.source_abc = txt
            self.g_midi.setVisible(False)
            self._render_js(self.current_abc())

    # ---------------------------------------------------------- műveletek
    def _need(self) -> bool:
        if not self.source_abc or self.view is None or not self._ready:
            QMessageBox.information(self, "Kotta", "Előbb tölts be egy kottát (bal felső gombok).")
            return False
        return True

    def play(self):
        if not self._need():
            return
        if self.main.player:
            self.main.player.pause()
        if getattr(self.main, "daw", None):
            self.main.daw.engine.pause()
        self.view.page().runJavaScript("play()", 0, lambda r: None)

    def stop(self):
        if self.view is not None:
            self.view.page().runJavaScript("stop()")

    def _default_name(self, ext):
        return str(SCORE_DIR / f"{safe_name(self.ed_title.text() or 'kotta', 60)}.{ext}")

    def save_pdf(self, path: str | None = None):
        if not self._need():
            return
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, "PDF mentése", self._default_name("pdf"), "PDF (*.pdf)")
            if not path:
                return
        layout = QPageLayout(QPageSize(QPageSize.A4), QPageLayout.Portrait, QMarginsF(12, 12, 12, 12), QPageLayout.Millimeter)
        self._pdf_target = path
        if not getattr(self, "_pdf_connected", False):
            self.view.page().pdfPrintingFinished.connect(self._pdf_done)
            self._pdf_connected = True
        self.view.page().printToPdf(path, layout)

    def _pdf_done(self, path, ok):
        if ok:
            self.main.log(f"Kotta PDF mentve: {path}")
            if not getattr(self, "_silent", False):
                box = QMessageBox(self)
                box.setWindowTitle("PDF kész")
                box.setText(f"Elmentve:\n{path}")
                b = box.addButton("Megnyitás", QMessageBox.ActionRole)
                box.addButton("OK", QMessageBox.AcceptRole)
                box.exec()
                if box.clickedButton() is b:
                    try:
                        os.startfile(path)  # type: ignore[attr-defined]
                    except Exception:
                        open_in_explorer(Path(path))
        else:
            QMessageBox.warning(self, "PDF", "Nem sikerült a PDF mentése.")

    def print_score(self):
        if not self._need():
            return
        from PySide6.QtPrintSupport import QPrintDialog, QPrinter
        self._printer = QPrinter(QPrinter.HighResolution)
        self._printer.setPageSize(QPageSize(QPageSize.A4))
        self._printer.setPageMargins(QMarginsF(12, 12, 12, 12), QPageLayout.Millimeter)
        dlg = QPrintDialog(self._printer, self)
        dlg.setWindowTitle("Kotta nyomtatása")
        if dlg.exec() != QDialog.Accepted:
            return
        if not getattr(self, "_print_connected", False):
            self.view.printFinished.connect(lambda ok: self.main.log("Kotta nyomtatva." if ok else "Nyomtatási hiba."))
            self._print_connected = True
        self.view.print(self._printer)

    def save_midi(self):
        if not self._need():
            return
        path, _ = QFileDialog.getSaveFileName(self, "MIDI mentése", self._default_name("mid"), "MIDI (*.mid)")
        if not path:
            return
        self._midi_to(path, lambda p: self.main.log(f"Kotta MIDI mentve: {p}"))

    def _midi_to(self, path, done):
        def cb(r):
            if not isinstance(r, str) or r.startswith("ERR"):
                QMessageBox.warning(self, "MIDI", f"Nem sikerült MIDI-t készíteni: {r}")
                return
            Path(path).write_bytes(base64.b64decode(r))
            done(path)
        self.view.page().runJavaScript("midiB64()", 0, cb)

    def save_abc(self):
        abc = self.current_abc()
        if not abc:
            return
        path, _ = QFileDialog.getSaveFileName(self, "ABC mentése", self._default_name("abc"), "ABC (*.abc)")
        if path:
            Path(path).write_text(abc, encoding="utf-8")

    def open_musescore(self):
        if not self._need():
            return
        ms = find_musescore()
        if not ms:
            QMessageBox.information(self, "MuseScore", "A MuseScore nincs telepítve (vagy nem találom).\n"
                                    "Ingyenesen letölthető: https://musescore.org\n\n"
                                    "Addig a 'MIDI mentése…' gombbal mentett fájlt bármikor megnyithatod vele.")
            return
        tmp = SCORE_DIR / f"{safe_name(self.ed_title.text() or 'kotta', 60)}.mid"
        if self.source_kind == "midi" and self.midi_path and not self.sp_tr.value():
            subprocess.Popen([ms, self.midi_path])
            return
        self._midi_to(str(tmp), lambda p: subprocess.Popen([ms, p]))


def score_missing_widget() -> QWidget:
    w = QWidget()
    l = QVBoxLayout(w)
    lb = QLabel("<h2>A kottanézethez a Qt WebEngine kell</h2><p>Frissítsd a PySide6 csomagot:</p>"
                "<pre>python -m pip install --upgrade PySide6</pre>")
    lb.setTextInteractionFlags(Qt.TextSelectableByMouse)
    l.addWidget(lb)
    l.addStretch()
    return w


# ===========================================================================
# ELSŐ LÉPÉSEK KALAUZ - kiemelő réteg magyarázó buborékkal
# ===========================================================================
from PySide6.QtCore import QPoint, QRect
from PySide6.QtGui import QRegion



# ===========================================================================
# ÁTDOLGOZÁS - meglévő dal (MP3/WAV) újragenerálása új stílusban a YuE2-vel
# ===========================================================================
REMAKE_DIR = APP_DIR / "atdolgozas"

ASR_LANGS = [("Automatikus felismerés", ""), ("Magyar", "Hungarian"), ("Angol", "English"), ("Német", "German"),
             ("Francia", "French"), ("Olasz", "Italian"), ("Spanyol", "Spanish"), ("Orosz", "Russian"),
             ("Lengyel", "Polish"), ("Cseh", "Czech"), ("Portugál", "Portuguese"), ("Holland", "Dutch")]

REMAKE_STYLES = {
    "Jazz": "jazz, smooth, piano, upright bass, brushed drums, saxophone",
    "Blues": "delta blues, smoky, slide guitar, harmonica, live recording",
    "Rock": "rock, energetic, electric guitar, bass guitar, drums, polished mix",
    "Akusztikus": "acoustic folk, warm, acoustic guitar, soft percussion, intimate",
    "Pop": "pop, uplifting, synth, drums, polished mix",
    "Elektronikus": "synthwave, nostalgic, synth, 808, wide stereo",
    "Zenekari": "cinematic, orchestral, epic, strings, brass section, timpani",
    "Country": "country, warm, acoustic guitar, fiddle, pedal steel",
    "Lo-fi": "lo-fi, chill, rhodes, soft drums, vinyl crackle",
    "Gospel": "gospel, uplifting, choir, organ, piano",
    "Bossa nova": "bossa nova, calm, nylon guitar, soft percussion",
    "Metal": "metal, aggressive, distorted electric guitar, double bass drums",
}


def detect_vocal_phrases(mono16k, sr: int = 16000) -> list[tuple[float, float]]:
    """Ének-frázisok (sorok) keresése a hangerő alapján: ahol az énekes levegőt vesz / szünetet tart, ott vágunk.
    Bemenet: leválasztott ének, mono float tömb. Kimenet: [(kezdet_mp, vég_mp), …]."""
    x = np.asarray(mono16k, dtype=np.float32).ravel()
    hop = int(sr * 0.02)
    n = len(x) // hop
    if n < 10:
        return []
    rms = np.sqrt(np.mean(x[:n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    db = np.convolve(db, np.ones(3) / 3, mode="same")          # 60 ms simítás
    floor, top = np.percentile(db, 20), np.percentile(db, 97)
    if top - floor < 6:
        return []
    thr = floor + 0.35 * (top - floor)
    on = db > thr
    segs, i = [], 0
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            segs.append([i, j])
            i = j
        else:
            i += 1
    merged = []                                                  # rövid (< 0,2 mp) szünetek összevonása
    for a, b in segs:
        if merged and a - merged[-1][1] < 10:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    merged = [m_ for m_ in merged if m_[1] - m_[0] >= 20]       # < 0,4 mp: zaj
    out = []

    def split(a, b):                                             # túl hosszú (> 9 mp) frázis: a leghalkabb ponton vágjuk
        if b - a <= 450:
            out.append((a, b))
            return
        lo, hi = a + 100, b - 100
        k = lo + int(np.argmin(db[lo:hi]))
        split(a, k)
        split(k, b)

    for a, b in merged:
        split(a, b)
    return [(a * hop / sr, b * hop / sr) for a, b in out]


def structure_timed_lyrics(lines: list[dict], total: float = 0.0) -> str:
    """Időzített sorokból tagolt dalszöveg: hosszú szünetnél új szakasz, az ismétlődő szakasz a refrén."""
    lines = [l for l in lines if (l.get("text") or "").strip()]
    if not lines:
        return ""
    blocks, cur = [], [lines[0]]
    for prev, l in zip(lines, lines[1:]):
        gap = l["start"] - prev["end"]
        if gap > 2.8 or len(cur) >= 8:
            blocks.append(cur)
            cur = [l]
        else:
            cur.append(l)
    blocks.append(cur)
    norm = lambda t: re.sub(r"[^\w ]", "", t.lower()).strip()
    counts: dict[str, int] = {}
    for b in blocks:
        for l in b:
            counts[norm(l["text"])] = counts.get(norm(l["text"]), 0) + 1
    out = []
    if lines[0]["start"] > 6:
        out.append("[Intro]")
    for b in blocks:
        rep_ = sum(1 for l in b if counts[norm(l["text"])] > 1)
        out.append("[Chorus]" if rep_ >= max(1, len(b) // 2 + (len(b) % 2)) else "[Verse]")
        for l in b:
            t = l["text"].strip().rstrip(",;")
            out.append(t[:1].upper() + t[1:])
    if total and total - lines[-1]["end"] > 6:
        out.append("[Outro]")
    return "\n".join(out)


def clean_asr_text(text: str) -> tuple[str, str]:
    text = (text or "").strip()
    lang = ""
    m_ = re.match(r"^\s*language\s+(\w+)\s*<asr_text>(.*)$", text, re.S | re.I)
    if m_:
        lang, text = m_.group(1), m_.group(2).strip()
    return text.replace("<asr_text>", " ").strip(), lang


def auto_structure_lyrics(text: str) -> str:
    """Felismert (tagolatlan) szöveg -> sorok + [Verse]/[Chorus] címkék. Az ismétlődő sorok refrénnek számítanak."""
    t = re.sub(r"\s+", " ", text).strip()
    if not t:
        return ""
    if re.search(r"^\s*\[.+\]\s*$", text, re.M):
        return text.strip()
    # mondatokra, majd max ~8 szavas sorokra bontás
    parts = []
    for sent in re.split(r"(?<=[.!?;])\s+", t):       # mondatok; hosszú mondatnál vesszőnél is törünk
        if len(sent.split()) > 10:
            sub, cur = re.split(r"(?<=,)\s+", sent), ""
            for x in sub:
                cur = (cur + " " + x).strip()
                if len(cur.split()) >= 4:
                    parts.append(cur)
                    cur = ""
            if cur:
                parts.append(cur)
        else:
            parts.append(sent)
    lines = []
    for p in parts:
        words = p.split()
        while words:
            n = 8 if len(words) > 10 else len(words)
            lines.append(" ".join(words[:n]).strip(" ,;"))
            words = words[n:]
    lines = [l for l in lines if l]
    norm = [re.sub(r"[^\w ]", "", l.lower()) for l in lines]
    counts = {}
    for n_ in norm:
        counts[n_] = counts.get(n_, 0) + 1
    out, block, block_is_chorus = [], [], None

    def flush():
        if block:
            out.append("[Chorus]" if block_is_chorus else "[Verse]")
            out.extend(block)

    for l, n_ in zip(lines, norm):
        is_ch = counts[n_] > 1
        if block and (is_ch != block_is_chorus or len(block) >= 4):
            flush()
            block = []
        if not block:
            block_is_chorus = is_ch
        block.append(l[:1].upper() + l[1:])
    flush()
    out.append("[Outro]")
    return "\n".join(out)


class RemakePanel(QWidget):
    STEP_NAMES = ["Dal", "Ének leválasztása", "Dallam + akkordok", "Dalszöveg", "Stílus", "Generálás"]

    def __init__(self, main):
        super().__init__()
        self.main = main
        self.src = ""            # eredeti fájl
        self.work_src = ""       # (részlet) WAV, amivel dolgozunk
        self.vocals = ""
        self.abc = ""
        self.src_len = 0.0
        self.timed: list[dict] = []
        self.auto_chain = False
        self.jobs: set[int] = set()
        REMAKE_DIR.mkdir(exist_ok=True)
        root = QHBoxLayout(self)
        form = QWidget()
        fl = QVBoxLayout(form)

        intro = QLabel("Tölts be egy dalt, és a YuE2 <b>a dallamát és akkordjait megtartva</b> új stílusban, új "
                       "hangszereléssel, új énekessel újrakészíti. Menj végig a lépéseken fentről lefelé - vagy a 2–4. "
                       "lépést futtasd egyben a ⚡ gombbal.")
        intro.setWordWrap(True)
        intro.setObjectName("dim")
        intro.setToolTip(tip_html("Átdolgozás|Meglévő dal újrakészítése új stílusban. A Kalauz (jobb alul) "
                                  "végigvezet a lépéseken."))
        r0 = QHBoxLayout()
        r0.addWidget(intro, 1)
        self.btn_clear = clear_button(lambda: self.clear(ask=True), "Az Átdolgozás fül minden mezőjét kiüríti: dal, "
                                      "dallam, szöveg, stílusok, eredménylista, napló. A kész dalok a Könyvtárban maradnak.")
        r0.addWidget(self.btn_clear, 0, Qt.AlignTop)
        fl.addLayout(r0)

        # 1. dal
        g1 = QGroupBox("1. A dal")
        l1 = QVBoxLayout(g1)
        r = QHBoxLayout()
        self.ed_file = QLineEdit()
        self.ed_file.setReadOnly(True)
        self.ed_file.setPlaceholderText("Válassz egy dalt (MP3, WAV, FLAC, OGG, M4A)…")
        self.ed_file.setToolTip(tip_html("A dal|Az átdolgozandó dal fájlja. Bármilyen gyakori hangformátum jó."))
        b = QPushButton("Tallózás…")
        b.setToolTip(tip_html("Tallózás|Dal kiválasztása a gépről."))
        b.clicked.connect(self.pick_file)
        b2 = QPushButton("Könyvtárból")
        b2.setToolTip(tip_html("Könyvtárból|A Könyvtár fülön kijelölt (generált) dalt veszi át."))
        b2.clicked.connect(self.from_library)
        self.btn_listen = QPushButton("▶ Meghallgatás")
        self.btn_listen.setToolTip(tip_html("Meghallgatás|Lejátssza a dalt (vagy a kijelölt részletet) az alsó lejátszósávon."))
        self.btn_listen.clicked.connect(lambda: self.src and self.main.play_file(self.work_src or self.src, Path(self.src).stem))
        r.addWidget(self.ed_file, 1)
        l1.addLayout(r)
        r = QHBoxLayout()
        for w_ in (b, b2, self.btn_listen):
            r.addWidget(w_)
        r.addStretch()
        l1.addLayout(r)
        r = QHBoxLayout()
        self.chk_part = QCheckBox("Csak egy részlet:")
        self.chk_part.setToolTip(tip_html("Részlet|Hosszú dalnál érdemes csak egy részt (pl. versszak + refrén, "
                                          "60–90 mp) átdolgozni: gyorsabb és pontosabb."))
        self.sp_from = QDoubleSpinBox()
        self.sp_to = QDoubleSpinBox()
        for sp in (self.sp_from, self.sp_to):
            sp.setRange(0, 3600)
            sp.setDecimals(1)
            sp.setSuffix(" mp")
            sp.setEnabled(False)
        self.sp_to.setValue(90)
        self.sp_from.setToolTip(tip_html("Részlet kezdete|Hányadik másodperctől."))
        self.sp_to.setToolTip(tip_html("Részlet vége|Hányadik másodpercig."))
        self.chk_part.toggled.connect(lambda on: (self.sp_from.setEnabled(on), self.sp_to.setEnabled(on), self._prepare_source()))
        self.sp_from.editingFinished.connect(self._prepare_source)
        self.sp_to.editingFinished.connect(self._prepare_source)
        self.lbl_info = QLabel("")
        self.lbl_info.setObjectName("dim")
        self.lbl_info.setToolTip(tip_html("Adatok|A dal hossza, becsült tempója."))
        r.addWidget(self.chk_part)
        r.addWidget(QLabel("tól"))
        r.addWidget(self.sp_from)
        r.addWidget(QLabel("ig"))
        r.addWidget(self.sp_to)
        r.addStretch()
        l1.addLayout(r)
        self.lbl_info.setWordWrap(True)
        l1.addWidget(self.lbl_info)
        fl.addWidget(g1)

        # ⚡ egyben
        self.btn_auto = QPushButton("⚡ 2–4. lépés egyben (ének → dallam → szöveg)")
        self.btn_auto.setObjectName("primary")
        self.btn_auto.setToolTip(tip_html("Egyben|Egymás után elvégzi az előkészítést. Pár perc. Utána nézd át a "
                                          "dallamot és a szöveget, válassz stílust, és indítsd a generálást."))
        self.btn_auto.clicked.connect(self.run_all)
        fl.addWidget(self.btn_auto)

        # 2. ének
        g2 = QGroupBox("2. Ének leválasztása (ajánlott)")
        l2 = QHBoxLayout(g2)
        self.btn_sep = QPushButton("✂ Ének leválasztása")
        self.btn_sep.setToolTip(tip_html("Ének leválasztása|A HTDemucs modell kiszedi az éneket a zenéből. A "
                                         "dalszöveg-felismerés így sokkal pontosabb. Ha kihagyod, a teljes dalból dolgozik."))
        self.btn_sep.clicked.connect(self.run_sep)
        self.btn_play_voc = QPushButton("▶ Ének")
        self.btn_play_voc.setToolTip(tip_html("Ének meghallgatása|A leválasztott énekhang."))
        self.btn_play_voc.setEnabled(False)
        self.btn_play_voc.clicked.connect(lambda: self.vocals and self.main.play_file(self.vocals, "Leválasztott ének"))
        self.lbl_sep = QLabel("")
        self.lbl_sep.setObjectName("dim")
        self.lbl_sep.setToolTip(tip_html("Állapot|Az ének-leválasztás állapota."))
        l2.addWidget(self.btn_sep)
        l2.addWidget(self.btn_play_voc)
        l2.addWidget(self.lbl_sep, 1)
        fl.addWidget(g2)

        # 3. dallam
        g3 = QGroupBox("3. Dallam és akkordok (SheetSage2)")
        l3 = QVBoxLayout(g3)
        r = QHBoxLayout()
        self.btn_abc = QPushButton("🎼 Dallam kinyerése")
        self.btn_abc.setToolTip(tip_html("Dallam kinyerése|A SheetSage2 kiolvassa a dal dallamát és akkordjait "
                                         "zenei tervvé (ABC-kotta). Ezt fogja követni a YuE2."))
        self.btn_abc.clicked.connect(self.run_abc)
        self.btn_abc_x2 = QPushButton("Hosszabbítás ×2")
        self.btn_abc_x2.setToolTip(tip_html("Hosszabbítás|A tervet kétszer egymás után ismétli - hosszabb dal lesz."))
        self.btn_abc_x2.clicked.connect(self._abc_x2)
        self.btn_abc_score = QPushButton("Megnézem kottaként")
        self.btn_abc_score.setToolTip(tip_html("Kotta|A kinyert dallamot kottaként mutatja a Kotta fülön."))
        self.btn_abc_score.clicked.connect(self._abc_to_score)
        self.lbl_abc = QLabel("")
        self.lbl_abc.setObjectName("dim")
        self.lbl_abc.setToolTip(tip_html("Terv adatai|Sorok, ütemek, tempó, hangnem."))
        for w_ in (self.btn_abc, self.btn_abc_x2, self.btn_abc_score):
            r.addWidget(w_)
        r.addStretch()
        l3.addLayout(r)
        self.lbl_abc.setWordWrap(True)
        l3.addWidget(self.lbl_abc)
        self.ed_abc = QPlainTextEdit()
        self.ed_abc.setMaximumHeight(110)
        self.ed_abc.setPlaceholderText("Itt jelenik meg a kinyert zenei terv (ABC-kotta). Szerkeszthető.")
        self.ed_abc.setToolTip(tip_html("Zenei terv|A kinyert dallam és akkordok ABC formában. Ha üres, a YuE2 "
                                        "saját dallamot ír (akkor nem igazi átdolgozás)."))
        self.ed_abc.textChanged.connect(lambda: self.lbl_abc.setText(abc_stats(self.ed_abc.toPlainText())))
        l3.addWidget(self.ed_abc)
        fl.addWidget(g3)

        # 4. szöveg
        g4 = QGroupBox("4. Dalszöveg")
        l4 = QVBoxLayout(g4)
        r = QHBoxLayout()
        self.cb_lang = QComboBox()
        for t_, d_ in ASR_LANGS:
            self.cb_lang.addItem(t_, d_)
        self.cb_lang.setToolTip(tip_html("Nyelv|A dal nyelve. Ha tudod, add meg - pontosabb lesz a felismerés."))
        self.btn_asr = QPushButton("🎤 Szöveg felismerése (Qwen3-ASR)")
        self.btn_asr.setToolTip(tip_html("Szövegfelismerés|A Qwen3-ASR modell leírja, mit énekelnek (a leválasztott "
                                         "énekből, ha van). Az eredményt mindig nézd át!"))
        self.btn_asr.clicked.connect(self.run_asr)
        b_struct = QPushButton("Automatikus tagolás")
        b_struct.setToolTip(tip_html("Tagolás|Sorokra bontja a szöveget, és [Verse] / [Chorus] címkéket tesz bele "
                                     "(az ismétlődő sorokat refrénnek veszi)."))
        b_struct.clicked.connect(lambda: self.ed_lyrics.setPlainText(auto_structure_lyrics(self.ed_lyrics.toPlainText())))
        b_new = QPushButton("🤖 Új szöveg (Ollama)")
        b_new.setToolTip(tip_html("Új szöveg|A helyi Ollama modell ugyanerre a szerkezetre új szöveget ír (vagy "
                                  "lefordítja). A modellt a Beállításokban adod meg."))
        b_new.clicked.connect(self.ollama_rewrite)
        r.addWidget(QLabel("Nyelv:"))
        r.addWidget(self.cb_lang)
        r.addWidget(self.btn_asr)
        r.addStretch()
        l4.addLayout(r)
        r = QHBoxLayout()
        self.chk_timed = QCheckBox("Időzített felismerés (sorok az eredeti levegővételeinél)")
        self.chk_timed.setChecked(True)
        self.chk_timed.setToolTip(tip_html("Időzített felismerés|A leválasztott éneket ott darabolja fel, ahol az "
                                           "énekes levegőt vesz, és minden darabot külön ismer fel. Így minden sor "
                                           "pontosan egy énekelt frázis, a szakaszhatárok a hosszabb szüneteknél "
                                           "vannak - a szöveg sokkal jobban illik a dallamhoz, kevesebb az ugrálás.\n"
                                           "Leválasztott ének kell hozzá (2. lépés). Kicsit lassabb."))
        self.btn_timing = QPushButton("⏱ Időzítés…")
        self.btn_timing.setEnabled(False)
        self.btn_timing.setToolTip(tip_html("Időzítés|Táblázat: melyik sor hányadik másodpercnél és hányadik "
                                            "ütemnél kezdődik az eredetiben. Ellenőrzéshez, és ha kézzel "
                                            "szeretnéd igazítani a sorokat."))
        self.btn_timing.clicked.connect(self.show_timing)
        r.addWidget(self.chk_timed)
        r.addWidget(self.btn_timing)
        r.addStretch()
        l4.addLayout(r)
        r = QHBoxLayout()
        r.addWidget(b_struct)
        r.addWidget(b_new)
        r.addStretch()
        l4.addLayout(r)
        self.ed_lyrics = QPlainTextEdit()
        self.ed_lyrics.setMinimumHeight(150)
        self.ed_lyrics.setPlaceholderText("[Verse]\nA felismert vagy saját szöveg…\n[Chorus]\n…")
        self.ed_lyrics.setToolTip(tip_html("Dalszöveg|Ezt fogja énekelni az új változat. Lehet az eredeti, egy "
                                           "fordítás, vagy teljesen új szöveg. A sorok ritmusa nagyjából illeszkedjen a "
                                           "dallamhoz."))
        l4.addWidget(self.ed_lyrics)
        self.chk_instr = QCheckBox("Instrumentális változat (ének nélkül)")
        self.chk_instr.setToolTip(tip_html("Instrumentális|Szöveg nélkül, csak a dallam hangszerekkel."))
        l4.addWidget(self.chk_instr)
        fl.addWidget(g4)

        # 5. stílus
        g5 = QGroupBox("5. Új stílus - több is választható, mindegyikből készül változat")
        l5 = QVBoxLayout(g5)
        grid = QGridLayout()
        self.style_checks = {}
        for i, (name, tags) in enumerate(REMAKE_STYLES.items()):
            cb = QCheckBox(name)
            cb.setToolTip(tip_html(f"{name}|Tagek: {tags}"))
            self.style_checks[name] = cb
            grid.addWidget(cb, i // 4, i % 4)
        l5.addLayout(grid)
        r = QHBoxLayout()
        self.cb_voice = QComboBox()
        for t_, d_ in (("Énekhang: mindegy", ""), ("Férfi ének", "male vocal"), ("Női ének", "female vocal"),
                       ("Kórus", "choir")):
            self.cb_voice.addItem(t_, d_)
        self.cb_voice.setToolTip(tip_html("Énekhang|Milyen hang énekeljen az új változatban."))
        self.cb_lang_tag = QComboBox()
        for t_, d_ in (("Ének nyelve: angol", "English"), ("Ének nyelve: magyar", "Hungarian"),
                       ("Ének nyelve: német", "German"), ("Ének nyelve: ne add meg", "")):
            self.cb_lang_tag.addItem(t_, d_)
        self.cb_lang_tag.setToolTip(tip_html("Ének nyelve|Az első stílus-tag. Igazítsd a dalszöveg nyelvéhez."))
        r.addWidget(self.cb_voice)
        r.addWidget(self.cb_lang_tag)
        r.addStretch()
        l5.addLayout(r)
        self.ed_style = QLineEdit()
        self.ed_style.setPlaceholderText("Saját stílus is (angol tagek, vesszővel) - pl. reggae, relaxed, skank guitar")
        self.ed_style.setToolTip(tip_html("Saját stílus|Ha a fentiek közül egyik sem jó, ide írhatsz sajátot. "
                                          "Ebből is készül változat."))
        l5.addWidget(self.ed_style)
        fl.addWidget(g5)

        # 6. generálás
        g6 = QGroupBox("6. Generálás")
        l6v = QVBoxLayout(g6)
        l6 = QHBoxLayout()
        l6v.addLayout(l6)
        self.sp_n = QSpinBox()
        self.sp_n.setRange(1, 10)
        self.sp_n.setValue(2)
        self.sp_n.setToolTip(tip_html("Változatok|Stílusonként ennyi változat készül (különböző seeddel)."))
        self.cb_cot = QComboBox()
        self.cb_cot.addItem("Dallam követése (melody)", "melody")
        self.cb_cot.addItem("Dallam + akkordok (full)", "full")
        self.cb_cot.setToolTip(tip_html("Követés módja|melody: a dallamot követi. full: a dallamot és az akkordokat is."))
        self.ed_title = QLineEdit()
        self.ed_title.setPlaceholderText("Cím (opcionális)")
        self.ed_title.setToolTip(tip_html("Cím|A változatok neve a Könyvtárban; mögé kerül a stílus neve."))
        self.btn_go = QPushButton("▶ Átdolgozás indítása")
        self.btn_go.setObjectName("primary")
        self.btn_go.setToolTip(tip_html("Indítás|Minden kiválasztott stílusból elkészíti a változatokat a "
                                        "feladatsorban. Az eredmények a jobb oldalon és a Könyvtárban jelennek meg."))
        self.btn_go.clicked.connect(self.start)
        self.chk_same_len = QCheckBox("Hossz = eredeti")
        self.chk_same_len.setChecked(True)
        self.chk_same_len.setToolTip(tip_html("Hossz = eredeti|Az új változat nagyjából olyan hosszú lesz, mint az "
                                              "eredeti dal (vagy a kijelölt részlet). Kikapcsolva a Generálás fül "
                                              "„Dal hossza” beállítása érvényes."))
        l6.addWidget(QLabel("Stílusonként:"))
        l6.addWidget(self.sp_n)
        l6.addWidget(self.cb_cot)
        l6.addWidget(self.ed_title, 1)
        l6b = QHBoxLayout()
        l6b.addWidget(self.chk_same_len)
        l6b.addStretch()
        l6b.addWidget(self.btn_go)
        l6v.addLayout(l6b)
        fl.addWidget(g6)
        fl.addStretch()
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(form)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        root.addWidget(sc, 3)

        # jobb oldal: állapot + eredmények
        right = QVBoxLayout()
        gs = QGroupBox("Hol tartunk?")
        gl = QVBoxLayout(gs)
        self.step_lbls = []
        for i, n in enumerate(self.STEP_NAMES):
            lb = QLabel(f"○  {i + 1}. {n}")
            lb.setToolTip(tip_html("Lépés állapota|○ még nincs kész · ⏳ folyamatban · ✓ kész · – kihagyva"))
            self.step_lbls.append(lb)
            gl.addWidget(lb)
        right.addWidget(gs)
        gr = QGroupBox("Elkészült változatok")
        grl = QVBoxLayout(gr)
        self.lst = QListWidget()
        self.lst.setToolTip(tip_html("Változatok|Dupla kattintás: lejátszás. A Könyvtárban is megtalálod őket."))
        self.lst.itemDoubleClicked.connect(self._play_item)
        grl.addWidget(self.lst)
        b = QPushButton("▶ Eredeti dal")
        b.setToolTip(tip_html("Összehasonlítás|Az eredeti dal (részlet) lejátszása, hogy összehasonlíthasd."))
        b.clicked.connect(lambda: self.src and self.main.play_file(self.work_src or self.src, "Eredeti"))
        grl.addWidget(b)
        right.addWidget(gr, 1)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(150)
        self.log.setToolTip(tip_html("Napló|Az átdolgozás lépéseinek üzenetei."))
        right.addWidget(self.log)
        rw = QWidget()
        rw.setLayout(right)
        rw.setMaximumWidth(460)
        rw.setMinimumWidth(300)
        root.addWidget(rw, 2)

    # ---------------------------------------------------------- segéd
    def say(self, msg):
        self.log.appendPlainText(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}")

    def step(self, i, state):
        sym = {"todo": "○", "run": "⏳", "ok": "✓", "skip": "–", "err": "✗"}[state]
        col = {"todo": "#a8a9b3", "run": "#facc15", "ok": "#4ade80", "skip": "#80828e", "err": "#f87171"}[state]
        self.step_lbls[i].setText(f"{sym}  {i + 1}. {self.STEP_NAMES[i]}")
        self.step_lbls[i].setStyleSheet(f"color:{col}; font-weight:{'700' if state == 'run' else '400'};")

    def _need_src(self) -> bool:
        if not self.src:
            QMessageBox.information(self, "Átdolgozás", "Előbb válassz egy dalt (1. lépés).")
            return False
        return True

    # ---------------------------------------------------------- 1. dal
    def pick_file(self):
        f, _ = QFileDialog.getOpenFileName(self, "Átdolgozandó dal", str(Path.home() / "Music"),
                                           "Hang (*.mp3 *.wav *.flac *.ogg *.m4a *.aac);;Minden (*)")
        if f:
            self.set_source(f)

    def from_library(self):
        t = self.main._sel_track()
        if not t:
            QMessageBox.information(self, "Átdolgozás", "Jelölj ki egy dalt a Könyvtár fülön.")
            return
        self.set_source(t["wav_path"])

    def set_source(self, path):
        self.src = path
        self.vocals = ""
        self.ed_file.setText(path)
        self.lst.clear()
        for i in range(6):
            self.step(i, "todo")
        self.btn_play_voc.setEnabled(False)
        self.lbl_sep.setText("")
        if not self.ed_title.text():
            self.ed_title.setText(Path(path).stem[:50])
        self._prepare_source()

    def _prepare_source(self):
        if not self.src:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            data = read_audio(self.src)
            total = data.shape[1] / DAW_SR
            seg = data
            if self.chk_part.isChecked():
                a = max(0.0, min(self.sp_from.value(), total))
                b = max(a + 1.0, min(self.sp_to.value(), total))
                seg = data[:, int(a * DAW_SR):int(b * DAW_SR)]
                out = REMAKE_DIR / f"reszlet_{safe_name(Path(self.src).stem, 40)}_{int(a)}-{int(b)}.wav"
                with PBAudioFile(str(out), "w", DAW_SR, 2, bit_depth=16) as f:
                    f.write(np.ascontiguousarray(seg))
                self.work_src = str(out)
            elif Path(self.src).suffix.lower() != ".wav":     # MP3 stb. -> WAV munkapéldány (ffmpeg nem kell)
                out = REMAKE_DIR / f"{safe_name(Path(self.src).stem, 50)}.wav"
                with PBAudioFile(str(out), "w", DAW_SR, 2, bit_depth=16) as f:
                    f.write(np.ascontiguousarray(seg))
                self.work_src = str(out)
            else:
                self.work_src = self.src
            bpm = estimate_bpm(seg)
            L = seg.shape[1] / DAW_SR
            self.src_len = L
            self.lbl_info.setText(f"Hossz: {int(L // 60)}:{int(L % 60):02d}" + (f" (teljes: {int(total // 60)}:{int(total % 60):02d})"
                                  if self.chk_part.isChecked() else "") + f" · tempó kb. {bpm:.0f} BPM")
            if L > 150:
                self.lbl_info.setText(self.lbl_info.text() + " · tipp: hosszú, érdemes részletet választani")
            self.step(0, "ok")
        except Exception as e:
            self.step(0, "err")
            QMessageBox.warning(self, "Átdolgozás", f"Nem sikerült beolvasni a dalt:\n{e}")
        finally:
            QApplication.restoreOverrideCursor()
        self.vocals = ""

    # ---------------------------------------------------------- 2. ének
    def run_sep(self):
        if not self._need_src():
            return
        mroot = Path(self.main.cfg["models_dir"])
        if not any((mroot / "HTDemucs-GGUF" / n).is_file() for n in ("htdemucs-q8_0.gguf", "htdemucs-f16.gguf")):
            self.step(1, "skip")
            self.lbl_sep.setText("Kihagyva: a HTDemucs modell nincs letöltve (Telepítés fül).")
            self.say("Ének leválasztás kihagyva (nincs HTDemucs).")
            self._chain_next(1)
            return
        self.step(1, "run")
        self.lbl_sep.setText("Folyamatban… (1–3 perc)")
        self.say("Ének leválasztása…")
        self.main.request_separation(self.work_src, {"remake": True})

    def on_stems(self, stems: dict):
        self.vocals = stems.get("vocals", "")
        self.btn_play_voc.setEnabled(bool(self.vocals))
        self.lbl_sep.setText("Kész: " + ", ".join(tr(STEM_NAMES.get(k, k)) for k in stems))
        self.step(1, "ok")
        self.say("✓ Ének leválasztva.")
        self._chain_next(1)

    # ---------------------------------------------------------- 3. dallam
    def run_abc(self):
        if not self._need_src():
            return
        if not (Path(self.main.cfg["models_dir"]) / SHEET_DIRNAME / "sheetsage2-orig.gguf").is_file():
            self.step(2, "err")
            QMessageBox.warning(self, "Átdolgozás", "A SheetSage2 modell nincs letöltve (Telepítés fül).")
            self.auto_chain = False
            return
        self.step(2, "run")
        self.say("Dallam és akkordok kinyerése…")
        self.main.job_counter += 1
        self.main.worker.add({"id": self.main.job_counter, "kind": "sheetsage", "audio": self.work_src, "target": "remake",
                               "fail_kind": "remake_abc"})
        self.main._update_queue_label()

    def on_abc(self, abc: str):
        self.abc = abc
        self.ed_abc.setPlainText(abc)
        self.step(2, "ok")
        self.say("✓ Dallam kinyerve.")
        self._chain_next(2)

    def _abc_x2(self):
        r_ = abc_extend(self.ed_abc.toPlainText(), 2)
        if r_:
            self.ed_abc.setPlainText(r_)

    def _abc_to_score(self):
        if self.main.score and self.ed_abc.toPlainText().strip():
            self.main.score.set_abc(self.ed_abc.toPlainText(), "Forrás: Átdolgozás", self.ed_title.text())
            self.main.show_page("score")

    # ---------------------------------------------------------- 4. szöveg
    def asr_model(self) -> Path | None:
        mroot = Path(self.main.cfg["models_dir"])
        for rel in ("Qwen3-ASR-1.7B-GGUF/qwen3-asr-1.7b-q8_0.gguf", "Qwen3-ASR-0.6B-GGUF/qwen3-asr-0.6b-q8_0.gguf",
                    "Qwen3-ASR-1.7B-GGUF/qwen3-asr-1.7b-f16.gguf"):
            if (mroot / rel).is_file():
                return mroot / rel
        return None

    def run_asr(self):
        if not self._need_src():
            return
        if self.asr_model() is None:
            self.step(3, "skip")
            self.say("Szövegfelismerés kihagyva: a Qwen3-ASR modell nincs letöltve (Telepítés fül).")
            if not self.auto_chain:
                QMessageBox.information(self, "Átdolgozás", "A szövegfelismeréshez töltsd le a 'Qwen3-ASR' csomagot "
                                        "a Telepítés & Szerver fülön. Addig a szöveget kézzel is beírhatod.")
            self._chain_next(3)
            return
        self.step(3, "run")
        src = self.vocals or self.work_src
        self.say("Dalszöveg felismerése" + (" a leválasztott énekből…" if self.vocals else " a teljes dalból…"))
        self.main.job_counter += 1
        timed = self.chk_timed.isChecked() and bool(self.vocals)
        if self.chk_timed.isChecked() and not self.vocals:
            self.say("Időzítéshez leválasztott ének kell - most időzítés nélkül ismerem fel.")
        self.main.worker.add({"id": self.main.job_counter, "kind": "asr", "audio": src, "model": str(self.asr_model()),
                              "language": self.cb_lang.currentData(), "timed": timed})
        self.main._update_queue_label()

    def on_asr(self, text: str, lang: str, keep_text: bool = False):
        if not keep_text:
            self.timed = []
            self.btn_timing.setEnabled(False)
            self.ed_lyrics.setPlainText(auto_structure_lyrics(text))
        self.step(3, "ok")
        self.say(f"✓ Szöveg felismerve{(' (' + lang + ')') if lang else ''} - nézd át és javítsd!")
        if lang:
            low = lang.lower()
            for i in range(self.cb_lang_tag.count()):
                if self.cb_lang_tag.itemData(i) and self.cb_lang_tag.itemData(i).lower().startswith(low[:3]):
                    self.cb_lang_tag.setCurrentIndex(i)
        self._chain_next(3)

    def on_asr_timed(self, lines: list, lang: str, total: float):
        self.timed = lines
        self.btn_timing.setEnabled(True)
        self.ed_lyrics.setPlainText(structure_timed_lyrics(lines, total))
        self.say(f"⏱ {len(lines)} időzített sor (az első {fmt_mmss(lines[0]['start'])}-nál kezdődik). "
                 "Az ⏱ Időzítés… gombbal megnézheted, melyik sor hol szól az eredetiben.")
        self.on_asr("", lang, keep_text=True)

    def _bpm_and_meter(self) -> tuple[float, int]:
        abc = self.ed_abc.toPlainText()
        bpm, beats = 0.0, 4
        m_ = re.search(r"^Q:\s*(?:\d+/\d+\s*=\s*)?(\d+(?:\.\d+)?)", abc, re.M)
        if m_:
            bpm = float(m_.group(1))
        m_ = re.search(r"^M:\s*(\d+)/(\d+)", abc, re.M)
        if m_:
            beats = int(m_.group(1))
        if not bpm and self.work_src:
            try:
                bpm = estimate_bpm(read_audio(self.work_src))
            except Exception:
                bpm = 0.0
        return bpm, beats

    def show_timing(self):
        if not self.timed:
            return
        bpm, beats = self._bpm_and_meter()
        dlg = QDialog(self)
        dlg.setWindowTitle("Időzítés – melyik sor hol szól az eredetiben")
        dlg.resize(760, 560)
        v = QVBoxLayout(dlg)
        head = f"Tempó: kb. {bpm:.0f} BPM, {beats} negyed / ütem. " if bpm else ""
        info = QLabel(head + "Dupla kattintás egy sorra: az eredeti dal lejátszása attól a ponttól.")
        info.setWordWrap(True)
        info.setObjectName("dim")
        v.addWidget(info)
        tbl = QTableWidget(len(self.timed), 4)
        tbl.setHorizontalHeaderLabels(["Kezdet", "Ütem", "Hossz", "Sor"])
        tbl.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        tbl.verticalHeader().setVisible(False)
        tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        for r_, l in enumerate(self.timed):
            bar = f"{int(l['start'] * bpm / 60 / beats) + 1}." if bpm else "–"
            for c, val in enumerate((fmt_mmss(l["start"]), bar, f"{l['end'] - l['start']:.1f} s", l["text"])):
                tbl.setItem(r_, c, QTableWidgetItem(val))
        tbl.resizeColumnsToContents()
        tbl.cellDoubleClicked.connect(lambda r_, c: self._play_at(self.timed[r_]["start"]))
        v.addWidget(tbl, 1)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.button(QDialogButtonBox.Close).setText("Bezárás")
        bb.rejected.connect(dlg.reject)
        v.addWidget(bb)
        install_tip_filter(dlg)
        dlg.exec()

    def _play_at(self, sec: float):
        src = self.work_src or self.src
        if not src:
            return
        self.main.play_file(src, "Eredeti")
        if self.main.player:
            QTimer.singleShot(150, lambda: self.main.player.setPosition(int(sec * 1000)))

    def ollama_rewrite(self):
        m = self.main
        if not m._ollama_ready():
            return
        what, ok = QInputDialog.getItem(self, "Új szöveg", "Mit csináljon?",
                                        ["Fordítsa angolra, megtartva a szótagszámot",
                                         "Fordítsa magyarra, megtartva a szótagszámot",
                                         "Írjon új szöveget ugyanerre a szerkezetre (téma megadása)"], 0, False)
        if not ok:
            return
        cur = self.ed_lyrics.toPlainText().strip()
        if what.startswith("Írjon"):
            theme, ok = QInputDialog.getText(self, "Téma", "Miről szóljon az új szöveg?")
            if not ok:
                return
            prompt = ("Write NEW song lyrics about: " + theme + ". Keep EXACTLY the same section tags, the same number "
                      "of lines per section and roughly the same syllable count per line as this original:\n\n" + cur +
                      "\n\nOutput only the lyrics with the section tags.")
        else:
            lang = "English" if "angol" in what else "Hungarian"
            prompt = (f"Translate these song lyrics into {lang} so they can be SUNG to the same melody: keep the section "
                      "tags, the line count and roughly the syllable count per line. Output only the lyrics.\n\n" + cur)
        m.ollama_worker = OllamaWorker(m.cfg["ollama_url"], m.cfg["ollama_model"], prompt, "remake")
        m.ollama_worker.done.connect(lambda tgt, txt: self.ed_lyrics.setPlainText(txt.strip()))
        m.ollama_worker.failed.connect(lambda msg: QMessageBox.warning(self, "Ollama hiba", msg))
        m.ollama_worker.start()
        self.say("Ollama dolgozik a szövegen…")

    # ---------------------------------------------------------- láncolás
    def run_all(self):
        if not self._need_src():
            return
        if not self.main.server_ok:
            QMessageBox.information(self, "Átdolgozás", "A szerver nem fut - indítsd el a Telepítés & Szerver fülön.")
            return
        self.auto_chain = True
        self.say("⚡ Előkészítés egyben indul.")
        self.run_sep()

    def _chain_next(self, done_step):
        if not self.auto_chain:
            return
        if done_step == 1:
            self.run_abc()
        elif done_step == 2:
            self.run_asr()
        elif done_step == 3:
            self.auto_chain = False
            self.step(4, "run")
            self.say("Előkészítés kész. Nézd át a dallamot és a szöveget, válassz stílust (5.), majd indítsd (6.).")
            QApplication.beep()

    def on_failed(self, kind, msg):
        idx = {"remake_sep": 1, "remake_abc": 2, "asr": 3}.get(kind)
        if idx is not None:
            self.step(idx, "err")
        self.say(f"HIBA ({kind}): {msg}")
        if self.auto_chain and idx is not None and idx in (1, 3):
            self._chain_next(idx)      # ének/szöveg hibája nem állítja meg a láncot
        else:
            self.auto_chain = False

    # ---------------------------------------------------------- 6. generálás
    def styles(self) -> list[tuple[str, str]]:
        voice = self.cb_voice.currentData()
        lang = self.cb_lang_tag.currentData()
        out = []
        chosen = [(n, REMAKE_STYLES[n]) for n, cb in self.style_checks.items() if cb.isChecked()]
        if self.ed_style.text().strip():
            chosen.append(("Saját", self.ed_style.text().strip()))
        for name, tags in chosen:
            parts = ([lang] if lang and not self.chk_instr.isChecked() else []) + [tags]
            if voice and not self.chk_instr.isChecked():
                parts.append(voice)
            out.append((name, ", ".join(parts)))
        return out

    def start(self):
        if not self._need_src():
            return
        sts = self.styles()
        if not sts:
            QMessageBox.information(self, "Átdolgozás", "Válassz legalább egy stílust (5. lépés).")
            return
        lyrics = self.ed_lyrics.toPlainText().strip()
        if self.chk_instr.isChecked():
            lyrics = "[instrumental]"
        if not lyrics:
            QMessageBox.information(self, "Átdolgozás", "Kell dalszöveg (4. lépés) - vagy jelöld be: Instrumentális.")
            return
        abc = self.ed_abc.toPlainText().strip()
        if not abc and QMessageBox.question(self, "Átdolgozás", "Nincs kinyert dallam (3. lépés), így a YuE2 saját "
                                            "dallamot ír. Így is indítsuk?") != QMessageBox.Yes:
            return
        title = self.ed_title.text().strip() or Path(self.src).stem
        n_total = 0
        for name, style in sts:
            jids = self.main.enqueue_custom(lyrics=lyrics, style=style, abc=abc, cot=self.cb_cot.currentData(),
                                            count=self.sp_n.value(), title=f"{title} – {name}",
                                            instrumental=self.chk_instr.isChecked(),
                                            length_sec=int(self.src_len) + 5 if self.chk_same_len.isChecked() and self.src_len else None)
            self.jobs.update(jids)
            n_total += len(jids)
        self.step(4, "ok")
        self.step(5, "run")
        self.say(f"▶ {n_total} változat a feladatsorban ({len(sts)} stílus × {self.sp_n.value()}).")

    def on_result(self, jid, info):
        if jid not in self.jobs:
            return
        self.jobs.discard(jid)
        t = self.main.db.get(info["track_id"])
        it = QListWidgetItem(f"♪ {t.get('title')}   ({info['dur']:.0f} mp)")
        it.setData(Qt.UserRole, info["track_id"])
        self.lst.addItem(it)
        if not self.jobs:
            self.step(5, "ok")
            self.say("✓ Minden változat elkészült. Dupla kattintás: lejátszás.")

    def _play_item(self, it):
        t = self.main.db.get(int(it.data(Qt.UserRole)))
        if t:
            self.main._play_track(t)

    # ---------------------------------------------------------- projekt
    def has_content(self) -> bool:
        return bool(self.src or self.ed_abc.toPlainText().strip() or self.ed_lyrics.toPlainText().strip()
                    or self.lst.count())

    def clear(self, ask: bool = False):
        if ask and self.has_content() and QMessageBox.question(
                self, "Tiszta lap", "Kiürítsem az Átdolgozás fület? (A kész dalok a Könyvtárban maradnak.)") != QMessageBox.Yes:
            return
        self.set_state({})

    def get_state(self) -> dict:
        return {"src": self.src, "part": self.chk_part.isChecked(), "from": self.sp_from.value(),
                "to": self.sp_to.value(), "vocals": self.vocals, "abc": self.ed_abc.toPlainText(),
                "lang": self.cb_lang.currentIndex(), "lyrics": self.ed_lyrics.toPlainText(),
                "instr": self.chk_instr.isChecked(),
                "styles": [n for n, cb in self.style_checks.items() if cb.isChecked()],
                "voice": self.cb_voice.currentIndex(), "lang_tag": self.cb_lang_tag.currentIndex(),
                "custom": self.ed_style.text(), "n": self.sp_n.value(), "cot": self.cb_cot.currentIndex(),
                "title": self.ed_title.text(), "same_len": self.chk_same_len.isChecked(),
                "timed": self.timed, "timed_on": self.chk_timed.isChecked(),
                "results": [int(self.lst.item(i).data(Qt.UserRole)) for i in range(self.lst.count())]}

    def set_state(self, st: dict):
        self.auto_chain = False
        self.jobs.clear()
        self.src = st.get("src", "")
        self.work_src = ""
        self.src_len = 0.0
        self.ed_file.setText(self.src)
        for w_ in (self.chk_part, self.sp_from, self.sp_to):
            w_.blockSignals(True)
        self.chk_part.setChecked(bool(st.get("part")))
        self.sp_from.setValue(float(st.get("from", 0)))
        self.sp_to.setValue(float(st.get("to", 90)))
        self.sp_from.setEnabled(self.chk_part.isChecked())
        self.sp_to.setEnabled(self.chk_part.isChecked())
        for w_ in (self.chk_part, self.sp_from, self.sp_to):
            w_.blockSignals(False)
        self.lbl_info.setText("")
        for i in range(6):
            self.step(i, "todo")
        if self.src and Path(self.src).is_file():
            self._prepare_source()
        self.vocals = st.get("vocals", "") if Path(st.get("vocals") or "").is_file() else ""
        self.btn_play_voc.setEnabled(bool(self.vocals))
        self.lbl_sep.setText("Kész (betöltve)" if self.vocals else "")
        if self.vocals:
            self.step(1, "ok")
        self.ed_abc.setPlainText(st.get("abc", ""))
        if st.get("abc"):
            self.step(2, "ok")
        self.cb_lang.setCurrentIndex(int(st.get("lang", 0)))
        self.ed_lyrics.setPlainText(st.get("lyrics", ""))
        if st.get("lyrics"):
            self.step(3, "ok")
        self.chk_instr.setChecked(bool(st.get("instr")))
        for n, cb in self.style_checks.items():
            cb.setChecked(n in (st.get("styles") or []))
        self.cb_voice.setCurrentIndex(int(st.get("voice", 0)))
        self.cb_lang_tag.setCurrentIndex(int(st.get("lang_tag", 0)))
        self.ed_style.setText(st.get("custom", ""))
        self.sp_n.setValue(int(st.get("n", 2)))
        self.cb_cot.setCurrentIndex(int(st.get("cot", 0)))
        self.ed_title.setText(st.get("title", ""))
        self.chk_same_len.setChecked(bool(st.get("same_len", True)))
        self.timed = list(st.get("timed") or [])
        self.btn_timing.setEnabled(bool(self.timed))
        self.chk_timed.setChecked(bool(st.get("timed_on", True)))
        self.lst.clear()
        for tid in st.get("results") or []:
            t = self.main.db.get(int(tid))
            if t:
                it = QListWidgetItem(f"♪ {t.get('title')}   ({(t.get('duration') or 0):.0f} mp)")
                it.setData(Qt.UserRole, int(tid))
                self.lst.addItem(it)
        if self.lst.count():
            self.step(4, "ok")
            self.step(5, "ok")
        self.log.clear()


class TourOverlay(QWidget):
    def __init__(self, main, steps: list[dict], title: str):
        super().__init__(main)
        self.main = main
        self.steps = steps
        self.title = title
        self.i = 0
        self.hole = QRect()
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setFocusPolicy(Qt.StrongFocus)
        self.bubble = QFrame(self)
        self.bubble.setObjectName("tourbubble")
        bl = QVBoxLayout(self.bubble)
        bl.setContentsMargins(16, 12, 16, 12)
        self.lbl_count = QLabel()
        self.lbl_count.setObjectName("tourcount")
        self.lbl_title = QLabel()
        self.lbl_title.setObjectName("tourtitle")
        self.lbl_title.setWordWrap(True)
        self.lbl_text = QLabel()
        self.lbl_text.setObjectName("tourtext")
        self.lbl_text.setWordWrap(True)
        self.lbl_text.setTextFormat(Qt.RichText)
        bl.addWidget(self.lbl_count)
        bl.addWidget(self.lbl_title)
        bl.addWidget(self.lbl_text)
        r = QHBoxLayout()
        self.btn_quit = QPushButton("Kilépés")
        self.btn_prev = QPushButton("◀ Előző")
        self.btn_next = QPushButton("Következő ▶")
        self.btn_next.setObjectName("primary")
        self.btn_quit.clicked.connect(self.finish)
        self.btn_prev.clicked.connect(lambda: self.go(self.i - 1))
        self.btn_next.clicked.connect(lambda: self.go(self.i + 1))
        r.addWidget(self.btn_quit)
        r.addStretch()
        r.addWidget(self.btn_prev)
        r.addWidget(self.btn_next)
        bl.addLayout(r)
        self.bubble.setFixedWidth(430)
        main.installEventFilter(self)
        self.setGeometry(main.rect())
        self.show()
        self.raise_()
        self.setFocus()
        self.go(0)

    def eventFilter(self, obj, ev):
        if obj is self.main and ev.type() == QEvent.Resize:
            self.setGeometry(self.main.rect())
            QTimer.singleShot(0, self._place)
        return False

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Escape:
            self.finish()
        elif ev.key() in (Qt.Key_Right, Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self.go(self.i + 1)
        elif ev.key() == Qt.Key_Left:
            self.go(self.i - 1)

    def finish(self):
        self.main.removeEventFilter(self)
        self.hide()
        self.deleteLater()
        self.main._tour = None

    def go(self, i):
        if i >= len(self.steps):
            self.finish()
            return
        self.i = max(0, i)
        st = self.steps[self.i]
        if st.get("before"):
            try:
                st["before"]()
            except Exception:
                pass
        self.lbl_count.setText(f"{self.title}  ·  {self.i + 1} / {len(self.steps)}")
        self.lbl_title.setText(st["title"])
        self.lbl_text.setText(st["text"])
        self.btn_prev.setEnabled(self.i > 0)
        self.btn_next.setText("Kész ✓" if self.i == len(self.steps) - 1 else "Következő ▶")
        tw = self._target()
        if isinstance(tw, (list, tuple)):
            tw = next((x for x in tw if x is not None), None)
        if isinstance(tw, QWidget):        # görgethető oldalon görgessünk oda
            p_ = tw.parentWidget()
            while p_ is not None:
                if isinstance(p_, QScrollArea):
                    p_.ensureWidgetVisible(tw, 20, 40)
                    break
                p_ = p_.parentWidget()
        QTimer.singleShot(80, self._place)

    def _target(self):
        st = self.steps[self.i]
        try:
            return st["target"]() if st.get("target") else None
        except Exception:
            return None

    def _place(self):
        if not self.isVisible():
            return
        w = self._target()
        if isinstance(w, (list, tuple)):
            ws = [x for x in w if x is not None and x.isVisible()]
            if ws:
                rect = QRect()
                for x in ws:
                    rect = rect.united(QRect(x.mapTo(self.main, QPoint(0, 0)), x.size()))
                self.hole = rect.adjusted(-6, -6, 6, 6).intersected(self.rect())
                w = None
                self._finish_place()
                return
            w = None
        if w is not None and w.isVisible():
            p = w.parentWidget()
            while p is not None:
                if isinstance(p, QScrollArea):
                    p.ensureWidgetVisible(w, 30, 30)
                    break
                p = p.parentWidget()
            tl = w.mapTo(self.main, QPoint(0, 0))
            r = QRect(tl, w.size()).adjusted(-6, -6, 6, 6).intersected(self.rect())
            self.hole = r
        else:
            self.hole = QRect()
        self._finish_place()

    def _finish_place(self):
        full = QRegion(self.rect())
        self.setMask(full.subtracted(QRegion(self.hole)) if not self.hole.isNull() else full)
        self.bubble.adjustSize()
        bw, bh = self.bubble.width(), self.bubble.sizeHint().height()
        W, H = self.width(), self.height()
        if self.hole.isNull():
            x, y = (W - bw) // 2, (H - bh) // 2
        else:
            h = self.hole
            if h.bottom() + bh + 16 < H:
                x, y = h.left(), h.bottom() + 14
            elif h.top() - bh - 16 > 0:
                x, y = h.left(), h.top() - bh - 14
            elif h.right() + bw + 16 < W:
                x, y = h.right() + 14, h.top()
            elif h.left() - bw - 16 > 0:
                x, y = h.left() - bw - 14, h.top()
            else:
                x, y = (W - bw) // 2, H - bh - 20
            x = max(10, min(x, W - bw - 10))
            y = max(10, min(y, H - bh - 10))
        self.bubble.setGeometry(x, y, bw, bh)
        self.bubble.raise_()
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addRect(QRectF(self.rect()))
        if not self.hole.isNull():
            hp = QPainterPath()
            hp.addRoundedRect(QRectF(self.hole), 8, 8)
            path = path.subtracted(hp)
        p.fillPath(path, QColor(5, 5, 10, 175))
        if not self.hole.isNull():
            p.setPen(QPen(QColor("#a78bfa"), 3))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(QRectF(self.hole).adjusted(1, 1, -1, -1), 8, 8)
        p.end()

    def mousePressEvent(self, ev):
        ev.accept()


def _find_button(widget, text):
    if widget is None:
        return None
    for b in widget.findChildren(QPushButton):
        if b.text() == text:
            return b
    return None


def _find_group(widget, title_start):
    if widget is None:
        return None
    for g in widget.findChildren(QGroupBox):
        if g.title().startswith(title_start):
            return g
    return None


def build_tours(m) -> dict:
    """A kalauz túrái. A lépések célpontjai függvények, mert a felület közben változhat."""
    T = m.tabs
    gen, abc, lib = m.pages["gen"], m.pages["abc"], m.pages["lib"]
    remake = getattr(m, "remake", None)
    daw = getattr(m, "daw", None)
    score = getattr(m, "score", None)

    def tab(w):
        return lambda: (w is not None) and T.setCurrentWidget(w)

    tours = {}
    tours["first"] = ("Az első dal", [
        {"title": "Üdv a YuE2 Studióban! 👋",
         "text": "Ez a rövid kalauz végigvezet az <b>első dalodon</b>. Mindig csak a <b>kiemelt</b> részre figyelj - "
                 "arra kattintani is lehet, ha ki akarod próbálni.<br><br>Lépés: <b>Következő</b> gomb vagy a "
                 "<b>→</b> nyíl. Kilépés: <b>Esc</b>.",
         "before": tab(gen)},
        {"title": "A motor állapota", "target": lambda: m.sb_server,
         "text": "Itt látod, fut-e a zenegeneráló motor. <b>'✓ fut'</b> = minden rendben.<br>Ha nem fut: "
                 "<b>Telepítés &amp; Szerver</b> fül → <b>▶ Indítás</b>."},
        {"title": "1. lépés: a dal szövege", "target": lambda: m.ed_lyrics, "before": tab(gen),
         "text": "Ide írod a dalszöveget. A dal részeit szögletes zárójeles címkék jelölik, külön sorban: "
                 "<b>[Verse]</b> = versszak, <b>[Chorus]</b> = refrén.<br><br>Tipp: rövid sorok, és a refrén "
                 "mindig szó szerint ugyanaz legyen. A <b>Példa</b> gomb betölt egy kész mintát."},
        {"title": "Szakasz-gombok", "target": lambda: [_find_button(gen, t) for t in ("[Intro]", "[instrumental]")],
         "text": "Ezekkel a gombokkal a címkéket egy kattintással beszúrhatod oda, ahol a kurzor áll."},
        {"title": "2. lépés: a stílus", "target": lambda: m.ed_style,
         "text": "Itt írod le <b>angol szavakkal</b>, vesszővel elválasztva, milyen legyen a zene. "
                 "5–8 szó az ideális, például:<br><i>English, blues, melancholic, piano, harmonica, male vocal</i>"},
        {"title": "Stílus kattintással", "target": lambda: m.tag_combos["Műfaj"],
         "text": "Nem kell fejből tudni a szavakat: ezekből a listákból (műfaj, hangulat, hangszer, ének…) "
                 "kattintással is hozzáadhatsz. A <b>🎲 Véletlen stílus</b> gomb ötletet is ad."},
        {"title": "3. lépés: zenei terv", "target": lambda: m.cb_cot,
         "text": "Maradjon <b>full</b>: így a gép előbb megírja a dal kottáját (dallam + akkordok), és ahhoz igazodik. "
                 "Ez adja a legjobb dalszerkezetet."},
        {"title": "4. lépés: hány változat?", "target": lambda: m.sp_batch,
         "text": "Érdemes <b>2–3</b> változatot kérni egyszerre - mindegyik más lesz, és kiválasztod a legjobbat."},
        {"title": "5. lépés: indítás!", "target": lambda: m.btn_gen,
         "text": "Erre kattintva indul a generálás. Az <b>első</b> alkalommal 1–2 percig tölt a modell, utána "
                 "dalonként pár perc. Közben nyugodtan dolgozhatsz tovább."},
        {"title": "A feladatsor", "target": lambda: m.tbl_queue,
         "text": "Itt követheted a munkát: <i>várakozik → ▶ fut → ✓ kész</i>. Egy <b>kész</b> sorra "
                 "<b>duplán kattintva</b> rögtön meghallgatod."},
        {"title": "A lejátszó", "target": lambda: m.btn_play.parentWidget(),
         "text": "Ez az ablak alján mindig látszik: <b>⏮ ⏸ ■ ⏭</b>, tekerés, hangerő. A "
                 "<b>'következő automatikusan'</b> pipával végighallgathatod az összes változatot."},
        {"title": "A Könyvtár", "target": lambda: m.tbl_lib, "before": tab(lib),
         "text": "Minden elkészült dalod itt van. Duplakattintás = lejátszás. A legjobbat jelöld <b>★ Kedvenc</b>-nek; "
                 "a <b>⟲ Beállítások vissza</b> gombbal pedig ugyanabból tovább finomíthatsz."},
        {"title": "Kész! 🎉",
         "text": "Ennyi az alap. Ha kíváncsi vagy a többire, a <b>Súgó</b> fülön (vagy lent a <b>🧭 Kalauz</b> "
                 "gombbal) indíthatod a többi bemutatót: <b>DAW – sávok</b>, <b>Kotta</b>, <b>Dal átdolgozása</b>.<br><br>"
                 "És ne feledd: minden elem fölött, ha megállsz az egérrel, megjelenik egy magyar magyarázat."},
    ])
    if daw is not None:
        tours["daw"] = ("DAW – sávok szerkesztése", [
            {"title": "A DAW fül", "before": tab(daw),
             "text": "A <b>DAW</b> (digitális hangstúdió) egy dalt sávokra szedve mutat: minden sor egy sáv "
                     "(pl. ének, dob), a színes téglalapok a <b>klipek</b> - ezeket lehet mozgatni, vágni, effektezni."},
            {"title": "1. Egy dal betöltése", "target": lambda: daw.btn["lib"],
             "text": "Kezdésnek: jelölj ki egy dalt a Könyvtárban, majd ezzel a gombbal hozd ide. "
                     "Saját hangfájlt a <b>＋ Hangfájl…</b> gombbal tölthetsz be."},
            {"title": "2. Sávokra bontás", "target": lambda: daw.btn["sep"],
             "text": "A varázslat: egy kész dalból <b>4 külön sávot</b> készít - ének, dob, basszus, egyéb hangszerek. "
                     "Pár perc. (Kell hozzá a <b>HTDemucs</b> csomag a Telepítés fülön.)<br>Egy klipen jobb klikkel "
                     "is elérhető."},
            {"title": "3. A sávfejlécek", "target": lambda: daw.hdr_host if daw.project["tracks"] else daw.btn["track"],
             "text": "Minden sáv bal oldalán:<br><b>M</b> = némítás, <b>S</b> = csak ez szóljon, "
                     "<b>A</b> = hangerő-görbe rajzolása, <b>FX</b> = effektek (zengés, EQ…), alatta "
                     "<b>hangerő</b> és <b>panoráma</b> (bal-jobb)."},
            {"title": "4. Az idővonal", "target": lambda: daw.canvas if daw.project["tracks"] else daw.empty_hint,
             "text": "• <b>Húzás</b>: klip mozgatása (másik sávra is)<br>• <b>Szélét húzva</b>: levágás<br>"
                     "• <b>Fehér pöttyök</b> a tetején: halkítva indul/végződik<br>• <b>S</b> billentyű: kettévágás a "
                     "sárga vonalnál<br>• <b>Dupla kattintás</b>: tempó, hangmagasság<br>• <b>Jobb klikk</b>: minden más"},
            {"title": "5. A vonalzó", "target": lambda: daw.ruler,
             "text": "Kattintás: oda ugrik a lejátszás. <b>Húzás</b>: kijelölsz egy részt, amit a 🔁 gombbal "
                     "ismételtethetsz - gyakorláshoz, finomhangoláshoz kiváló."},
            {"title": "6. Lejátszás", "target": lambda: daw.btn_play,
             "text": "Indítás és megállítás - vagy egyszerűen a <b>Szóköz</b> billentyű, ha az idővonalra kattintottál."},
            {"title": "7. Mester effektek", "target": lambda: daw.btn_mfx,
             "text": "Az egész dalra ható effektek. Jó tipp: tölts be egy kész láncot (<i>'Mester - fényes, hangos'</i>) - "
                     "a végén lévő <b>Limiter</b> megvédi a hangot a torzulástól."},
            {"title": "8. Exportálás", "target": lambda: daw.btn["export"],
             "text": "Ha kész vagy, ezzel mented el az egészet WAV vagy MP3 fájlba - és felveheted a Könyvtárba is."},
            {"title": "Hibáztál? Semmi gond!", "target": lambda: daw.btn["undo"],
             "text": "A <b>↶</b> gomb (vagy <b>Ctrl+Z</b>) visszavon bármit. A projekt magától is mentődik."},
            {"title": "Kész! 🎛",
             "text": "Próbáld ki: bonts sávokra egy dalt, némítsd el az éneket (<b>M</b>) - és máris kész a karaoke-változat!"},
        ])
    if score is not None:
        tours["score"] = ("Kotta", [
            {"title": "A Kotta fül", "before": tab(score),
             "text": "Itt a zenéből <b>igazi kotta</b> lesz: ötvonalas rendszer, hangjegyek, akkordok - "
                     "kinyomtathatod vagy PDF-be mentheted."},
            {"title": "1. Honnan jöjjön a kotta?", "target": lambda: _find_group(score, "1. Honnan"),
             "text": "• <b>ABC-kotta fül / Könyvtár</b>: a YuE2 által tervezett dallam és akkordok<br>"
                     "• <b>MIDI</b>: a DAW-ban készített hangjegyek (pl. az Ének sávból: jobb klikk → ♫ MIDI)"},
            {"title": "2. Megjelenés", "target": lambda: _find_group(score, "2. Megjelenés"),
             "text": "Cím, szerző, melyik szólam látsszon, és <b>transzponálás</b>: ha túl magas a dal, tedd lejjebb pár "
                     "félhanggal."},
            {"title": "3. Nyomtatás, PDF, lejátszás", "target": lambda: _find_group(score, "3. Mit"),
             "text": "<b>🖨 Nyomtatás</b>, <b>PDF mentése</b>, és <b>▶ Lejátszás</b> - ilyenkor a kottán lilával "
                     "mutatja, hol tart."},
            {"title": "A kotta", "target": lambda: score.view,
             "text": "Itt látod a kész kottát. A <b>Kód szerkesztése</b> gombbal szövegesen is javíthatod."},
        ])
    if remake is not None:
        R = remake
        tours["remake"] = ("Meglévő dal átdolgozása", [
            {"title": "Az Átdolgozás fül", "before": tab(R),
             "text": "Itt egy meglévő dalt (MP3, WAV…) készíthetsz újra <b>új stílusban</b>: a YuE2 megtartja a "
                     "dallamot és az akkordokat, de új hangszereléssel, új énekessel játssza el. Fentről lefelé "
                     "kell végigmenni a lépéseken."},
            {"title": "1. A dal", "target": lambda: _find_group(R, "1."),
             "text": "<b>Tallózás…</b> - válaszd ki a dalt. Hosszú dalnál pipáld be a <b>Csak egy részlet</b>-et "
                     "(pl. 30-tól 120 mp-ig): gyorsabb és pontosabb lesz."},
            {"title": "⚡ Egy kattintással", "target": lambda: R.btn_auto,
             "text": "Ez a gomb egymás után elvégzi a 2., 3. és 4. lépést: leválasztja az éneket, kinyeri a "
                     "dallamot, és felismeri a szöveget. Pár perc."},
            {"title": "2. Ének leválasztása", "target": lambda: _find_group(R, "2."),
             "text": "A HTDemucs kiszedi az éneket a zenéből - ebből sokkal pontosabb a szövegfelismerés."},
            {"title": "3. Dallam és akkordok", "target": lambda: _find_group(R, "3."),
             "text": "A SheetSage2 kiolvassa a dallamot és az akkordokat zenei tervvé. Ezt követi majd a YuE2. "
                     "<b>Megnézem kottaként</b>: igazi kottán is látod."},
            {"title": "4. Dalszöveg", "target": lambda: _find_group(R, "4."),
             "text": "A Qwen3-ASR leírja, mit énekelnek. <b>Mindig nézd át</b> és javítsd! Írhatsz teljesen új "
                     "szöveget is, vagy az Ollamával lefordíttathatod."},
            {"title": "5. Új stílus", "target": lambda: _find_group(R, "5."),
             "text": "Pipálj be egy vagy <b>több</b> stílust - mindegyikből külön változat készül (pl. Jazz + Blues + "
                     "Rock egyszerre). Válassz énekhangot és az ének nyelvét."},
            {"title": "6. Indítás", "target": lambda: R.btn_go,
             "text": "Stílusonként 2–3 változatot érdemes kérni. Az eredmények a jobb oldali listában és a "
                     "Könyvtárban jelennek meg. Kész!"},
            {"title": "Hol tartunk?", "target": lambda: R.step_lbls,
             "text": "Itt követed a lépéseket: ○ még nincs kész · ⏳ folyamatban · ✓ kész · – kihagyva."},
        ])
    tours["project"] = ("Projektek és tiszta lap", [
        {"title": "A projekt sáv", "target": lambda: m.lbl_proj.parentWidget(),
         "text": "Fent mindig látod, melyik <b>projekten</b> dolgozol. Egy projekt = minden fül tartalma (Generálás, "
                 "Átdolgozás, ABC-kotta, DAW, Kotta) + a hozzá generált dalok. A sárga <b>●</b> jel: van még nem "
                 "mentett változás."},
        {"title": "Új, Megnyitás, Mentés", "target": lambda: m.lbl_proj.parentWidget(),
         "text": "<b>🆕 Új projekt</b> (Ctrl+N): minden fül kiürül, új nevet adsz. <b>💾 Mentés</b> (Ctrl+S): "
                 "minden egy fájlba kerül. <b>📂 Megnyitás</b> (Ctrl+O) vagy <b>Legutóbbiak</b>: pontosan "
                 "visszaáll, ahogy hagytad."},
        {"title": "🧹 Tiszta lap", "target": lambda: m.btn_clear_gen, "before": tab(gen),
         "text": "Minden munkafülön van egy ilyen gomb: csak <b>azt az egy fület</b> üríti ki. A kész dalok a "
                 "Könyvtárban maradnak."},
        {"title": "A Könyvtár szűrője", "target": lambda: m.cb_lib_proj, "before": tab(lib),
         "text": "A Könyvtár alapból <b>csak az aktuális projekt</b> dalait mutatja - így nem keverednek. A régi "
                 "projektek dalait itt választhatod ki, vagy a <b>Minden dal</b> pontot."},
        {"title": "Könyvtár ürítése", "target": lambda: m.btn_clear_lib,
         "text": "Ha tényleg törölni szeretnél: választhatsz, hogy melyik dalokat (projekt / látható / mind), és "
                 "hogy csak a listából vagy a lemezről is. A kedvencek alapból megmaradnak."},
    ])
    return tours


TOUR_QSS = """
QFrame#tourbubble { background: #22232d; border: 2px solid #a78bfa; border-radius: 12px; }
QLabel#tourcount { color: #a78bfa; font-size: 9pt; }
QLabel#tourtitle { color: #ffffff; font-size: 14pt; font-weight: 700; }
QLabel#tourtext { color: #e6e6ea; font-size: 11pt; }
"""


# ---------------------------------------------------------------------------
# Fő ablak
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cfg = load_settings()
        self.db = TrackDB(DB_FILE)
        PRESET_DIR.mkdir(exist_ok=True)
        PROJECT_DIR.mkdir(exist_ok=True)
        self.project_name = self.cfg.get("project_name") or DEFAULT_PROJECT
        self.project_file: Path | None = Path(self.cfg["project_file"]) if self.cfg.get("project_file") else None
        if self.project_file is not None and not self.project_file.is_file():
            self.project_file = None
        self._proj_clean = ""
        self.setWindowTitle(f"{APP_NAME} {APP_VERSION} - helyi YuE2 zenegenerátor")
        self.resize(1500, 950)

        self.server = QProcess(self)
        self.server.setProcessChannelMode(QProcess.MergedChannels)
        self.server.readyReadStandardOutput.connect(self._server_output)
        self.server.finished.connect(self._server_finished)
        self.server_external = False
        self.server_ok = False
        self.downloader: Downloader | None = None
        self.ollama_worker: OllamaWorker | None = None
        self.job_counter = 0
        self.job_rows: dict[int, int] = {}
        self.job_t0: dict[int, float] = {}
        self.last_abc = ""
        self.current_track: dict | None = None
        self.lib_ids: list[int] = []
        self.now_id: int | None = None
        self.job_track: dict[int, int] = {}

        self.worker = GenWorker(lambda: self.cfg, self.db)
        self.worker.log.connect(self.log)
        self.worker.job_started.connect(self._job_started)
        self.worker.job_finished.connect(self._job_finished)
        self.worker.job_failed.connect(self._job_failed)
        self.worker.abc_extracted.connect(self._abc_extracted)
        self.worker.abc_failed.connect(lambda m: (self.btn_extract.setEnabled(True),
                                                  QMessageBox.warning(self, "SheetSage2", m)))
        self.worker.sep_done.connect(self._sep_done)
        self.worker.midi_done.connect(self._midi_done)
        self.worker.task_failed.connect(self._task_failed)
        self.worker.asr_done.connect(lambda t, l: self.remake and self.remake.on_asr(t, l))
        self.worker.asr_timed.connect(lambda ls, l, tot: self.remake and self.remake.on_asr_timed(ls, l, tot))
        self.worker.remake_abc.connect(lambda a: self.remake and self.remake.on_abc(a))
        self.worker.start()

        tabs = QTabWidget()
        self.tabs = tabs
        P = self.pages = {}

        def add(name, w, label):
            P[name] = w
            tabs.addTab(w, label)

        add("gen", self._build_generate_tab(), "🎵  Generálás")
        if HAS_NUMPY and HAS_PEDALBOARD:
            self.remake = RemakePanel(self)
            add("remake", self.remake, "🔁  Átdolgozás")
        else:
            self.remake = None
            add("remake", daw_missing_widget(), "🔁  Átdolgozás")
        add("abc", self._build_abc_tab(), "🎼  ABC-kotta")
        add("lib", self._build_library_tab(), "📚  Könyvtár")
        if HAS_NUMPY and HAS_PEDALBOARD:
            self.daw = DawPanel(self)
            add("daw", self.daw, "🎛  DAW – sávszerkesztő")
        else:
            self.daw = None
            add("daw", daw_missing_widget(), "🎛  DAW – sávszerkesztő")
        if HAS_WEBENGINE:
            self.score = ScorePanel(self)
            add("score", self.score, "🎶  Kotta")
        else:
            self.score = None
            add("score", score_missing_widget(), "🎶  Kotta")
        add("setup", self._build_setup_tab(), "⚙  Telepítés && Szerver")
        add("settings", self._build_settings_tab(), "🔧  Beállítások")
        add("help", self._build_help_tab(), "❓  Súgó")
        central = QWidget()
        cl = QVBoxLayout(central)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(4)
        cl.addWidget(self._build_project_bar())
        cl.addWidget(tabs, 1)
        cl.addWidget(self._build_player_bar())
        self.setCentralWidget(central)

        self.sb_server = QLabel("Szerver: ?")
        self.sb_gpu = QLabel("")
        self.sb_queue = QLabel("Sor: 0")
        self.statusBar().addWidget(self.sb_server)
        self.statusBar().addWidget(QLabel("   "))
        self.statusBar().addWidget(self.sb_queue)
        self.btn_tour = QPushButton("🧭 Kalauz")
        self.btn_tour.setToolTip(tip_html("Kalauz|Lépésről lépésre bemutatók: első dal, DAW, kotta, dal átdolgozása."))
        self.btn_tour.clicked.connect(self.tour_menu)
        self.statusBar().addPermanentWidget(self.btn_tour)
        self.statusBar().addPermanentWidget(self.sb_gpu)

        self.poller = StatusPoller(int(self.cfg["port"]))
        self.poller.health.connect(self._on_health)
        self.poller.gpu.connect(self.sb_gpu.setText)
        self.poller.start()
        self.elapsed_timer = QTimer(self)
        self.elapsed_timer.timeout.connect(self._tick_elapsed)
        self.elapsed_timer.start(1000)

        self.refresh_install_state()
        self._refresh_presets()
        if self.cfg.get("last_form"):
            try:
                self.set_form(self.cfg["last_form"])
            except Exception:
                pass
        else:
            self.ed_lyrics.setPlainText(LYRICS_EXAMPLE)
            self.ed_style.setText(STYLE_EXAMPLE)
        self._restore_session()
        self._apply_tooltips()
        self.proj_timer = QTimer(self)
        self.proj_timer.timeout.connect(self._update_project_label)
        self.proj_timer.start(3000)
        for seq, fn in (("Ctrl+S", self.save_project), ("Ctrl+Shift+S", lambda: self.save_project(True)),
                        ("Ctrl+O", self.open_project), ("Ctrl+N", self.new_project)):
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.ApplicationShortcut)
            sc.activated.connect(fn)
        QTimer.singleShot(1800, self._maybe_autostart)
        QTimer.singleShot(3500, self._offer_tour)

    # ------------------------------------------------------------------ log
    def log(self, msg: str):
        ts = dt.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        for w in (getattr(self, "gen_log", None), getattr(self, "srv_log", None)):
            if w is not None:
                w.appendPlainText(line)

    # =========================================================== GENERÁLÁS
    def _build_generate_tab(self) -> QWidget:
        root = QSplitter(Qt.Horizontal)

        # ---------- bal oldal: űrlap (görgethető)
        form_w = QWidget()
        fl = QVBoxLayout(form_w)

        r0 = QHBoxLayout()
        r0.addStretch()
        self.btn_clear_gen = clear_button(lambda: self.clear_generate(ask=True),
                                          "A Generálás fül kiürítése: dalszöveg, stílus, cím, haladó paraméterek "
                                          "(alapértékre), LoRA-k, a kész feladatok sorai, az utolsó eredmény és a "
                                          "napló. A választott fő modell és VAE megmarad, a kész dalok a Könyvtárban "
                                          "maradnak, a várakozó feladatok nem törlődnek.")
        r0.addWidget(self.btn_clear_gen)
        fl.addLayout(r0)

        # Dalszöveg
        g_ly = QGroupBox("Dalszöveg")
        lyl = QVBoxLayout(g_ly)
        tagrow = QHBoxLayout()
        for t in ["[Intro]", "[Verse]", "[Pre-Chorus]", "[Chorus]", "[Bridge]", "[Solo]", "[Outro]", "[instrumental]"]:
            b = QPushButton(t)
            b.setObjectName("tag")
            b.clicked.connect(lambda _=False, t=t: self._insert_section(t))
            tagrow.addWidget(b)
        tagrow.addStretch()
        lyl.addLayout(tagrow)
        self.ed_lyrics = QPlainTextEdit()
        self.ed_lyrics.setMinimumHeight(220)
        self.ed_lyrics.setPlaceholderText("[Verse]\nElső sor\nMásodik sor\n[Chorus]\nRefrén…")
        self.ed_lyrics.textChanged.connect(self._lyrics_stats)
        lyl.addWidget(self.ed_lyrics)
        r = QHBoxLayout()
        self.lbl_lyrics = QLabel("")
        self.lbl_lyrics.setObjectName("dim")
        r.addWidget(self.lbl_lyrics, 1)
        for txt, fn in [("Példa", lambda: self.ed_lyrics.setPlainText(LYRICS_EXAMPLE)),
                        ("Fájlból…", self._lyrics_from_file), ("Mentés…", self._lyrics_to_file),
                        ("Törlés", self.ed_lyrics.clear), ("🤖 Ollama: dalszöveg", self._ollama_lyrics)]:
            b = QPushButton(txt)
            b.clicked.connect(fn)
            r.addWidget(b)
        lyl.addLayout(r)
        fl.addWidget(g_ly)

        # Stílus
        g_st = QGroupBox("Stílus (vesszővel elválasztott angol tagek, ideális 5–8)")
        stl = QVBoxLayout(g_st)
        r = QHBoxLayout()
        self.ed_style = QLineEdit()
        self.ed_style.setPlaceholderText(STYLE_EXAMPLE)
        self.ed_style.textChanged.connect(self._style_stats)
        r.addWidget(self.ed_style, 1)
        self.lbl_style = QLabel("0 tag")
        self.lbl_style.setObjectName("dim")
        r.addWidget(self.lbl_style)
        stl.addLayout(r)
        grid = QGridLayout()
        self.tag_combos = {}
        for i, (cat, tags) in enumerate(TAG_BANK.items()):
            cb = QComboBox()
            cb.addItem(f"{cat}…")
            cb.addItems(tags)
            cb.setEditable(False)
            cb.activated.connect(lambda idx, cb=cb: self._add_tag_from(cb, idx))
            self.tag_combos[cat] = cb
            grid.addWidget(cb, i // 4, i % 4)
        stl.addLayout(grid)
        r = QHBoxLayout()
        for txt, fn in [("🎲 Véletlen stílus", self._random_style), ("Utolsó tag törlése", self._pop_tag),
                        ("Rendezés/duplikátum ki", self._clean_tags), ("🤖 Ollama: leírásból tagek", self._ollama_style)]:
            b = QPushButton(txt)
            b.clicked.connect(fn)
            r.addWidget(b)
        r.addStretch()
        stl.addLayout(r)
        fl.addWidget(g_st)

        # Mód + modell
        g_md = QGroupBox("Mód és modell")
        mdl = QFormLayout(g_md)
        self.cb_cot = QComboBox()
        self.cb_cot.addItem("off - gyors, terv nélkül", "off")
        self.cb_cot.addItem("melody - dallamterv (ABC), saját ABC megadható", "melody")
        self.cb_cot.addItem("full - dallam + akkord terv (legjobb szerkezet)", "full")
        self.cb_cot.setCurrentIndex(2)
        mdl.addRow("COT mód:", self.cb_cot)
        self.chk_use_abc = QCheckBox("Az 'ABC-kotta' fül tartalmát használja tervként (melody/full módban)")
        mdl.addRow("", self.chk_use_abc)
        self.chk_instr = QCheckBox("Instrumentális (ének nélkül: [instrumental], full COT, instrumentális LoRA ha van)")
        self.chk_instr.toggled.connect(self._instr_toggled)
        mdl.addRow("", self.chk_instr)
        self.cb_main = QComboBox()
        self.cb_vae = QComboBox()
        mdl.addRow("Fő modell:", self.cb_main)
        mdl.addRow("VAE:", self.cb_vae)
        self.cb_wt_model = QComboBox()
        self.cb_wt_model.addItems(WEIGHT_TYPES)
        self.cb_wt_vae = QComboBox()
        self.cb_wt_vae.addItems(WEIGHT_TYPES)
        r = QHBoxLayout()
        r.addWidget(QLabel("modell:"))
        r.addWidget(self.cb_wt_model)
        r.addWidget(QLabel("VAE:"))
        r.addWidget(self.cb_wt_vae)
        r.addStretch()
        mdl.addRow("Súly tárolás:", r)
        fl.addWidget(g_md)

        # Dal hossza
        g_len = QGroupBox("Dal hossza")
        lenl = QVBoxLayout(g_len)
        r = QHBoxLayout()
        self.chk_len = QCheckBox("Hossz korlátozása:")
        self.chk_len.setToolTip(tip_html("Hossz korlátozása|Bekapcsolva a dal legfeljebb a csúszkán beállított hosszú "
                                         "lesz. Kikapcsolva a szöveg hossza dönt (felső határ kb. 6 perc).\n"
                                         "Ének nélküli (instrumentális) dalnál erősen ajánlott, mert ott nincs szöveg, "
                                         "amihez a modell igazodhatna."))
        self.sl_len = QSlider(Qt.Horizontal)
        self.sl_len.setRange(15, 360)
        self.sl_len.setSingleStep(5)
        self.sl_len.setPageStep(30)
        self.sl_len.setValue(150)
        self.sl_len.setToolTip(tip_html("Dal hossza|Húzd a kívánt hosszra (15 másodperctől 6 percig). Nyilakkal "
                                        "5 másodpercenként, PageUp/PageDown-nal fél percenként léptethető."))
        self.sp_len = QSpinBox()
        self.sp_len.setRange(15, 360)
        self.sp_len.setSingleStep(5)
        self.sp_len.setSuffix(" mp")
        self.sp_len.setValue(150)
        self.sp_len.setToolTip(tip_html("Hossz másodpercben|Pontos érték begépelése."))
        self.sl_len.valueChanged.connect(self.sp_len.setValue)
        self.sp_len.valueChanged.connect(self.sl_len.setValue)
        self.lbl_len = QLabel("")
        self.lbl_len.setMinimumWidth(70)
        self.lbl_len.setToolTip(tip_html("Hossz|Perc:másodperc formában, és hogy ez hány tokennek felel meg "
                                         "(25 token = 1 másodperc)."))
        r.addWidget(self.chk_len)
        r.addWidget(self.sl_len, 1)
        r.addWidget(self.sp_len)
        r.addWidget(self.lbl_len)
        lenl.addLayout(r)
        r = QHBoxLayout()
        self.chk_len_min = QCheckBox("Ne legyen sokkal rövidebb (legalább a 85%-a)")
        self.chk_len_min.setChecked(True)
        self.chk_len_min.setToolTip(tip_html("Legalább ennyi|Bekapcsolva a modell nem fejezheti be a dalt a "
                                             "beállított hossz 85%-a előtt. Így pl. 2:00-nál 1:42 és 2:00 közötti "
                                             "dal lesz. Kikapcsolva rövidebb is lehet, ha hamar elfogy a szöveg."))
        r.addWidget(self.chk_len_min)
        r.addStretch()
        lenl.addLayout(r)
        self.lbl_len_hint = QLabel("")
        self.lbl_len_hint.setWordWrap(True)
        self.lbl_len_hint.setObjectName("dim")
        self.lbl_len_hint.setToolTip(tip_html("Becslés|A dalszöveg alapján becsült hossz összevetése a beállítással. "
                                              "Csak tájékoztató: a valódi hossz a tempótól és a stílustól is függ."))
        lenl.addWidget(self.lbl_len_hint)
        for w_ in (self.sl_len, self.sp_len, self.chk_len_min):
            w_.setEnabled(False)
        self.chk_instr.toggled.connect(self._length_hint)
        self.chk_len.toggled.connect(lambda on: [w_.setEnabled(on) for w_ in (self.sl_len, self.sp_len, self.chk_len_min)])
        for sig in (self.chk_len.toggled, self.sl_len.valueChanged, self.chk_len_min.toggled):
            sig.connect(self._length_hint)
        fl.addWidget(g_len)

        # Seed + mennyiség
        g_sd = QGroupBox("Seed és mennyiség")
        sdl = QHBoxLayout(g_sd)
        self.sp_seed = QSpinBox()
        self.sp_seed.setRange(0, 2_147_483_647)
        self.sp_seed.setValue(1234)
        self.chk_rand = QCheckBox("véletlen")
        self.chk_rand.setChecked(True)
        self.chk_rand.toggled.connect(lambda on: self.sp_seed.setEnabled(not on))
        self.sp_seed.setEnabled(False)
        self.sp_batch = QSpinBox()
        self.sp_batch.setRange(1, 50)
        self.sp_batch.setValue(1)
        self.cb_batch_mode = QComboBox()
        self.cb_batch_mode.addItem("seed, seed+1, seed+2…", "inc")
        self.cb_batch_mode.addItem("mind véletlen seed", "rand")
        self.cb_batch_mode.addItem("azonos seed (más COT/paraméter próbához)", "same")
        sdl.addWidget(QLabel("Seed:"))
        sdl.addWidget(self.sp_seed)
        sdl.addWidget(self.chk_rand)
        sdl.addSpacing(16)
        sdl.addWidget(QLabel("Darab:"))
        sdl.addWidget(self.sp_batch)
        sdl.addWidget(self.cb_batch_mode)
        sdl.addStretch()
        fl.addWidget(g_sd)

        # Haladó
        adv = QWidget()
        al = QVBoxLayout(adv)
        al.setContentsMargins(0, 0, 0, 0)
        g_core = QGroupBox("Fő paraméterek")
        cfl = QFormLayout(g_core)
        self.opt_guid = OptNum(float, 0.0, 20.0, 0.05, 1.0,
                               "Classifier-free guidance: mennyire kövesse a stílust/szöveget. off módban 1.01 az alap.")
        self.opt_steps = OptNum(int, 1, 200, 1, 8, "NAR ODE lépések: több = esetleg tisztább, lassabb.")
        cfl.addRow("Guidance scale:", self.opt_guid)
        cfl.addRow("Inferencia lépések:", self.opt_steps)
        al.addWidget(g_core)

        g_sem = QGroupBox("Semantic sampling (hangtokenek)")
        sfl = QFormLayout(g_sem)
        self.opt_sem = {}
        for key, label, kind, lo, hi, step, tip in SAMPLING:
            w = OptNum(kind, lo, hi, step, SEM_DEFAULTS[key], tip)
            self.opt_sem[key] = w
            sfl.addRow(label + ":", w)
        al.addWidget(g_sem)

        g_abc = QGroupBox("ABC planner sampling (csak melody/full, ha a modell tervez)")
        afl = QFormLayout(g_abc)
        self.opt_abc = {}
        for key, label, kind, lo, hi, step, tip in SAMPLING:
            w = OptNum(kind, lo, hi, step, ABC_DEFAULTS[key], tip)
            self.opt_abc[key] = w
            afl.addRow(label + ":", w)
        al.addWidget(g_abc)

        g_lora = QGroupBox("LoRA adapterek (a modellmappában lévő .safetensors fájlok)")
        lfl = QFormLayout(g_lora)
        self.cb_ar_lora = QComboBox()
        self.sp_ar_scale = QDoubleSpinBox()
        self.sp_ar_scale.setRange(-5, 5)
        self.sp_ar_scale.setSingleStep(0.05)
        self.sp_ar_scale.setValue(1.0)
        self.cb_nar_lora = QComboBox()
        self.sp_nar_scale = QDoubleSpinBox()
        self.sp_nar_scale.setRange(-5, 5)
        self.sp_nar_scale.setSingleStep(0.05)
        self.sp_nar_scale.setValue(1.0)
        r = QHBoxLayout()
        r.addWidget(self.cb_ar_lora, 1)
        r.addWidget(QLabel("erő:"))
        r.addWidget(self.sp_ar_scale)
        lfl.addRow("AR LoRA (terv/szemantika):", r)
        r = QHBoxLayout()
        r.addWidget(self.cb_nar_lora, 1)
        r.addWidget(QLabel("erő:"))
        r.addWidget(self.sp_nar_scale)
        lfl.addRow("NAR LoRA (hangzás):", r)
        r = QHBoxLayout()
        b = QPushButton("LoRA fájl hozzáadása…")
        b.clicked.connect(self._add_lora_file)
        r.addWidget(b)
        b = QPushButton("Lista frissítése")
        b.clicked.connect(self.refresh_install_state)
        r.addWidget(b)
        r.addStretch()
        lfl.addRow("", r)
        al.addWidget(g_lora)

        g_mem = QGroupBox("GPU memória arenák (szakértői - csak ha hibát kapsz)")
        mfl = QFormLayout(g_mem)
        self.opt_arena = {}
        for key, label, default in ARENAS:
            w = OptNum(int, 1, 65536, 256, default)
            self.opt_arena[key] = w
            mfl.addRow(label + ":", w)
        al.addWidget(g_mem)
        fl.addWidget(collapsible("Haladó beállítások", adv, False))

        # Presetek
        g_pr = QGroupBox("Presetek")
        prl = QHBoxLayout(g_pr)
        self.cb_preset = QComboBox()
        self.cb_preset.setMinimumWidth(220)
        prl.addWidget(self.cb_preset, 1)
        for txt, fn in [("Betöltés", self._load_preset), ("Mentés másként…", self._save_preset),
                        ("Preset törlése", self._delete_preset), ("Alaphelyzet", self._reset_form)]:
            b = QPushButton(txt)
            b.clicked.connect(fn)
            prl.addWidget(b)
        fl.addWidget(g_pr)

        # Cím + gomb
        r = QHBoxLayout()
        self.ed_title = QLineEdit()
        self.ed_title.setPlaceholderText("Cím (opcionális - különben seed + stílus)")
        r.addWidget(self.ed_title, 1)
        self.btn_gen = QPushButton("▶  Sorba állít és generál")
        self.btn_gen.setObjectName("primary")
        self.btn_gen.clicked.connect(self.enqueue_generation)
        r.addWidget(self.btn_gen)
        fl.addLayout(r)
        fl.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(form_w)
        root.addWidget(scroll)

        # ---------- jobb oldal: sor, eredmény, napló
        right = QWidget()
        rl = QVBoxLayout(right)
        g_q = QGroupBox("Feladatsor")
        ql = QVBoxLayout(g_q)
        self.tbl_queue = QTableWidget(0, 5)
        self.tbl_queue.setHorizontalHeaderLabels(["#", "Cím / stílus", "Seed", "Állapot", "Idő"])
        self.tbl_queue.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tbl_queue.verticalHeader().setVisible(False)
        self.tbl_queue.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_queue.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl_queue.cellDoubleClicked.connect(self._queue_double_clicked)
        ql.addWidget(self.tbl_queue)
        r = QHBoxLayout()
        self.btn_pause = QPushButton("⏸ Sor szüneteltetése")
        self.btn_pause.setCheckable(True)
        self.btn_pause.toggled.connect(self._pause_toggled)
        r.addWidget(self.btn_pause)
        b = QPushButton("Kijelölt törlése a sorból")
        b.clicked.connect(self._remove_selected_job)
        r.addWidget(b)
        b = QPushButton("Várakozók törlése")
        b.clicked.connect(self._clear_pending)
        r.addWidget(b)
        b = QPushButton("⛔ Azonnali megszakítás")
        b.setObjectName("danger")
        b.setToolTip("A futó generálást csak a szerver újraindításával lehet leállítani.")
        b.clicked.connect(self._abort_now)
        r.addWidget(b)
        ql.addLayout(r)
        rl.addWidget(g_q, 2)

        g_res = QGroupBox("Utolsó eredmény")
        resl = QVBoxLayout(g_res)
        self.lbl_last = QLabel("Még nincs eredmény.")
        self.lbl_last.setWordWrap(True)
        resl.addWidget(self.lbl_last)
        r = QHBoxLayout()
        self.btn_play_last = QPushButton("▶ Lejátszás")
        self.btn_play_last.clicked.connect(lambda: self._play_track(self.current_track))
        self.btn_folder_last = QPushButton("📂 Mappa")
        self.btn_folder_last.clicked.connect(lambda: self.current_track and open_in_explorer(Path(self.current_track["wav_path"])))
        self.btn_abc_last = QPushButton("Generált ABC → ABC fül")
        self.btn_abc_last.clicked.connect(self._last_abc_to_editor)
        self.btn_again = QPushButton("Ugyanez új seeddel")
        self.btn_again.clicked.connect(self._again_new_seed)
        for b in (self.btn_play_last, self.btn_folder_last, self.btn_abc_last, self.btn_again):
            b.setEnabled(False)
            r.addWidget(b)
        resl.addLayout(r)
        rl.addWidget(g_res)

        g_log = QGroupBox("Napló")
        lgl = QVBoxLayout(g_log)
        self.gen_log = QPlainTextEdit()
        self.gen_log.setReadOnly(True)
        self.gen_log.setMaximumBlockCount(3000)
        lgl.addWidget(self.gen_log)
        rl.addWidget(g_log, 2)
        root.addWidget(right)
        root.setSizes([820, 680])
        return root

    # --- dalszöveg / stílus segédek
    def _insert_section(self, tag: str):
        c = self.ed_lyrics.textCursor()
        if c.positionInBlock() > 0:
            c.insertText("\n")
        c.insertText(tag + "\n")
        self.ed_lyrics.setTextCursor(c)
        self.ed_lyrics.setFocus()

    def _lyrics_stats(self):
        t = self.ed_lyrics.toPlainText()
        lines = [l for l in t.splitlines() if l.strip()]
        sections = [l for l in lines if re.match(r"^\s*\[.+\]\s*$", l)]
        words = len(re.findall(r"\w+", "\n".join(l for l in lines if l not in sections)))
        est = estimate_lyrics_seconds(t)
        self.lbl_lyrics.setText(f"{len(sections)} szakasz · {len(lines) - len(sections)} sor · {words} szó"
                                + (f" · kb. {fmt_mmss(est)} ének" if est else ""))
        if hasattr(self, "lbl_len_hint"):
            self._length_hint()

    def _length_hint(self, *_):
        if not hasattr(self, "lbl_len_hint"):
            return
        sec = self.sl_len.value()
        self.lbl_len.setText(f"{fmt_mmss(sec)}  ({sec * TOKENS_PER_SEC} token)")
        est = estimate_lyrics_seconds(self.ed_lyrics.toPlainText())
        instr = self.chk_instr.isChecked() or est is None
        on = self.chk_len.isChecked()
        col, msg = "#a8a9b3", ""
        if not on:
            if instr:
                col, msg = "#facc15", ("⚠ Nincs (énekelt) szöveg: a modell nem tudja, mikor fejezze be, ezért akár "
                                       "6 perces dal is lehet, és sokáig tart. Kapcsold be a hossz korlátozását!")
            else:
                msg = f"A szöveg alapján kb. {fmt_mmss(est)} ének + intro/outro. Felső határ korlátozás nélkül: kb. 6 perc."
        elif instr:
            msg = f"Instrumentális: kb. {fmt_mmss(sec * (0.85 if self.chk_len_min.isChecked() else 0.3))}–{fmt_mmss(sec)} hosszú lesz."
        elif est > sec * 1.1:
            col, msg = "#f87171", (f"⚠ A szöveg kb. {fmt_mmss(est)} – a beállított korlát ({fmt_mmss(sec)}) valószínűleg "
                                   "levágja a végét (befejezés nélkül). Rövidítsd a szöveget, vagy növeld a hosszt.")
        elif self.chk_len_min.isChecked() and est < sec * 0.6:
            col, msg = "#facc15", (f"A szöveg csak kb. {fmt_mmss(est)} – a dal hosszabb lesz nála: a modell "
                                   "hangszeres részekkel vagy ismétléssel tölti ki.")
        else:
            col, msg = "#4ade80", f"✓ A szöveg (kb. {fmt_mmss(est)}) jól illik a beállított hosszhoz ({fmt_mmss(sec)})."
        self.lbl_len_hint.setText(msg)
        self.lbl_len_hint.setStyleSheet(f"color:{col};")

    def _tags(self) -> list[str]:
        return [t.strip() for t in self.ed_style.text().split(",") if t.strip()]

    def _style_stats(self):
        n = len(self._tags())
        mark = "✓" if 5 <= n <= 8 else ("kevés" if n < 5 else "sok")
        self.lbl_style.setText(f"{n} tag ({mark})")

    def _add_tag_from(self, cb: QComboBox, idx: int):
        if idx <= 0:
            return
        tag = cb.itemText(idx)
        tags = self._tags()
        if tag not in tags:
            tags.append(tag)
        self.ed_style.setText(", ".join(tags))
        cb.setCurrentIndex(0)

    def _random_style(self):
        pick = random.choice
        tags = ["English", pick(TAG_BANK["Műfaj"]), pick(TAG_BANK["Hangulat"]),
                *random.sample(TAG_BANK["Hangszer"], 2), pick(TAG_BANK["Ének"]),
                pick(TAG_BANK["Tempó"]), pick(TAG_BANK["Produkció"])]
        self.ed_style.setText(", ".join(tags))

    def _pop_tag(self):
        tags = self._tags()[:-1]
        self.ed_style.setText(", ".join(tags))

    def _clean_tags(self):
        seen, out = set(), []
        for t in self._tags():
            if t.lower() not in seen:
                seen.add(t.lower())
                out.append(t)
        self.ed_style.setText(", ".join(out))

    def _lyrics_from_file(self):
        f, _ = QFileDialog.getOpenFileName(self, "Dalszöveg betöltése", str(APP_DIR), "Szöveg (*.txt *.md);;Minden (*)")
        if f:
            self.ed_lyrics.setPlainText(Path(f).read_text(encoding="utf-8", errors="replace"))

    def _lyrics_to_file(self):
        f, _ = QFileDialog.getSaveFileName(self, "Dalszöveg mentése", str(APP_DIR / "dalszoveg.txt"), "Szöveg (*.txt)")
        if f:
            Path(f).write_text(self.ed_lyrics.toPlainText(), encoding="utf-8")

    def _instr_toggled(self, on: bool):
        if on:
            self.cb_cot.setCurrentIndex(2)
            if not self.ed_lyrics.toPlainText().strip():
                self.ed_lyrics.setPlainText("[instrumental]")
            i = self.cb_ar_lora.findText("ar_lora_inst_v3abc.safetensors")
            if i >= 0:
                self.cb_ar_lora.setCurrentIndex(i)

    def _add_lora_file(self):
        f, _ = QFileDialog.getOpenFileName(self, "LoRA (.safetensors) kiválasztása", str(APP_DIR), "SafeTensors (*.safetensors)")
        if not f:
            return
        dst = Path(self.cfg["models_dir"]) / YUE2_DIRNAME / Path(f).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if Path(f).resolve() != dst.resolve():
            shutil.copy2(f, dst)
        self.refresh_install_state()
        self.log(f"LoRA bemásolva: {dst.name}")

    # --- űrlap állapot
    def get_form(self) -> dict:
        return {
            "lyrics": self.ed_lyrics.toPlainText(), "style": self.ed_style.text(), "title": self.ed_title.text(),
            "cot": self.cb_cot.currentData(), "use_abc": self.chk_use_abc.isChecked(),
            "instrumental": self.chk_instr.isChecked(), "main": self.cb_main.currentText(),
            "vae": self.cb_vae.currentText(), "wt_model": self.cb_wt_model.currentText(),
            "wt_vae": self.cb_wt_vae.currentText(), "seed": self.sp_seed.value(),
            "random_seed": self.chk_rand.isChecked(), "batch": self.sp_batch.value(),
            "batch_mode": self.cb_batch_mode.currentData(), "guidance": self.opt_guid.value(),
            "steps": self.opt_steps.value(), "sem": {k: w.value() for k, w in self.opt_sem.items()},
            "abc": {k: w.value() for k, w in self.opt_abc.items()},
            "arena": {k: w.value() for k, w in self.opt_arena.items()},
            "ar_lora": self.cb_ar_lora.currentText(), "ar_scale": self.sp_ar_scale.value(),
            "nar_lora": self.cb_nar_lora.currentText(), "nar_scale": self.sp_nar_scale.value(),
            "length_on": self.chk_len.isChecked(), "length_sec": self.sl_len.value(),
            "length_min": self.chk_len_min.isChecked(),
        }

    def set_form(self, f: dict):
        def set_combo_text(cb, txt):
            if txt is None:
                return
            i = cb.findText(txt)
            if i >= 0:
                cb.setCurrentIndex(i)
        if "lyrics" in f:
            self.ed_lyrics.setPlainText(f["lyrics"])
        if "style" in f:
            self.ed_style.setText(f["style"])
        self.ed_title.setText(f.get("title", ""))
        i = self.cb_cot.findData(f.get("cot", "full"))
        self.cb_cot.setCurrentIndex(i if i >= 0 else 2)
        self.chk_use_abc.setChecked(bool(f.get("use_abc")))
        self.chk_instr.blockSignals(True)
        self.chk_instr.setChecked(bool(f.get("instrumental")))
        self.chk_instr.blockSignals(False)
        set_combo_text(self.cb_main, f.get("main"))
        set_combo_text(self.cb_vae, f.get("vae"))
        set_combo_text(self.cb_wt_model, f.get("wt_model"))
        set_combo_text(self.cb_wt_vae, f.get("wt_vae"))
        if "seed" in f:
            self.sp_seed.setValue(int(f["seed"]))
        self.chk_rand.setChecked(bool(f.get("random_seed", True)))
        self.sp_batch.setValue(int(f.get("batch", 1)))
        i = self.cb_batch_mode.findData(f.get("batch_mode", "inc"))
        self.cb_batch_mode.setCurrentIndex(max(i, 0))
        self.opt_guid.set_value(f.get("guidance"))
        self.opt_steps.set_value(f.get("steps"))
        for k, w in self.opt_sem.items():
            w.set_value((f.get("sem") or {}).get(k))
        for k, w in self.opt_abc.items():
            w.set_value((f.get("abc") or {}).get(k))
        for k, w in self.opt_arena.items():
            w.set_value((f.get("arena") or {}).get(k))
        set_combo_text(self.cb_ar_lora, f.get("ar_lora"))
        set_combo_text(self.cb_nar_lora, f.get("nar_lora"))
        self.sp_ar_scale.setValue(float(f.get("ar_scale", 1.0)))
        self.sp_nar_scale.setValue(float(f.get("nar_scale", 1.0)))
        self.chk_len.setChecked(bool(f.get("length_on", False)))
        self.sl_len.setValue(int(f.get("length_sec", 150)))
        self.chk_len_min.setChecked(bool(f.get("length_min", True)))
        self._length_hint()

    def _reset_form(self):
        self.set_form({"lyrics": LYRICS_EXAMPLE, "style": STYLE_EXAMPLE, "cot": "full", "random_seed": True,
                       "batch": 1, "wt_model": "native", "wt_vae": "native", "ar_lora": "(nincs)",
                       "nar_lora": "(nincs)"})

    # --- presetek
    def _refresh_presets(self):
        self.cb_preset.clear()
        for p in sorted(PRESET_DIR.glob("*.json")):
            self.cb_preset.addItem(p.stem)

    def _save_preset(self):
        name, ok = QInputDialog.getText(self, "Preset mentése", "Név:")
        if ok and name.strip():
            p = PRESET_DIR / f"{safe_name(name, 80)}.json"
            p.write_text(json.dumps(self.get_form(), ensure_ascii=False, indent=2), encoding="utf-8")
            self._refresh_presets()
            self.cb_preset.setCurrentText(p.stem)

    def _load_preset(self):
        n = self.cb_preset.currentText()
        if n:
            self.set_form(json.loads((PRESET_DIR / f"{n}.json").read_text(encoding="utf-8")))
            self.log(f"Preset betöltve: {n}")

    def _delete_preset(self):
        n = self.cb_preset.currentText()
        if n and QMessageBox.question(self, "Törlés", f"Törlöd a(z) '{n}' presetet?") == QMessageBox.Yes:
            (PRESET_DIR / f"{n}.json").unlink(missing_ok=True)
            self._refresh_presets()

    # --- feladat összeállítása
    def _build_options(self, f: dict) -> dict:
        opts: dict = {"style": f["style"].strip(), "cot": f["cot"]}
        if f["guidance"] is not None:
            opts["guidance_scale"] = f["guidance"]
        if f["steps"] is not None:
            opts["num_inference_steps"] = f["steps"]
        for k, v in f["sem"].items():
            if v is not None:
                opts[f"semantic_{k}"] = v
        if f.get("length_on"):
            mx = int(f.get("length_sec", 150)) * TOKENS_PER_SEC
            opts["semantic_max_tokens"] = mx
            mn = int(mx * 0.85) if f.get("length_min", True) else min(int(opts.get("semantic_min_tokens") or 200), mx)
            opts["semantic_min_tokens"] = mn
        if f["cot"] != "off":
            for k, v in f["abc"].items():
                if v is not None:
                    opts[f"abc_{k}"] = v
            if f["use_abc"] and self.ed_abc.toPlainText().strip():
                opts["abc"] = self.ed_abc.toPlainText().strip()
        return opts

    def _build_session(self, f: dict) -> dict:
        s = {"yue2.model_gguf": f["main"], "yue2.vae_gguf": f["vae"]}
        if f["wt_model"] != "native":
            s["yue2.model_weight_type"] = f["wt_model"]
        if f["wt_vae"] != "native":
            s["yue2.vae_weight_type"] = f["wt_vae"]
        if f["ar_lora"] and f["ar_lora"] != "(nincs)":
            s["yue2.ar_lora"] = f["ar_lora"]
            s["yue2.ar_lora_scale"] = f"{f['ar_scale']:g}"
        if f["nar_lora"] and f["nar_lora"] != "(nincs)":
            s["yue2.nar_lora"] = f["nar_lora"]
            s["yue2.nar_lora_scale"] = f"{f['nar_scale']:g}"
        for k, v in f["arena"].items():
            if v is not None:
                s[f"yue2.{k}"] = str(v)
        return s

    def enqueue_generation(self):
        f = self.get_form()
        if f["instrumental"]:
            f["cot"] = "full"
            if not f["lyrics"].strip():
                f["lyrics"] = "[instrumental]"
        if not f["lyrics"].strip():
            QMessageBox.warning(self, APP_NAME, "Add meg a dalszöveget (vagy jelöld be: Instrumentális).")
            return
        if not f["style"].strip():
            QMessageBox.warning(self, APP_NAME, "Add meg a stílust (tageket).")
            return
        if not f["main"] or not f["vae"]:
            QMessageBox.warning(self, APP_NAME, "Nincs letöltött fő modell vagy VAE. Nézd meg a 'Telepítés & Szerver' fület.")
            return
        if f["use_abc"] and f["cot"] == "off":
            QMessageBox.information(self, APP_NAME, "Saját ABC-kottához melody vagy full COT mód kell - átállítom melody-ra.")
            f["cot"] = "melody"
            self.cb_cot.setCurrentIndex(1)
        opts = self._build_options(f)
        sess = self._build_session(f)
        self._enqueue_jobs(f, opts, sess)
        self.cfg["last_form"] = f
        save_settings(self.cfg)

    def enqueue_custom(self, lyrics: str, style: str, abc: str, cot: str, count: int, title: str,
                       instrumental: bool = False, length_sec: int | None = None) -> list[int]:
        """Generálás a Generálás fül haladó beállításaival, de megadott szöveggel/stílussal/tervvel (Átdolgozás)."""
        f = self.get_form()
        if not f["main"] or not f["vae"]:
            QMessageBox.warning(self, APP_NAME, "Nincs letöltött fő modell vagy VAE. Nézd meg a 'Telepítés & Szerver' fület.")
            return []
        f.update({"lyrics": lyrics, "style": style, "cot": cot if abc else ("full" if instrumental else "off"),
                  "instrumental": instrumental, "use_abc": False, "title": title, "batch": max(1, int(count)),
                  "batch_mode": "rand", "random_seed": True})
        if length_sec:
            f.update({"length_on": True, "length_sec": int(max(15, min(360, length_sec))), "length_min": True})
        opts = self._build_options(f)
        if abc:
            opts["abc"] = abc
        return self._enqueue_jobs(f, opts, self._build_session(f))

    def _enqueue_jobs(self, f: dict, opts: dict, sess: dict) -> list[int]:
        jids = []
        base = random.randint(0, 2_147_483_647) if f["random_seed"] else f["seed"]
        for i in range(f["batch"]):
            if f["batch_mode"] == "rand" and i > 0:
                seed = random.randint(0, 2_147_483_647)
            elif f["batch_mode"] == "same":
                seed = base
            else:
                seed = base + i
            self.job_counter += 1
            jid = self.job_counter
            title = f["title"].strip()
            if title and f["batch"] > 1:
                title = f"{title} ({i + 1})"
            job = {"id": jid, "kind": "gen", "project": self.project_name, "lyrics": f["lyrics"].strip(), "options": dict(opts),
                   "session": dict(sess), "seed": seed, "title": title,
                   "form": {**f, "seed": seed, "random_seed": False, "batch": 1}}
            row = self.tbl_queue.rowCount()
            self.tbl_queue.insertRow(row)
            vals = [str(jid), title or f["style"][:70], str(seed), tr("várakozik"), ""]
            for c, v in enumerate(vals):
                self.tbl_queue.setItem(row, c, QTableWidgetItem(v))
            self.job_rows[jid] = row
            self.worker.add(job)
            jids.append(jid)
        self._update_queue_label()
        if not self.server_ok:
            self.log("Figyelem: a szerver még nem fut - a feladat várakozik, indítsd el a szervert.")
        return jids

    def _set_job_cell(self, jid: int, col: int, text: str, color: str | None = None):
        row = self.job_rows.get(jid)
        if row is None:
            return
        it = self.tbl_queue.item(row, col)
        if it:
            it.setText(text)
            if color:
                it.setForeground(QColor(color))

    def _job_started(self, jid: int):
        self.job_t0[jid] = time.time()
        self._set_job_cell(jid, 3, tr("▶ fut…"), "#facc15")

    def _tick_elapsed(self):
        for jid, t0 in list(self.job_t0.items()):
            s = int(time.time() - t0)
            self._set_job_cell(jid, 4, f"{s // 60}:{s % 60:02d}")

    def _queue_double_clicked(self, row: int, col: int):
        it = self.tbl_queue.item(row, 0)
        if it and int(it.text()) in self.job_track:
            self._play_track(self.db.get(self.job_track[int(it.text())]))

    def _job_finished(self, jid: int, info: dict):
        self.job_t0.pop(jid, None)
        self.job_track[jid] = info["track_id"]
        self._set_job_cell(jid, 3, tr("✓ kész"), "#4ade80")
        self.current_track = self.db.get(info["track_id"])
        self.last_abc = info.get("abc", "")
        self.lbl_last.setText(f"<b>{Path(info['wav']).name}</b><br>{info['dur']:.0f} mp hang, "
                              f"{info['secs']:.0f} mp generálási idő" + (" · ABC terv mentve" if self.last_abc else ""))
        for b in (self.btn_play_last, self.btn_folder_last, self.btn_again):
            b.setEnabled(True)
        self.btn_abc_last.setEnabled(bool(self.last_abc))
        self.refresh_library()
        self._update_queue_label()
        if self.remake:
            self.remake.on_result(jid, info)

    def _job_failed(self, jid: int, msg: str):
        if self.remake and jid in self.remake.jobs:
            self.remake.jobs.discard(jid)
            self.remake.say(f"HIBA egy változatnál: {msg}")
        self.job_t0.pop(jid, None)
        self._set_job_cell(jid, 3, tr("✗ hiba"), "#f87171")
        row = self.job_rows.get(jid)
        if row is not None and self.tbl_queue.item(row, 3):
            self.tbl_queue.item(row, 3).setToolTip(msg)
        self._update_queue_label()

    def _update_queue_label(self):
        n = len(self.worker.jobs)
        self.sb_queue.setText(f"Sor: {n} feladat")

    def _pause_toggled(self, on: bool):
        self.worker.paused = on
        if not on:
            self.worker.resume()
        self.btn_pause.setText("▶ Sor folytatása" if on else "⏸ Sor szüneteltetése")

    def _remove_selected_job(self):
        rows = {i.row() for i in self.tbl_queue.selectedIndexes()}
        for row in rows:
            it = self.tbl_queue.item(row, 0)
            if it:
                jid = int(it.text())
                self.worker.remove(jid)
                if self.tbl_queue.item(row, 3).text() == tr("várakozik"):
                    self._set_job_cell(jid, 3, tr("törölve"), "#80828e")
        self._update_queue_label()

    def _clear_pending(self):
        self.worker.clear_pending()
        for jid, row in self.job_rows.items():
            it = self.tbl_queue.item(row, 3)
            if it and it.text() == tr("várakozik"):
                it.setText(tr("törölve"))
                it.setForeground(QColor("#80828e"))
        self._update_queue_label()

    def _abort_now(self):
        if QMessageBox.question(self, "Megszakítás",
                                "A futó generálás csak a szerver újraindításával állítható le.\n"
                                "A várakozó feladatok is törlődnek. Mehet?") != QMessageBox.Yes:
            return
        self._clear_pending()
        self.restart_server()

    def _again_new_seed(self):
        if not self.current_track:
            return
        p = json.loads(self.current_track["params"] or "{}").get("form")
        if p:
            self.set_form(p)
            self.chk_rand.setChecked(True)
            self.sp_batch.setValue(1)
            self.enqueue_generation()

    def _last_abc_to_editor(self):
        if self.last_abc:
            self.ed_abc.setPlainText(self.last_abc)
            self.show_page("abc")

    # --- Ollama
    def _ollama_ready(self) -> bool:
        if not self.cfg.get("ollama_model"):
            QMessageBox.information(self, "Ollama", "Állíts be Ollama modellt a 'Beállítások' fülön.")
            return False
        if self.ollama_worker and self.ollama_worker.isRunning():
            QMessageBox.information(self, "Ollama", "Már fut egy kérés, várj.")
            return False
        return True

    def _ollama_lyrics(self):
        if not self._ollama_ready():
            return
        theme, ok = QInputDialog.getMultiLineText(self, "Dalszöveg Ollamával",
                                                  "Téma / hangulat / nyelv (pl. 'őszi eső a Dunán, angolul'):")
        if not ok or not theme.strip():
            return
        prompt = ("Write song lyrics for an AI music model. Theme and requirements: " + theme.strip() + "\n"
                  "Rules: use section tags on their own line: [Verse], [Pre-Chorus], [Chorus], [Bridge], [Outro]. "
                  "Short singable lines (6-10 syllables). Repeat the chorus word-for-word every time. "
                  "2 verses, 3 choruses, 1 bridge. Output ONLY the lyrics, no explanations. "
                  "If no language is specified, write in English.")
        self._run_ollama(prompt, "lyrics")

    def _ollama_style(self):
        if not self._ollama_ready():
            return
        desc, ok = QInputDialog.getText(self, "Stílus Ollamával", "Írd le a hangzást (magyarul is lehet):")
        if not ok or not desc.strip():
            return
        prompt = ("Convert this music description into 6-8 comma-separated English style tags for the YuE2 "
                  "music model. Order: language of vocals, genre, mood, 2-3 instruments, vocal type, "
                  "production. Output ONLY the tag line.\nDescription: " + desc.strip())
        self._run_ollama(prompt, "style")

    def _run_ollama(self, prompt: str, target: str):
        self.log("Ollama dolgozik…")
        self.ollama_worker = OllamaWorker(self.cfg["ollama_url"], self.cfg["ollama_model"], prompt, target)
        self.ollama_worker.done.connect(self._ollama_done)
        self.ollama_worker.failed.connect(lambda m: QMessageBox.warning(self, "Ollama hiba", m))
        self.ollama_worker.start()

    def _ollama_done(self, target: str, txt: str):
        if target == "lyrics":
            self.ed_lyrics.setPlainText(txt.strip())
        else:
            self.ed_style.setText(txt.strip().splitlines()[0].strip().strip('"'))
        self.log("✓ Ollama kész.")

    # ============================================================ ABC FÜL
    def _build_abc_tab(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        info = QLabel("Az ABC-kotta a dal szimbolikus terve (dallam + akkordok). melody/full COT módban a "
                      "'Generálás' fülön bejelölve a modell ezt követi. Kinyerhető meglévő dalból is.")
        info.setWordWrap(True)
        info.setObjectName("dim")
        l.addWidget(info)
        r = QHBoxLayout()
        for txt, fn in [("Megnyitás…", self._abc_open), ("Mentés…", self._abc_save),
                        ("Utolsó generált terv", self._last_abc_to_editor), ("Hosszabbítás ×2", lambda: self._abc_ext(2)),
                        ("Hosszabbítás ×3", lambda: self._abc_ext(3)), ("Ének szólam törlése", self._abc_mute),
                        ("🎼 Kottaként", self._abc_to_score)]:
            b = QPushButton(txt)
            b.clicked.connect(fn)
            r.addWidget(b)
        r.addStretch()
        self.btn_clear_abc = clear_button(lambda: self.clear_abc(ask=True), "Az ABC-kotta fül kiürítése: a zenei terv "
                                          "és a kinyeréshez megadott hangfájl. A Generálás fülön a 'terv használata' "
                                          "pipa is kikapcsol.")
        r.addWidget(self.btn_clear_abc)
        l.addLayout(r)
        self.ed_abc = QPlainTextEdit()
        self.ed_abc.setPlaceholderText("X:1\nT:Cím\nM:4/4\nL:1/8\nQ:1/4=120\nK:Am\nV: Vocal\n\"Am\" A2 c2 e2 c2 | …")
        self.ed_abc.textChanged.connect(lambda: self.lbl_abc.setText(abc_stats(self.ed_abc.toPlainText())))
        l.addWidget(self.ed_abc, 1)
        self.lbl_abc = QLabel("Üres")
        self.lbl_abc.setObjectName("dim")
        l.addWidget(self.lbl_abc)
        g = QGroupBox("Kinyerés hangfájlból (SheetSage2)")
        gl = QHBoxLayout(g)
        self.ed_ref = QLineEdit()
        self.ed_ref.setPlaceholderText("Referencia hang (WAV; MP3/FLAC/M4A ffmpeg-gel)")
        gl.addWidget(self.ed_ref, 1)
        b = QPushButton("Tallózás…")
        b.clicked.connect(self._pick_ref)
        gl.addWidget(b)
        self.btn_extract = QPushButton("🎼 ABC kinyerése")
        self.btn_extract.setObjectName("primary")
        self.btn_extract.clicked.connect(self._extract_abc)
        gl.addWidget(self.btn_extract)
        l.addWidget(g)
        return w

    def _abc_open(self):
        f, _ = QFileDialog.getOpenFileName(self, "ABC megnyitása", self.cfg["output_dir"], "ABC (*.abc *.txt);;Minden (*)")
        if f:
            self.ed_abc.setPlainText(Path(f).read_text(encoding="utf-8", errors="replace"))

    def _abc_save(self):
        f, _ = QFileDialog.getSaveFileName(self, "ABC mentése", str(Path(self.cfg["output_dir"]) / "terv.abc"), "ABC (*.abc)")
        if f:
            Path(f).write_text(self.ed_abc.toPlainText(), encoding="utf-8")

    def _abc_ext(self, n: int):
        r = abc_extend(self.ed_abc.toPlainText(), n)
        if r is None:
            QMessageBox.information(self, "ABC", "Nem találtam ismételhető szólamblokkokat (V: …) ebben a kottában.")
        else:
            self.ed_abc.setPlainText(r)

    def _abc_mute(self):
        r = abc_mute_vocal(self.ed_abc.toPlainText())
        if r is None:
            QMessageBox.information(self, "ABC", "Nincs 'V: Vocal' szólamblokk a kottában.")
        else:
            self.ed_abc.setPlainText(r)

    def _pick_ref(self):
        f, _ = QFileDialog.getOpenFileName(self, "Referencia hang", str(APP_DIR),
                                           "Hang (*.wav *.mp3 *.flac *.m4a *.ogg *.aac);;Minden (*)")
        if f:
            self.ed_ref.setText(f)

    def _extract_abc(self):
        p = self.ed_ref.text().strip()
        if not p or not Path(p).is_file():
            QMessageBox.warning(self, "SheetSage2", "Válassz egy létező hangfájlt.")
            return
        self.btn_extract.setEnabled(False)
        self.job_counter += 1
        self.worker.add({"id": self.job_counter, "kind": "sheetsage", "audio": p})
        self.log("SheetSage2 feladat a sorban.")

    def _abc_extracted(self, abc: str):
        self.btn_extract.setEnabled(True)
        self.ed_abc.setPlainText(abc)
        self.chk_use_abc.setChecked(True)
        if self.cb_cot.currentData() == "off":
            self.cb_cot.setCurrentIndex(1)

    # ======================================================== KÖNYVTÁR FÜL
    def _build_library_tab(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        r = QHBoxLayout()
        self.ed_search = QLineEdit()
        self.ed_search.setPlaceholderText("Keresés címben, stílusban, szövegben, megjegyzésben…")
        self.ed_search.textChanged.connect(self.refresh_library)
        r.addWidget(self.ed_search, 1)
        self.chk_fav = QCheckBox("csak kedvencek")
        self.chk_fav.toggled.connect(self.refresh_library)
        r.addWidget(self.chk_fav)
        self.cb_lib_proj = QComboBox()
        self.cb_lib_proj.setMinimumWidth(220)
        self.cb_lib_proj.currentIndexChanged.connect(self.refresh_library)
        lb_ = QLabel("Projekt:")
        lb_.setToolTip(tip_html("Projekt szűrő|Melyik projekt dalait mutassa a lista."))
        r.addWidget(lb_)
        r.addWidget(self.cb_lib_proj)
        self.btn_clear_lib = clear_button(self.clear_library, "")
        self.btn_clear_lib.setText("🧹 Könyvtár ürítése…")
        r.addWidget(self.btn_clear_lib)
        l.addLayout(r)
        self.tbl_lib = QTableWidget(0, 9)
        self.tbl_lib.setHorizontalHeaderLabels(["★", "Dátum", "Cím", "Stílus", "Seed", "COT", "Hossz", "Gen. idő", "Megjegyzés"])
        hh = self.tbl_lib.horizontalHeader()
        hh.setSectionResizeMode(2, QHeaderView.Stretch)
        hh.setSectionResizeMode(3, QHeaderView.Stretch)
        self.tbl_lib.verticalHeader().setVisible(False)
        self.tbl_lib.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_lib.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tbl_lib.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl_lib.setAlternatingRowColors(True)
        self.tbl_lib.cellDoubleClicked.connect(lambda r_, c: self._play_track(self._sel_track()))
        self.tbl_lib.itemSelectionChanged.connect(self._lib_selected)
        l.addWidget(self.tbl_lib, 1)

        # műveletek
        r = QHBoxLayout()
        acts = [("★ Kedvenc", self._toggle_fav), ("✎ Megjegyzés", self._edit_note), ("Átnevezés", self._rename_track),
                ("⟲ Beállítások vissza", self._reuse_track), ("Seed használata", self._reuse_seed),
                ("ABC → ABC fül", self._track_abc), ("📂 Mappa", self._track_folder), ("MP3 export", self._export_mp3),
                ("🎼 Kotta", self._lib_to_score), ("🎛 DAW-ba", self._lib_to_daw), ("✂ Sávokra bontás", self._lib_separate), ("🗑 Törlés", self._delete_track)]
        for txt, fn in acts:
            b = QPushButton(txt)
            if txt.startswith("🗑"):
                b.setObjectName("danger")
            b.clicked.connect(fn)
            r.addWidget(b)
        l.addLayout(r)
        self.lib_details = QPlainTextEdit()
        self.lib_details.setReadOnly(True)
        self.lib_details.setMaximumHeight(170)
        l.addWidget(self.lib_details)
        return w

    def refresh_library(self):
        if not hasattr(self, "tbl_lib"):
            return
        rows = self.db.all(self.ed_search.text().strip(), self.chk_fav.isChecked(), self._lib_project_filter())
        self.lib_ids = [t["id"] for t in rows]
        self.tbl_lib.setRowCount(0)
        for t in rows:
            r = self.tbl_lib.rowCount()
            self.tbl_lib.insertRow(r)
            d = t.get("duration") or 0
            g = t.get("gen_secs") or 0
            vals = ["★" if t.get("fav") else "", (t.get("created") or "").replace("T", " "), t.get("title") or "",
                    t.get("style") or "", str(t.get("seed")), t.get("cot") or "", f"{int(d) // 60}:{int(d) % 60:02d}",
                    f"{int(g) // 60}:{int(g) % 60:02d}", t.get("note") or ""]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if c == 0:
                    it.setData(Qt.UserRole, t["id"])
                    it.setForeground(QColor("#facc15"))
                if not Path(t.get("wav_path") or "").is_file():
                    it.setForeground(QColor("#80828e"))
                if t["id"] == self.now_id:
                    f = it.font()
                    f.setBold(True)
                    it.setFont(f)
                self.tbl_lib.setItem(r, c, it)
        self.tbl_lib.resizeColumnToContents(0)
        self.tbl_lib.resizeColumnToContents(1)

    def _sel_track(self) -> dict | None:
        rows = self.tbl_lib.selectionModel().selectedRows() if self.tbl_lib.selectionModel() else []
        if not rows:
            return None
        tid = self.tbl_lib.item(rows[0].row(), 0).data(Qt.UserRole)
        return self.db.get(int(tid))

    def _lib_selected(self):
        t = self._sel_track()
        if not t:
            self.lib_details.clear()
            return
        p = json.loads(t.get("params") or "{}")
        txt = (f"Fájl: {t['wav_path']}\nModell: {t.get('model')}   COT: {t.get('cot')}   Seed: {t.get('seed')}\n"
               f"Stílus: {t.get('style')}\nOpciók: {json.dumps({k: v for k, v in p.get('options', {}).items() if k not in ('style', 'abc')}, ensure_ascii=False)}\n"
               f"Session: {json.dumps(p.get('session', {}), ensure_ascii=False)}\n\n--- Dalszöveg ---\n{t.get('lyrics')}")
        self.lib_details.setPlainText(txt)

    # ============================================================ LEJÁTSZÓ
    def _build_player_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("playerbar")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(10, 6, 10, 6)
        self.btn_prev = QPushButton("⏮")
        self.btn_play = QPushButton("▶")
        self.btn_play.setObjectName("primary")
        self.btn_stop = QPushButton("■")
        self.btn_next = QPushButton("⏭")
        for b in (self.btn_prev, self.btn_play, self.btn_stop, self.btn_next):
            b.setFixedSize(52, 36)
            f = b.font()
            f.setPointSize(15)
            b.setFont(f)
            lay.addWidget(b)
        self.btn_prev.clicked.connect(lambda: self._step(-1))
        self.btn_next.clicked.connect(lambda: self._step(+1))
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_stop.clicked.connect(self._stop_play)
        self.lbl_now = QLabel("Nincs betöltött szám")
        self.lbl_now.setMinimumWidth(220)
        self.lbl_now.setMaximumWidth(380)
        lay.addWidget(self.lbl_now)
        self.sl_pos = QSlider(Qt.Horizontal)
        self.sl_pos.sliderMoved.connect(lambda v: self.player and self.player.setPosition(v))
        self.sl_pos.sliderReleased.connect(lambda: self.player and self.player.setPosition(self.sl_pos.value()))
        lay.addWidget(self.sl_pos, 1)
        self.lbl_time = QLabel("0:00 / 0:00")
        lay.addWidget(self.lbl_time)
        self.chk_autonext = QCheckBox("következő automatikusan")
        lay.addWidget(self.chk_autonext)
        lay.addWidget(QLabel("🔊"))
        self.sl_vol = QSlider(Qt.Horizontal)
        self.sl_vol.setRange(0, 100)
        self.sl_vol.setValue(80)
        self.sl_vol.setFixedWidth(110)
        lay.addWidget(self.sl_vol)
        self.btn_now_folder = QPushButton("📂")
        self.btn_now_folder.setFixedWidth(40)
        self.btn_now_folder.clicked.connect(self._now_folder)
        lay.addWidget(self.btn_now_folder)

        self.player = None
        if HAS_MULTIMEDIA:
            self.player = QMediaPlayer(self)
            self.audio_out = QAudioOutput(self)
            self.audio_out.setVolume(0.8)
            self.player.setAudioOutput(self.audio_out)
            self.sl_vol.valueChanged.connect(lambda v: self.audio_out.setVolume(v / 100))
            self.player.durationChanged.connect(lambda d: self.sl_pos.setRange(0, d))
            self.player.positionChanged.connect(self._pos_changed)
            self.player.playbackStateChanged.connect(
                lambda st_: self.btn_play.setText("⏸" if st_ == QMediaPlayer.PlayingState else "▶"))
            self.player.mediaStatusChanged.connect(self._media_status)
        # billentyűk: Ctrl+Szóköz = lejátszás/szünet, multimédia gombok
        for seq, fn in (("Ctrl+Space", self._toggle_play), ("Media Play", self._toggle_play),
                        ("Media Pause", self._toggle_play), ("Toggle Media Play/Pause", self._toggle_play),
                        ("Media Stop", self._stop_play), ("Media Next", lambda: self._step(+1)),
                        ("Media Previous", lambda: self._step(-1))):
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.ApplicationShortcut)
            sc.activated.connect(fn)
        return bar

    def _play_track(self, t: dict | None):
        if not t:
            return
        if not self.player:
            open_in_explorer(Path(t["wav_path"]))
            return
        if not Path(t["wav_path"]).is_file():
            QMessageBox.warning(self, APP_NAME, "A fájl nem található.")
            return
        if getattr(self, "daw", None):
            self.daw.engine.pause()
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(t["wav_path"]))
        self.player.play()
        self.now_id = t["id"]
        self.lbl_now.setText(f"♪ {t.get('title')}")
        self.lbl_now.setToolTip(tip_html(f"Most szól|{t.get('title')}\n{t.get('style') or ''}"))
        self._mark_playing_row()

    def play_file(self, path: str, title: str = ""):
        """Tetszőleges hangfájl lejátszása az alsó lejátszósávon (nem könyvtári szám)."""
        if not path or not Path(path).is_file():
            QMessageBox.warning(self, APP_NAME, "A fájl nem található.")
            return
        if not self.player:
            open_in_explorer(Path(path))
            return
        if getattr(self, "daw", None):
            self.daw.engine.pause()
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(str(path)))
        self.player.play()
        self.now_id = None
        self.lbl_now.setText(f"♪ {title or Path(path).stem}")
        self.lbl_now.setToolTip(tip_html(f"Most szól|{title or Path(path).name}\n{path}"))
        self._mark_playing_row()

    def _mark_playing_row(self):
        for r in range(self.tbl_lib.rowCount()):
            it0 = self.tbl_lib.item(r, 0)
            playing = it0 is not None and it0.data(Qt.UserRole) == self.now_id
            for c in range(self.tbl_lib.columnCount()):
                it = self.tbl_lib.item(r, c)
                if it:
                    f = it.font()
                    f.setBold(playing)
                    it.setFont(f)

    def _toggle_play(self):
        if not self.player:
            return
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        elif self.player.source().isEmpty():
            self._play_track(self._sel_track() or self.current_track or
                             (self.db.get(self.lib_ids[0]) if self.lib_ids else None))
        else:
            self.player.play()

    def _stop_play(self):
        if self.player:
            self.player.stop()

    def _step(self, d: int):
        if not self.lib_ids:
            return
        if self.now_id in self.lib_ids:
            i = (self.lib_ids.index(self.now_id) + d) % len(self.lib_ids)
        else:
            i = 0
        t = self.db.get(self.lib_ids[i])
        if t:
            self.tbl_lib.selectRow(i)
            self._play_track(t)

    def _media_status(self, status):
        if status == QMediaPlayer.EndOfMedia and self.chk_autonext.isChecked() and self.now_id is not None:
            QTimer.singleShot(300, lambda: self._step(+1))

    def _now_folder(self):
        t = self.db.get(self.now_id) if self.now_id else None
        if t:
            open_in_explorer(Path(t["wav_path"]))

    def _pos_changed(self, p: int):
        if not self.sl_pos.isSliderDown():
            self.sl_pos.setValue(p)
        d = self.player.duration() if self.player else 0
        self.lbl_time.setText(f"{p // 60000}:{p // 1000 % 60:02d} / {d // 60000}:{d // 1000 % 60:02d}")

    # ---------------------------------------------------- Kotta kapcsolat
    def _abc_to_score(self):
        if not self.score:
            QMessageBox.information(self, "Kotta", "A kottanézethez frissíteni kell a PySide6 csomagot (lásd a Kotta fület).")
            return
        self.score.from_abc_tab()
        self.tabs.setCurrentWidget(self.score)

    def _lib_to_score(self):
        if not self.score:
            return
        self.score.from_library()
        self.tabs.setCurrentWidget(self.score)

    # ---------------------------------------------------- Kalauz
    def start_tour(self, key: str):
        tours = build_tours(self)
        if key not in tours:
            QMessageBox.information(self, "Kalauz", "Ez a bemutató most nem érhető el (hiányzó csomag miatt).")
            return
        if getattr(self, "_tour", None):
            self._tour.finish()
        title, steps = tours[key]
        self._tour = TourOverlay(self, steps, title)

    def tour_menu(self):
        m = QMenu(self)
        for key, (title, _steps) in build_tours(self).items():
            m.addAction(f"🧭 {title}", lambda k=key: self.start_tour(k))
        m.exec(self.btn_tour.mapToGlobal(QPoint(0, -m.sizeHint().height())))

    def _offer_tour(self):
        if self.cfg.get("tour_offered"):
            return
        self.cfg["tour_offered"] = True
        save_settings(self.cfg)
        if QMessageBox.question(self, "Első lépések",
                                "Szeretnél egy rövid, 2 perces bemutatót az első dalodhoz?\n\n"
                                "(Később is bármikor elindíthatod: lent a 🧭 Kalauz gomb, vagy a Súgó fül.)") == QMessageBox.Yes:
            self.start_tour("first")

    # ---------------------------------------------------- DAW kapcsolat
    def _lib_to_daw(self):
        if not self.daw:
            QMessageBox.information(self, "DAW", "A DAW-hoz hiányzó csomagok kellenek (lásd a DAW fület).")
            return
        t = self._sel_track()
        if t:
            self.daw.import_from_library(t)
            self.tabs.setCurrentWidget(self.daw)

    def _lib_separate(self):
        t = self._sel_track()
        if not t or not self.daw:
            return
        self.request_separation(t["wav_path"], {"at": 0.0, "title": t.get("title")})
        self.tabs.setCurrentWidget(self.daw)
        self.daw.status("Sávokra bontás a feladatsorban…")

    def request_separation(self, path: str, ctx: dict):
        self.job_counter += 1
        self.worker.add({"id": self.job_counter, "kind": "sep", "audio": path, "ctx": ctx,
                         **({"fail_kind": "remake_sep"} if ctx.get("remake") else {})})
        self._update_queue_label()
        if not self.server_ok:
            self.log("Figyelem: a szerver nem fut - a sávbontás várakozik.")

    def request_midi(self, wav: str, instruments: str, name: str):
        self.job_counter += 1
        self.worker.add({"id": self.job_counter, "kind": "midi", "audio": wav, "instruments": instruments, "name": name})
        self._update_queue_label()

    def _sep_done(self, stems: dict, ctx: dict):
        self._update_queue_label()
        if ctx.get("remake"):
            if self.remake:
                self.remake.on_stems(stems)
            return
        if self.daw:
            if ctx.get("title") and not self.daw.project["tracks"]:
                self.daw.project["name"] = ctx["title"]
            self.daw.on_separated(stems, ctx)

    def _midi_done(self, path: str):
        self._update_queue_label()
        if self.daw:
            self.daw.status(f"MIDI kész: {path}")
        box = QMessageBox(self)
        box.setWindowTitle("MIDI kész")
        box.setText(f"Elkészült a MIDI kotta:\n{path}\n\nBármely kottaszerkesztővel vagy DAW-val megnyitható "
                    "(pl. MuseScore).")
        b = box.addButton("Mappa megnyitása", QMessageBox.ActionRole)
        b_score = box.addButton("🎼 Megnyitás kottaként", QMessageBox.ActionRole) if self.score else None
        box.addButton("OK", QMessageBox.AcceptRole)
        box.exec()
        if box.clickedButton() is b:
            open_in_explorer(Path(path))
        elif b_score is not None and box.clickedButton() is b_score:
            self.score.open_midi(path)
            self.tabs.setCurrentWidget(self.score)

    def _task_failed(self, kind: str, msg: str):
        self._update_queue_label()
        if kind.startswith("remake_") or kind == "asr":
            if self.remake:
                self.remake.on_failed(kind, msg)
            return
        title = "Sávokra bontás" if kind == "sep" else "MIDI"
        if self.daw:
            self.daw.status(f"{title} hiba: {msg}")
        QMessageBox.warning(self, title, msg)

    def _toggle_fav(self):
        t = self._sel_track()
        if t:
            self.db.update(t["id"], fav=0 if t.get("fav") else 1)
            self.refresh_library()

    def _edit_note(self):
        t = self._sel_track()
        if not t:
            return
        txt, ok = QInputDialog.getMultiLineText(self, "Megjegyzés", "Megjegyzés:", t.get("note") or "")
        if ok:
            self.db.update(t["id"], note=txt)
            self.refresh_library()

    def _rename_track(self):
        t = self._sel_track()
        if not t:
            return
        txt, ok = QInputDialog.getText(self, "Átnevezés", "Új cím:", text=t.get("title") or "")
        if ok and txt.strip():
            self.db.update(t["id"], title=txt.strip())
            self.refresh_library()

    def _reuse_track(self):
        t = self._sel_track()
        if not t:
            return
        p = json.loads(t.get("params") or "{}")
        if p.get("form"):
            self.set_form(p["form"])
            if p["options"].get("abc"):
                self.ed_abc.setPlainText(p["options"]["abc"])
            self.show_page("gen")
            self.log(f"Beállítások visszatöltve: {t.get('title')}")

    def _reuse_seed(self):
        t = self._sel_track()
        if t:
            self.chk_rand.setChecked(False)
            self.sp_seed.setValue(int(t["seed"]))
            self.show_page("gen")

    def _track_abc(self):
        t = self._sel_track()
        if t and t.get("abc_path") and Path(t["abc_path"]).is_file():
            self.ed_abc.setPlainText(Path(t["abc_path"]).read_text(encoding="utf-8"))
            self.show_page("abc")
        else:
            QMessageBox.information(self, "ABC", "Ehhez a számhoz nincs mentett ABC terv (cot=off?).")

    def _track_folder(self):
        t = self._sel_track()
        if t:
            open_in_explorer(Path(t["wav_path"]))

    def _export_mp3(self):
        t = self._sel_track()
        if not t:
            return
        ff = find_ffmpeg(self.cfg)
        if not ff:
            QMessageBox.warning(self, "MP3", "ffmpeg nem található. Add meg a Beállításokban.")
            return
        src = Path(t["wav_path"])
        dst, _ = QFileDialog.getSaveFileName(self, "MP3 mentése", str(src.with_suffix(".mp3")), "MP3 (*.mp3)")
        if not dst:
            return
        p = subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-b:a", "320k",
                            "-metadata", f"title={t.get('title')}", "-metadata", f"comment={t.get('style')}", dst],
                           capture_output=True, creationflags=NO_WINDOW)
        if p.returncode == 0:
            self.log(f"MP3 kész: {dst}")
        else:
            QMessageBox.warning(self, "MP3", p.stderr.decode("utf-8", "replace")[:500])

    def _delete_track(self):
        t = self._sel_track()
        if not t:
            return
        box = QMessageBox(self)
        box.setWindowTitle("Törlés")
        box.setText(f"'{t.get('title')}' törlése?")
        b_list = box.addButton("Csak a listából", QMessageBox.AcceptRole)
        b_all = box.addButton("Listából + fájlok", QMessageBox.DestructiveRole)
        box.addButton("Mégse", QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() not in (b_list, b_all):
            return
        if box.clickedButton() is b_all:
            if self.player and self.player.source().toLocalFile() == t["wav_path"]:
                self.player.stop()
                self.player.setSource(QUrl())
            for k in ("wav_path", "abc_path"):
                if t.get(k):
                    Path(t[k]).unlink(missing_ok=True)
            Path(t["wav_path"]).with_suffix(".mp3").unlink(missing_ok=True)
        self.db.delete(t["id"])
        self.refresh_library()

    # =================================================== TELEPÍTÉS FÜL
    def _build_setup_tab(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        top = QHBoxLayout()

        g_eng = QGroupBox(f"1. YuE2 motor (audio.cpp {ENGINE_VERSION}, Windows Vulkan)")
        el = QVBoxLayout(g_eng)
        self.lbl_engine = QLabel("?")
        el.addWidget(self.lbl_engine)
        info = QLabel("Kész, lefordított szerver (~270 MB + CUDA runtime). Fordítás, CUDA Toolkit nem kell, "
                      "csak friss NVIDIA driver (RTX 20-as vagy újabb).")
        info.setWordWrap(True)
        info.setObjectName("dim")
        el.addWidget(info)
        self.btn_engine = QPushButton("Motor letöltése")
        self.btn_engine.clicked.connect(self.download_engine)
        el.addWidget(self.btn_engine)
        el.addStretch()
        top.addWidget(g_eng, 1)

        g_mod = QGroupBox("2. Modellek (Hugging Face)")
        ml = QVBoxLayout(g_mod)
        self.pkg_checks = {}
        for pid, p in PACKAGES.items():
            cb = QCheckBox(p["label"])
            cb.setChecked(p["default"])
            self.pkg_checks[pid] = cb
            ml.addWidget(cb)
        r = QHBoxLayout()
        self.btn_models = QPushButton("Kijelöltek letöltése")
        self.btn_models.clicked.connect(self.download_models)
        r.addWidget(self.btn_models)
        b = QPushButton("Mappa megnyitása")
        b.clicked.connect(lambda: (Path(self.cfg["models_dir"]).mkdir(parents=True, exist_ok=True),
                                   open_in_explorer(Path(self.cfg["models_dir"]))))
        r.addWidget(b)
        r.addStretch()
        ml.addLayout(r)
        top.addWidget(g_mod, 1)
        l.addLayout(top)

        g_dl = QGroupBox("Letöltés állapota")
        dl = QVBoxLayout(g_dl)
        self.lbl_dl = QLabel("—")
        dl.addWidget(self.lbl_dl)
        self.pb_file = QProgressBar()
        self.pb_all = QProgressBar()
        dl.addWidget(QLabel("Aktuális fájl:"))
        dl.addWidget(self.pb_file)
        dl.addWidget(QLabel("Összesen:"))
        dl.addWidget(self.pb_all)
        self.btn_dl_stop = QPushButton("Letöltés megállítása")
        self.btn_dl_stop.setEnabled(False)
        self.btn_dl_stop.clicked.connect(lambda: self.downloader and self.downloader.stop())
        dl.addWidget(self.btn_dl_stop)
        l.addWidget(g_dl)

        g_srv = QGroupBox("3. Szerver")
        sl = QVBoxLayout(g_srv)
        r = QHBoxLayout()
        self.lbl_srv = QLabel("● leállítva")
        self.lbl_srv.setObjectName("status_bad")
        r.addWidget(self.lbl_srv)
        r.addStretch()
        self.btn_start = QPushButton("▶ Indítás")
        self.btn_start.setObjectName("primary")
        self.btn_start.clicked.connect(self.start_server)
        self.btn_stop_srv = QPushButton("■ Leállítás")
        self.btn_stop_srv.clicked.connect(self.stop_server)
        self.btn_restart = QPushButton("⟲ Újraindítás")
        self.btn_restart.clicked.connect(self.restart_server)
        b_unload = QPushButton("Modellek kiürítése (VRAM)")
        b_unload.clicked.connect(self._unload_all)
        b_gpu = QPushButton("GPU lista")
        b_gpu.clicked.connect(self._list_devices)
        for b in (self.btn_start, self.btn_stop_srv, self.btn_restart, b_unload, b_gpu):
            r.addWidget(b)
        sl.addLayout(r)
        self.srv_log = QPlainTextEdit()
        self.srv_log.setReadOnly(True)
        self.srv_log.setMaximumBlockCount(5000)
        sl.addWidget(self.srv_log)
        l.addWidget(g_srv, 1)
        return w

    def engine_exe(self) -> Path | None:
        d = Path(self.cfg["engine_dir"])
        name = "audiocpp_server.exe" if IS_WINDOWS else "audiocpp_server"
        if (d / name).is_file():
            return d / name
        if d.is_dir():
            for p in d.rglob(name):
                return p
        return None

    def refresh_install_state(self):
        exe = self.engine_exe()
        self.lbl_engine.setText(f"✓ Telepítve: {exe}" if exe else "✗ Nincs telepítve")
        self.lbl_engine.setObjectName("status_ok" if exe else "status_bad")
        self.lbl_engine.setStyleSheet("")
        self.lbl_engine.style().unpolish(self.lbl_engine)
        self.lbl_engine.style().polish(self.lbl_engine)
        mroot = Path(self.cfg["models_dir"])
        for pid, p in PACKAGES.items():
            ok = (mroot / p["check"]).is_file()
            cb = self.pkg_checks[pid]
            cb.setText(("✓ " if ok else "") + p["label"])
            if ok:
                cb.setChecked(False)
        ydir = mroot / YUE2_DIRNAME
        mains = sorted(p.name for p in ydir.glob("yue2-3b-*.gguf")) if ydir.is_dir() else []
        vaes = sorted(p.name for p in ydir.glob("yue2-vae-*.gguf")) if ydir.is_dir() else []
        loras = sorted(p.name for p in ydir.glob("*.safetensors")) if ydir.is_dir() else []
        for cb, items, pref in ((self.cb_main, mains, "yue2-3b-q8_0.gguf"), (self.cb_vae, vaes, "yue2-vae-f16.gguf")):
            cur = cb.currentText() or pref
            cb.clear()
            cb.addItems(items)
            i = cb.findText(cur)
            cb.setCurrentIndex(i if i >= 0 else 0)
        for cb in (self.cb_ar_lora, self.cb_nar_lora):
            cur = cb.currentText()
            cb.clear()
            cb.addItem("(nincs)")
            cb.addItems(loras)
            i = cb.findText(cur)
            cb.setCurrentIndex(i if i >= 0 else 0)

    def _start_download(self, jobs, extract_to=None):
        if self.downloader and self.downloader.isRunning():
            QMessageBox.information(self, APP_NAME, "Már fut egy letöltés.")
            return
        self.downloader = Downloader(jobs, extract_to)
        self.downloader.progress.connect(self._dl_progress)
        self.downloader.overall.connect(lambda i, n: (self.pb_all.setRange(0, max(n, 1)), self.pb_all.setValue(i)))
        self.downloader.message.connect(lambda m: (self.lbl_dl.setText(m), self.log(m)))
        self.downloader.done.connect(self._dl_done)
        self.btn_engine.setEnabled(False)
        self.btn_models.setEnabled(False)
        self.btn_dl_stop.setEnabled(True)
        self.downloader.start()

    def _dl_progress(self, name: str, done: int, total: int):
        if total:
            self.pb_file.setRange(0, 1000)
            self.pb_file.setValue(int(done * 1000 / total))
            self.pb_file.setFormat(f"{name}: {done / 1e9:.2f} / {total / 1e9:.2f} GB")
        else:
            self.pb_file.setRange(0, 0)
            self.pb_file.setFormat(f"{name}: {done / 1e6:.0f} MB")

    def _dl_done(self, ok: bool, msg: str):
        self.lbl_dl.setText(msg)
        self.log(msg)
        self.btn_engine.setEnabled(True)
        self.btn_models.setEnabled(True)
        self.btn_dl_stop.setEnabled(False)
        tmp = Path(self.cfg["engine_dir"]) / "_letoltes"
        if ok and tmp.is_dir() and self.engine_exe():
            shutil.rmtree(tmp, ignore_errors=True)
        self.refresh_install_state()
        if not ok:
            QMessageBox.warning(self, "Letöltés", msg)

    def download_engine(self):
        if not IS_WINDOWS:
            QMessageBox.information(self, APP_NAME, "Az automatikus motorletöltés csak Windowsra van beállítva.")
            return
        ed = Path(self.cfg["engine_dir"])
        tmp = ed / "_letoltes"
        jobs = [(url, tmp / url.rsplit("/", 1)[-1], sha) for url, sha in ENGINE_FILES]
        self._start_download(jobs, extract_to=ed)

    def download_models(self):
        mroot = Path(self.cfg["models_dir"])
        jobs, seen = [], set()
        for pid, cb in self.pkg_checks.items():
            if cb.isChecked():
                for url, rel in PACKAGES[pid]["files"]:
                    if rel not in seen:
                        seen.add(rel)
                        jobs.append((url, mroot / rel, None))
        if not jobs:
            QMessageBox.information(self, APP_NAME, "Nincs kijelölt (hiányzó) csomag.")
            return
        self._start_download(jobs)

    # --- szerver
    def show_page(self, name: str):
        w = self.pages.get(name)
        if w is not None:
            self.tabs.setCurrentWidget(w)

    def _maybe_autostart(self):
        if self.server_ok:
            self.server_external = True
            self.log(f"Már fut egy YuE2/audio.cpp szerver a {self.cfg['port']}-es porton - azt használom.")
            return
        if self.cfg.get("auto_start_server") and self.engine_exe():
            self.start_server()
        elif not self.engine_exe():
            self.show_page("setup")
            self.log("Első indítás: töltsd le a motort és a modelleket a 'Telepítés & Szerver' fülön.")

    def start_server(self):
        if self.server.state() != QProcess.NotRunning:
            return
        exe = self.engine_exe()
        if not exe:
            QMessageBox.warning(self, APP_NAME, "A motor nincs telepítve.")
            return
        if self.server_ok:
            self.server_external = True
            self.log("A porton már fut egy szerver - nem indítok újat.")
            return
        env = QProcessEnvironment.systemEnvironment()
        if self.cfg.get("gpu_device", "").strip():
            env.insert("CUDA_VISIBLE_DEVICES", self.cfg["gpu_device"].strip())
        self.server.setProcessEnvironment(env)
        ed = Path(self.cfg["engine_dir"])
        self.server.setWorkingDirectory(str(ed if (ed / "model_specs").is_dir() else exe.parent))
        args = ["--ui", "--ui-management", "--backend", self.cfg.get("backend") or "vulkan",
                "--host", "127.0.0.1", "--port", str(self.cfg["port"])]
        self.log(f"Szerver indítása: {exe.name} {' '.join(args)}")
        self.worker.reset_sessions()
        self.server_external = False
        self.server.start(str(exe), args)
        self.lbl_srv.setText("● indul…")
        self._restyle(self.lbl_srv, "status_mid")

    def stop_server(self):
        if self.server.state() != QProcess.NotRunning:
            self.server.terminate()
            if not self.server.waitForFinished(5000):
                self.server.kill()
                self.server.waitForFinished(3000)
        elif self.server_external:
            self.log("A szerver kívülről fut (pl. Remiqora) - azt onnan állítsd le.")
        self.worker.reset_sessions()

    def restart_server(self):
        self.stop_server()
        QTimer.singleShot(1500, self.start_server)

    def _server_output(self):
        data = bytes(self.server.readAllStandardOutput()).decode("utf-8", "replace")
        for line in data.splitlines():
            if line.strip():
                self.srv_log.appendPlainText(line)

    def _server_finished(self, code, status):
        self.log(f"Szerver leállt (kód: {code}).")
        self.worker.reset_sessions()

    def _restyle(self, w: QLabel, name: str):
        w.setObjectName(name)
        w.style().unpolish(w)
        w.style().polish(w)

    def _on_health(self, ok: bool):
        self.server_ok = ok
        running = self.server.state() != QProcess.NotRunning
        if ok:
            txt = "● fut" + (" (külső)" if self.server_external and not running else "")
            self.lbl_srv.setText(txt + f"  -  http://127.0.0.1:{self.cfg['port']}")
            self._restyle(self.lbl_srv, "status_ok")
            self.sb_server.setText("Szerver: ✓ fut")
        elif running:
            self.lbl_srv.setText("● indul…")
            self._restyle(self.lbl_srv, "status_mid")
            self.sb_server.setText("Szerver: indul…")
        else:
            self.lbl_srv.setText("● leállítva")
            self._restyle(self.lbl_srv, "status_bad")
            self.sb_server.setText("Szerver: ✗ leállítva")

    def _unload_all(self):
        def job():
            api = Yue2Api(int(self.cfg["port"]))
            for mid in api.loaded_ids():
                api.unload(mid)
        threading.Thread(target=job, daemon=True).start()
        self.worker.reset_sessions()
        self.log("Modellek kiürítése kérve.")

    def _list_devices(self):
        exe = self.engine_exe()
        if not exe:
            return
        try:
            p = subprocess.run([str(exe), "--list-devices"], capture_output=True, text=True, timeout=30,
                               cwd=str(exe.parent), creationflags=NO_WINDOW)
            self.srv_log.appendPlainText((p.stdout or "") + (p.stderr or ""))
        except Exception as e:
            self.srv_log.appendPlainText(f"Hiba: {e}")

    # =================================================== BEÁLLÍTÁSOK FÜL
    def _build_settings_tab(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        g = QGroupBox("Útvonalak és szerver")
        f = QFormLayout(g)
        self.set_widgets = {}

        def path_row(key, label, is_file=False):
            ed = QLineEdit(str(self.cfg.get(key, "")))
            b = QPushButton("…")
            b.setMaximumWidth(40)

            def pick():
                if is_file:
                    p, _ = QFileDialog.getOpenFileName(self, label, ed.text() or str(APP_DIR))
                else:
                    p = QFileDialog.getExistingDirectory(self, label, ed.text() or str(APP_DIR))
                if p:
                    ed.setText(p)
            b.clicked.connect(pick)
            r = QHBoxLayout()
            r.addWidget(ed, 1)
            r.addWidget(b)
            f.addRow(label + ":", r)
            self.set_widgets[key] = ed

        path_row("engine_dir", "Motor mappa")
        path_row("models_dir", "Modellek mappa")
        path_row("output_dir", "Kimeneti mappa")
        path_row("ffmpeg", "ffmpeg.exe (üres = PATH)", True)
        self.sp_port = QSpinBox()
        self.sp_port.setRange(1024, 65535)
        self.sp_port.setValue(int(self.cfg["port"]))
        f.addRow("Port:", self.sp_port)
        self.ed_gpu = QLineEdit(self.cfg.get("gpu_device", "0"))
        f.addRow("CUDA_VISIBLE_DEVICES:", self.ed_gpu)
        self.chk_auto = QCheckBox("Szerver automatikus indítása az app indulásakor")
        self.chk_auto.setChecked(bool(self.cfg.get("auto_start_server")))
        f.addRow("", self.chk_auto)
        self.chk_unload = QCheckBox("Másik modell kiürítése váltáskor (YuE2 ↔ SheetSage2) - 16 GB VRAM-hoz ajánlott")
        self.chk_unload.setChecked(bool(self.cfg.get("unload_other_model", True)))
        f.addRow("", self.chk_unload)
        self.chk_mp3 = QCheckBox("Minden generálás után MP3 is (ffmpeg)")
        self.chk_mp3.setChecked(bool(self.cfg.get("auto_mp3")))
        f.addRow("", self.chk_mp3)
        self.cb_ui_lang = QComboBox()
        for code, name in UI_LANGS:
            self.cb_ui_lang.addItem(name, code)
        i_ = self.cb_ui_lang.findData(self.cfg.get("ui_lang", "hu"))
        self.cb_ui_lang.setCurrentIndex(max(0, i_))
        self.cb_ui_lang.setToolTip(tip_html("Felület nyelve|A program feliratainak, súgó buborékainak, üzeneteinek, a "
                                            "Súgó fülnek és a Kalauznak a nyelve. Az új nyelv a program "
                                            "újraindítása után lép életbe.\nA fordítások a 'lang' mappában vannak, "
                                            "szövegszerkesztővel javíthatók."))
        self.cb_ui_lang.activated.connect(self._ui_lang_changed)
        f.addRow("Felület nyelve:", self.cb_ui_lang)
        self.chk_tips = QCheckBox("Súgó buborékok megjelenítése (egérmutató az elem fölött)")
        self.chk_tips.setChecked(bool(self.cfg.get("show_tooltips", True)))
        self.chk_tips.toggled.connect(self._tips_toggled)
        f.addRow("", self.chk_tips)
        l.addWidget(g)

        g2 = QGroupBox("Ollama segéd (opcionális, helyi LLM)")
        f2 = QFormLayout(g2)
        self.ed_oll_url = QLineEdit(self.cfg.get("ollama_url", ""))
        f2.addRow("URL:", self.ed_oll_url)
        r = QHBoxLayout()
        self.cb_oll_model = QComboBox()
        self.cb_oll_model.setEditable(True)
        self.cb_oll_model.setCurrentText(self.cfg.get("ollama_model", ""))
        b = QPushButton("Modellek lekérése")
        b.clicked.connect(self._fetch_ollama_models)
        r.addWidget(self.cb_oll_model, 1)
        r.addWidget(b)
        f2.addRow("Modell:", r)
        l.addWidget(g2)

        r = QHBoxLayout()
        b = QPushButton("💾 Mentés")
        b.setObjectName("primary")
        b.clicked.connect(self._save_settings_ui)
        r.addWidget(b)
        b = QPushButton("Remiqora telepítés keresése")
        b.setToolTip("Ha a Remiqora már letöltötte a YuE2 motort és modelleket, azokat használja")
        b.clicked.connect(self._use_remiqora)
        r.addWidget(b)
        b = QPushButton("App mappa megnyitása")
        b.clicked.connect(lambda: open_in_explorer(APP_DIR))
        r.addWidget(b)
        r.addStretch()
        l.addLayout(r)
        l.addStretch()
        return w

    def _use_remiqora(self):
        rq = find_remiqora_yue2()
        if not rq:
            QMessageBox.information(self, APP_NAME, "Nem találtam Remiqora által telepített YuE2-t.")
            return
        self.set_widgets["engine_dir"].setText(str(rq))
        self.set_widgets["models_dir"].setText(str(rq / "models"))
        self._save_settings_ui()

    def _fetch_ollama_models(self):
        ms = ollama_models(self.ed_oll_url.text())
        cur = self.cb_oll_model.currentText()
        self.cb_oll_model.clear()
        self.cb_oll_model.addItems(ms)
        self.cb_oll_model.setCurrentText(cur if cur in ms or not ms else ms[0])
        if not ms:
            QMessageBox.information(self, "Ollama", "Nem érhető el az Ollama, vagy nincs modell.")

    def _ui_lang_changed(self, *_):
        code = self.cb_ui_lang.currentData()
        if code == self.cfg.get("ui_lang", "hu"):
            return
        if code != "hu" and not (LANG_DIR / f"{code}.json").is_file() and not self._download_lang(code):
            QMessageBox.warning(self, "Felület nyelve", f"Hiányzik a nyelvi fájl: lang/{code}.json\n\n"
                                f"Letölthető innen: {LANG_URL}{code}.json\n"
                                f"Tedd ide: {LANG_DIR}")
            self.cb_ui_lang.setCurrentIndex(max(0, self.cb_ui_lang.findData(self.cfg.get("ui_lang", "hu"))))
            return
        self.cfg["ui_lang"] = code
        save_settings(self.cfg)
        if QMessageBox.question(self, "Felület nyelve", "Az új nyelv újraindítás után lép életbe. "
                                "Újraindítsam most a programot? (A munkád megmarad.)") == QMessageBox.Yes:
            self._restart_app()

    def _download_lang(self, code: str) -> bool:
        """Hiányzó nyelvi fájl letöltése a GitHubról (a program mellé, a lang mappába)."""
        if QMessageBox.question(self, "Felület nyelve", f"A(z) lang/{code}.json nyelvi fájl nincs meg a gépen.\n"
                                "Letöltsem most a GitHubról? (kb. 150 KB)") != QMessageBox.Yes:
            return False
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            req = urllib.request.Request(LANG_URL + f"{code}.json", headers={"User-Agent": "YuE2Studio"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            json.loads(data.decode("utf-8"))          # ellenőrzés: érvényes JSON
            LANG_DIR.mkdir(parents=True, exist_ok=True)
            (LANG_DIR / f"{code}.json").write_bytes(data)
            return True
        except Exception as e:
            QMessageBox.warning(self, "Felület nyelve", f"Nem sikerült letölteni: {e}")
            return False
        finally:
            QApplication.restoreOverrideCursor()

    def _restart_app(self):
        self._restarting = True
        QProcess.startDetached(sys.executable, [str(Path(__file__).resolve())] + sys.argv[1:])
        self.close()

    def _save_settings_ui(self):
        for k, ed in self.set_widgets.items():
            self.cfg[k] = ed.text().strip()
        old_port = self.cfg.get("port")
        self.cfg["port"] = self.sp_port.value()
        self.poller.port = self.cfg["port"]
        self.cfg["gpu_device"] = self.ed_gpu.text().strip()
        self.cfg["auto_start_server"] = self.chk_auto.isChecked()
        self.cfg["unload_other_model"] = self.chk_unload.isChecked()
        self.cfg["auto_mp3"] = self.chk_mp3.isChecked()
        self.cfg["show_tooltips"] = self.chk_tips.isChecked()
        self.cfg["ollama_url"] = self.ed_oll_url.text().strip()
        self.cfg["ollama_model"] = self.cb_oll_model.currentText().strip()
        save_settings(self.cfg)
        self.refresh_install_state()
        msg = "Beállítások mentve."
        if old_port != self.cfg["port"]:
            msg += " A port változás a szerver újraindítása után érvényes."
        self.log(msg)
        QMessageBox.information(self, APP_NAME, msg)

    # ===================================================== SÚGÓ BUBORÉKOK
    def _tips_toggled(self, on: bool):
        if TOOLTIP_FILTER is not None:
            TOOLTIP_FILTER.enabled = on
        self.cfg["show_tooltips"] = on
        save_settings(self.cfg)

    def _apply_tooltips(self):
        # fülek
        for name, t in TAB_TIPS.items():
            w = self.pages.get(name)
            if w is not None and self.tabs.indexOf(w) >= 0:
                self.tabs.setTabToolTip(self.tabs.indexOf(w), tip_html(t))
        # nevesített elemek
        for key, text in WIDGET_TIPS.items():
            if "." in key:
                attr, sub = key.split(".", 1)
                obj = (getattr(self, attr, None) or {}).get(sub)
            else:
                obj = getattr(self, key, None)
            if obj is not None:
                obj.setToolTip(tip_html(text))
        # mintavételezési és arena mezők
        labels = {k: lbl for k, lbl, *_ in SAMPLING}
        for group, widgets, defaults in (("sem", self.opt_sem, SEM_DEFAULTS), ("abc", self.opt_abc, ABC_DEFAULTS)):
            for k, w in widgets.items():
                extra = SEM_MAX_EXTRA if (group == "sem" and k == "max_tokens") else ""
                w.setToolTip(tip_html(f"{labels[k]}|{SAMPLING_EXTRA[group]}\n{SAMPLING_TIPS[k]}{extra}\n"
                                      f"Alapérték: {defaults[k]}. A pipával kapcsolod be a saját értéket."))
        for k, lbl, default in ARENAS:
            self.opt_arena[k].setToolTip(tip_html(f"{lbl}|{ARENA_TIP}\nAlapérték: {default} MB."))
        # gombok felirat szerint, fülenként
        for name, mapping in BUTTON_TIPS.items():
            page = self.pages.get(name)
            if page is None:
                continue
            for b in page.findChildren(QPushButton):
                t = mapping.get(b.text())
                if t:
                    b.setToolTip(tip_html(t))
        for tb in self.findChildren(QToolButton):
            if tb.objectName() == "collapse":
                tb.setToolTip(tip_html("Haladó beállítások|Kattintásra kinyílik: guidance, lépésszám, "
                                       "mintavételezés, LoRA-k, GPU memória. Pipa nélkül mindenhol a modell "
                                       "alapértéke érvényes - kezdésnek nem kell hozzányúlni."))
        # táblázat fejlécek
        for attr, heads in HEADER_TIPS.items():
            tbl = getattr(self, attr)
            for c, h in enumerate(heads):
                it = tbl.horizontalHeaderItem(c)
                if it:
                    it.setToolTip(h)
        # feliratok (Label) a mellettük lévő mező súgóját kapják
        for lay in self.findChildren(QLayout):
            if isinstance(lay, QFormLayout):
                for row in range(lay.rowCount()):
                    li = lay.itemAt(row, QFormLayout.LabelRole)
                    fi = lay.itemAt(row, QFormLayout.FieldRole)
                    if li and li.widget() and fi and not li.widget().toolTip():
                        tt = self._item_tip(fi)
                        if tt:
                            li.widget().setToolTip(tt)
            for i in range(lay.count() - 1):
                w = lay.itemAt(i).widget()
                if isinstance(w, QLabel) and not w.toolTip():
                    tt = self._item_tip(lay.itemAt(i + 1))
                    if tt:
                        w.setToolTip(tt)
        if TOOLTIP_FILTER is not None:
            TOOLTIP_FILTER.enabled = bool(self.cfg.get("show_tooltips", True))
            install_tip_filter(self)

    @staticmethod
    def _item_tip(item) -> str:
        w = item.widget()
        if w is not None:
            return w.toolTip()
        lay = item.layout()
        if lay is not None:
            for j in range(lay.count()):
                ww = lay.itemAt(j).widget()
                if ww is not None and ww.toolTip():
                    return ww.toolTip()
        return ""

    # ========================================================= SÚGÓ FÜL
    def _build_help_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        g = QGroupBox("🧭 Kalauz - lépésről lépésre, kiemelve, hova kell kattintani")
        r = QHBoxLayout(g)
        for key, txt, tip in (("first", "Az első dal", "Az alapok: szöveg, stílus, generálás, meghallgatás."),
                              ("daw", "DAW – sávok", "Sávokra bontás, klipek, effektek, export."),
                              ("score", "Kotta", "Kotta megjelenítése, nyomtatás, PDF."),
                              ("remake", "Dal átdolgozása", "Egy meglévő dal dallamából új stílusú dal."),
                              ("project", "Projektek", "Projekt mentése, megnyitása, tiszta lap, könyvtár szűrő.")):
            b = QPushButton("▶ " + txt)
            b.setObjectName("primary")
            b.setToolTip(tip_html(f"{txt}|{tip}"))
            b.clicked.connect(lambda _=False, k=key: self.start_tour(k))
            r.addWidget(b)
        v.addWidget(g)
        tb = QTextBrowser()
        tb.setOpenExternalLinks(True)
        tb.setHtml(HELP_HTML)
        v.addWidget(tb, 1)
        return w

    # ============================================================ PROJEKT
    def _build_project_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("projbar")
        h = QHBoxLayout(bar)
        h.setContentsMargins(8, 4, 8, 0)
        self.lbl_proj = QLabel("")
        self.lbl_proj.setTextFormat(Qt.RichText)
        self.lbl_proj.setToolTip(tip_html("Projekt|Az éppen nyitott projekt neve. Egy projekt = az összes fül "
                                          "tartalma (Generálás, Átdolgozás, ABC-kotta, DAW, Kotta) és a hozzá "
                                          "generált dalok. A ● jel: van még nem mentett változás."))
        h.addWidget(self.lbl_proj)
        h.addStretch()
        for txt, fn, tip in (
            ("🆕 Új projekt", self.new_project, "Új projekt (Ctrl+N)|Minden fület kiürít, és új nevű projektet "
             "kezd. A Könyvtár ezután az új projekt dalait mutatja - a régiek nem vesznek el, a Projekt választóval "
             "visszanézhetők."),
            ("📂 Megnyitás…", self.open_project, "Projekt megnyitása (Ctrl+O)|Egy mentett projekt (.y2proj) "
             "visszatöltése: minden fül pontosan úgy áll vissza, ahogy mentetted."),
            ("💾 Mentés", self.save_project, "Projekt mentése (Ctrl+S)|Az összes fül tartalma egyetlen .y2proj fájlba. "
             "Első alkalommal rákérdez a névre."),
            ("Mentés másként…", lambda: self.save_project(True), "Mentés másként (Ctrl+Shift+S)|Mentés új néven / "
             "új helyre. A projekt dalai az új névvel mennek tovább."),
        ):
            b = QPushButton(txt)
            b.setToolTip(tip_html(tip))
            b.clicked.connect(fn)
            h.addWidget(b)
        self.btn_recent = QPushButton("Legutóbbiak ▾")
        self.btn_recent.setToolTip(tip_html("Legutóbbi projektek|Az utoljára mentett vagy megnyitott projektek - "
                                            "egy kattintással megnyithatók."))
        self.btn_recent.clicked.connect(self._recent_menu)
        h.addWidget(self.btn_recent)
        return bar

    def _lib_project_filter(self):
        d = self.cb_lib_proj.currentData() if hasattr(self, "cb_lib_proj") else "__cur__"
        if d == "__cur__" or d is None:
            return self.project_name
        if d == "__all__":
            return None
        return d

    def _refresh_lib_projects(self):
        cur = self.cb_lib_proj.currentData()
        self.cb_lib_proj.blockSignals(True)
        self.cb_lib_proj.clear()
        self.cb_lib_proj.addItem(f"Aktuális: {self.project_name}", "__cur__")
        self.cb_lib_proj.addItem("Minden dal", "__all__")
        for pn in self.db.projects():
            if pn != self.project_name:
                self.cb_lib_proj.addItem(pn or "(projekt nélküli régi dalok)", pn)
        i = self.cb_lib_proj.findData(cur)
        self.cb_lib_proj.setCurrentIndex(i if i >= 0 else 0)
        self.cb_lib_proj.setToolTip(tip_html("Projekt szűrő|Melyik projekt dalait mutassa a lista. Alapból csak az "
                                             "aktuális projektét - így nem keverednek a különböző munkák."))
        self.cb_lib_proj.blockSignals(False)

    def project_state(self) -> dict:
        return {"gen": self.get_form(),
                "abc": {"text": self.ed_abc.toPlainText(), "ref": self.ed_ref.text()},
                "remake": self.remake.get_state() if self.remake else {},
                "daw": self.daw.project if self.daw else {},
                "score": self.score.get_state() if self.score else {}}

    def apply_project_state(self, st: dict):
        if st.get("gen"):
            self.set_form(st["gen"])
        else:
            self.clear_generate()
        self.ed_abc.setPlainText((st.get("abc") or {}).get("text", ""))
        self.ed_ref.setText((st.get("abc") or {}).get("ref", ""))
        if self.remake:
            self.remake.set_state(st.get("remake") or {})
        if self.daw:
            self.daw.engine.pause()
            self.daw.engine.seek(0)
            self.daw.project = st["daw"] if st.get("daw") else new_project()
            self.daw.project_path = None
            self.daw.undo_stack.clear()
            self.daw.redo_stack.clear()
            self.daw.rebuild()
            if self.daw.project.get("tracks"):
                self.daw.zoom_fit()
            self.daw.autosave()
        if self.score:
            self.score.set_state(st.get("score") or {})

    def _state_key(self) -> str:
        try:
            return json.dumps(self.project_state(), sort_keys=True, ensure_ascii=False, default=str)
        except Exception:
            return ""

    def _mark_clean(self):
        self._proj_clean = self._state_key()
        self._update_project_label()

    def _project_dirty(self) -> bool:
        return bool(self._proj_clean) and self._state_key() != self._proj_clean

    def _update_project_label(self):
        if not hasattr(self, "lbl_proj"):
            return
        dirty = self._project_dirty()
        where = f" <span style='color:#80828e'>· {self.project_file.name}</span>" if self.project_file else \
            " <span style='color:#80828e'>· még nincs mentve</span>"
        self.lbl_proj.setText(f"📁 Projekt: <b>{html_escape(self.project_name)}</b>{where}"
                              + ("  <span style='color:#facc15'>● nem mentett változás</span>" if dirty else ""))
        self.setWindowTitle(f"{'● ' if dirty else ''}{self.project_name} – {APP_NAME} {APP_VERSION}")

    def _ask_save_first(self, what: str) -> bool:
        """True = mehet tovább."""
        if not self._project_dirty():
            return True
        r = QMessageBox.question(self, what, f"A(z) „{self.project_name}” projektben nem mentett változások vannak."
                                 "\n\nMented előtte?", QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if r == QMessageBox.Cancel:
            return False
        if r == QMessageBox.Save:
            return self.save_project()
        return True

    def _set_project(self, name: str, file: Path | None):
        self.project_name = name
        self.project_file = file
        self.cfg["project_name"] = name
        self.cfg["project_file"] = str(file) if file else ""
        if file:
            rec = [str(file)] + [x for x in self.cfg.get("recent_projects", []) if x != str(file)]
            self.cfg["recent_projects"] = rec[:10]
        save_settings(self.cfg)
        self._refresh_lib_projects()
        self.cb_lib_proj.setCurrentIndex(0)
        self.refresh_library()

    def new_project(self):
        if not self._ask_save_first("Új projekt"):
            return
        name, ok = QInputDialog.getText(self, "Új projekt", "Az új projekt neve:",
                                        text=f"Projekt {dt.datetime.now().strftime('%Y-%m-%d %H.%M')}")
        if not ok or not name.strip():
            return
        name = name.strip()
        if name in self.db.projects() and QMessageBox.question(
                self, "Új projekt", f"Már vannak dalok „{name}” néven. Hozzájuk csatlakozzon az új projekt?") != QMessageBox.Yes:
            return
        self.clear_all()
        self._set_project(name, None)
        self.log(f"Új projekt: {name}")
        self._mark_clean()

    def clear_all(self):
        self.clear_generate()
        self.clear_abc()
        if self.remake:
            self.remake.clear()
        if self.daw:
            self.daw.new_project(ask=False)
        if self.score:
            self.score.clear()

    def save_project(self, save_as: bool = False) -> bool:
        f = self.project_file
        if f is None or save_as:
            fn, _ = QFileDialog.getSaveFileName(self, "Projekt mentése", str(PROJECT_DIR / f"{safe_name(self.project_name, 60)}.y2proj"),
                                                "YuE2 Studio projekt (*.y2proj)")
            if not fn:
                return False
            f = Path(fn)
            if f.suffix.lower() != ".y2proj":
                f = f.with_suffix(".y2proj")
        new_name = f.stem if (save_as or self.project_file is None) else self.project_name
        if new_name != self.project_name:
            self.db.retag(self.project_name, new_name)
        st = self.project_state()
        tracks = self.db.all(project=self.project_name if new_name == self.project_name else new_name)
        data = {"app": APP_NAME, "format": 1, "name": new_name, "saved": dt.datetime.now().isoformat(timespec="seconds"),
                **st, "tracks": tracks}
        try:
            tmp = f.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, f)
        except Exception as e:
            QMessageBox.warning(self, "Projekt mentése", f"Nem sikerült menteni:\n{e}")
            return False
        self._set_project(new_name, f)
        self._mark_clean()
        self.log(f"Projekt mentve: {f}  ({len(tracks)} dal)")
        self.statusBar().showMessage(f"💾 Projekt mentve: {f.name}", 4000)
        return True

    def open_project(self, path: str | None = None) -> bool:
        if not self._ask_save_first("Projekt megnyitása"):
            return False
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Projekt megnyitása", str(PROJECT_DIR), "YuE2 Studio projekt (*.y2proj)")
            if not path:
                return False
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as e:
            QMessageBox.warning(self, "Projekt megnyitása", f"Nem sikerült beolvasni:\n{e}")
            return False
        name = data.get("name") or Path(path).stem
        # a projekt dalai: ha valamelyik hiányzik a könyvtárból (pl. törölted a listából), visszakerül
        back = 0
        for t in data.get("tracks") or []:
            wav = t.get("wav_path") or ""
            if not wav or not Path(wav).is_file() or self.db.by_wav(wav):
                continue
            rec = {k: t.get(k) for k in ("created", "title", "style", "lyrics", "seed", "cot", "model", "params",
                                         "wav_path", "abc_path", "duration", "gen_secs", "fav", "note")}
            rec["project"] = name
            self.db.add(**rec)
            back += 1
        self.apply_project_state(data)
        self._set_project(name, Path(path))
        self._mark_clean()
        self.log(f"Projekt megnyitva: {path}" + (f" ({back} dal visszakerült a Könyvtárba)" if back else ""))
        return True

    def _recent_menu(self):
        m = QMenu(self)
        rec = [x for x in self.cfg.get("recent_projects", []) if Path(x).is_file()]
        if not rec:
            a = m.addAction("(még nincs mentett projekt)")
            a.setEnabled(False)
        for x in rec:
            m.addAction(f"📁 {Path(x).stem}", lambda x=x: self.open_project(x))
        m.addSeparator()
        m.addAction("Projektek mappája…", lambda: open_in_explorer(PROJECT_DIR))
        m.exec(self.btn_recent.mapToGlobal(QPoint(0, self.btn_recent.height())))

    def _write_session(self):
        data = {"app": APP_NAME, "format": 1, "name": self.project_name,
                "file": str(self.project_file) if self.project_file else "", "clean": self._proj_clean,
                **self.project_state()}
        SESSION_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def _restore_session(self):
        self._refresh_lib_projects()
        self.refresh_library()
        if SESSION_FILE.is_file():
            try:
                data = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
                self.apply_project_state(data)
                self._proj_clean = data.get("clean", "")
            except Exception as e:
                self.log(f"A legutóbbi munkamenet nem tölthető vissza: {e}")
        if not self._proj_clean:
            self._mark_clean()
        self._update_project_label()

    # ---------------------------------------------------- tiszta lap (fülenként)
    def clear_generate(self, ask: bool = False):
        if ask and QMessageBox.question(self, "Tiszta lap", "Kiürítsem a Generálás fület?\n(A fő modell és a VAE "
                                        "megmarad; a kész dalok a Könyvtárban maradnak.)") != QMessageBox.Yes:
            return
        self.set_form({"lyrics": "", "style": "", "title": "", "cot": "full", "use_abc": False, "instrumental": False,
                       "random_seed": True, "batch": 1, "batch_mode": "inc", "wt_model": "native", "wt_vae": "native",
                       "ar_lora": "(nincs)", "nar_lora": "(nincs)", "ar_scale": 1.0, "nar_scale": 1.0,
                       "main": self.cb_main.currentText(), "vae": self.cb_vae.currentText()})
        for r in reversed(range(self.tbl_queue.rowCount())):
            it = self.tbl_queue.item(r, 3)
            if it and (it.text().startswith("✓") or it.text().startswith("✗")):
                self.tbl_queue.removeRow(r)
        self.job_rows = {}
        for r in range(self.tbl_queue.rowCount()):
            self.job_rows[int(self.tbl_queue.item(r, 0).text())] = r
        self.current_track = None
        self.last_abc = ""
        self.lbl_last.setText("Még nincs eredmény.")
        for b in (self.btn_play_last, self.btn_folder_last, self.btn_abc_last, self.btn_again):
            b.setEnabled(False)
        self.gen_log.clear()

    def clear_abc(self, ask: bool = False):
        if ask and self.ed_abc.toPlainText().strip() and QMessageBox.question(
                self, "Tiszta lap", "Kiürítsem az ABC-kotta fület?") != QMessageBox.Yes:
            return
        self.ed_abc.clear()
        self.ed_ref.clear()
        self.chk_use_abc.setChecked(False)

    def clear_library(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Könyvtár ürítése")
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel("<b>Mely dalokat?</b>"))
        n_cur = len(self.db.all(project=self.project_name))
        n_view = len(self.lib_ids)
        n_all = len(self.db.all())
        rb_cur = QRadioButton(f"Az aktuális projekt („{self.project_name}”) dalai – {n_cur} db")
        rb_view = QRadioButton(f"A listában most látható dalok – {n_view} db")
        rb_all = QRadioButton(f"Minden dal, minden projektből – {n_all} db")
        rb_cur.setChecked(True)
        for rb in (rb_cur, rb_view, rb_all):
            v.addWidget(rb)
        v.addSpacing(8)
        v.addWidget(QLabel("<b>Hogyan?</b>"))
        rb_list = QRadioButton("Csak a listából (a hangfájlok megmaradnak a lemezen)")
        rb_files = QRadioButton("A listából ÉS a lemezről is (WAV, MP3, ABC) – végleges, nem vonható vissza!")
        rb_list.setChecked(True)
        v.addWidget(rb_list)
        v.addWidget(rb_files)
        keep_fav = QCheckBox("A kedvencek (★) maradjanak meg")
        keep_fav.setChecked(True)
        v.addWidget(keep_fav)
        info = QLabel("Tipp: ha csak tisztább listát szeretnél, nem kell törölnöd semmit –\n"
                      "a Projekt választó alapból csak az aktuális projekt dalait mutatja.")
        info.setObjectName("dim")
        v.addWidget(info)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Ürítés")
        bb.button(QDialogButtonBox.Cancel).setText("Mégse")
        for b_ in bb.buttons():
            b_.setMinimumHeight(36)
            b_.setMinimumWidth(110)
        v.setSizeConstraint(QLayout.SetMinimumSize)
        dlg.setMinimumWidth(640)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        v.addWidget(bb)
        install_tip_filter(dlg)
        if dlg.exec() != QDialog.Accepted:
            return
        if rb_cur.isChecked():
            rows = self.db.all(project=self.project_name)
        elif rb_view.isChecked():
            rows = [t for t in (self.db.get(i) for i in self.lib_ids) if t]
        else:
            rows = self.db.all()
        if keep_fav.isChecked():
            rows = [t for t in rows if not t.get("fav")]
        if not rows:
            QMessageBox.information(self, "Könyvtár ürítése", "Nincs törlendő dal.")
            return
        files = rb_files.isChecked()
        if QMessageBox.question(self, "Könyvtár ürítése", f"{len(rows)} dal törlése a Könyvtárból"
                                + (" ÉS a lemezről (végleges!)" if files else " (a fájlok megmaradnak)") + ". Mehet?"
                                ) != QMessageBox.Yes:
            return
        if self.player:
            self.player.stop()
            self.player.setSource(QUrl())
        for t in rows:
            if files:
                for k in ("wav_path", "abc_path"):
                    if t.get(k):
                        Path(t[k]).unlink(missing_ok=True)
                if t.get("wav_path"):
                    Path(t["wav_path"]).with_suffix(".mp3").unlink(missing_ok=True)
            self.db.delete(t["id"])
        self.ed_search.clear()
        self._refresh_lib_projects()
        self.refresh_library()
        self.log(f"Könyvtár ürítve: {len(rows)} dal" + (" (fájlokkal együtt)" if files else ""))

    # ============================================================ bezárás
    def closeEvent(self, ev):
        # nincs kérdés: a teljes munkamenet elmentődik, és a következő indításkor visszajön
        try:
            self.cfg["last_form"] = self.get_form()
            self._write_session()
            save_settings(self.cfg)
        except Exception:
            pass
        if self.downloader and self.downloader.isRunning():
            self.downloader.stop()
            self.downloader.wait(3000)
        self.worker.quit_worker()
        self.poller.stop()
        if getattr(self, "daw", None):
            self.daw.shutdown()
        if self.server.state() != QProcess.NotRunning:
            self.stop_server()
        ev.accept()


HELP_HTML = """
<style> h2{color:#b9a8ff} h3{color:#d9ccff} td,th{padding:4px 10px;border-bottom:1px solid #2c2e37} code{color:#facc15} </style>
<h2>YuE2 Studio - gyors útmutató</h2>
<h3>Első indítás</h3>
<ol>
<li><b>Telepítés & Szerver</b> fül → <b>Motor letöltése</b> (~300 MB).</li>
<li>Pipáld ki a modelleket (alapból: Q8_0 fő modell, VAE F16, SheetSage2) → <b>Kijelöltek letöltése</b> (~6 GB).
A letöltés megszakítható és később folytatható.</li>
<li><b>▶ Indítás</b> - a szerver elindul (zöld = fut). Később automatikusan indul.</li>
<li><b>Generálás</b> fül → dalszöveg + stílus → <b>Sorba állít és generál</b>.</li>
</ol>
<h3>Dalszöveg</h3>
<p>Szakaszok szögletes zárójelben, külön sorban: <code>[Intro] [Verse] [Pre-Chorus] [Chorus] [Bridge] [Outro]</code>.
Rövid sorok, a refrén szó szerint ismételve. A YuE2 főleg angolul tanult - magyar szöveg működhet, de a kiejtés gyengébb.
Ének nélküli számhoz: <code>[instrumental]</code> + Instrumentális pipa.</p>
<h3>Stílus</h3>
<p>Vesszővel elválasztott <b>angol</b> tagek, ideálisan 5–8. Ajánlott sorrend:
nyelv, műfaj, hangulat, hangszerek, ének típusa, produkció. Pl.
<code>English, delta blues, smoky, slide guitar, harmonica, raspy male vocal, live recording</code></p>
<h3>COT mód</h3>
<table>
<tr><th>off</th><td>Nincs zenei terv - leggyorsabb, vázlatokhoz.</td></tr>
<tr><th>melody</th><td>Dallamtervet (ABC) készít, vagy a te ABC-dat követi.</td></tr>
<tr><th>full</th><td>Dallam + akkord terv - a legjobb szerkezet. Alapértelmezett.</td></tr>
</table>
<h3>ABC-kotta</h3>
<p>A generált terv minden melody/full számhoz elmentődik (.abc). Az ABC fülön szerkesztheted, ×2/×3-mal hosszabbíthatod
(a szólamok szinkronban maradnak), az ének szólamot törölheted (zümmögés ellen instrumentálisnál), vagy
<b>kinyerheted meglévő dalból</b> a SheetSage2-vel. Utána a Generálás fülön pipáld be: „ABC-kotta használata”.</p>
<h3>Fontos paraméterek</h3>
<table>
<tr><th>Guidance scale</th><td>Mennyire szigorúan kövesse a leírást. Túl magas → artefaktok.</td></tr>
<tr><th>Inferencia lépések</th><td>Hangszintézis lépései (alap 8). Több → lassabb, néha tisztább.</td></tr>
<tr><th>Semantic max tokens</th><td>A szám maximális hossza ezzel nő/csökken (alap 9000).</td></tr>
<tr><th>Temperature</th><td>Magasabb = kreatívabb, kiszámíthatatlanabb.</td></tr>
<tr><th>Ismétlés büntetés</th><td>Ha a zene „beragad”, növeld kicsit.</td></tr>
</table>
<p>A pipa nélküli paraméternél a modell saját alapértéke érvényes.</p>
<h3>Modellek és VRAM (16 GB)</h3>
<ul>
<li><b>Q8_0</b>: ajánlott, ~12 GB VRAM-mal fut kényelmesen.</li>
<li><b>Q4_0</b>: kevesebb memória, kicsit gyengébb minőség.</li>
<li><b>BF16</b>: legjobb minőség, LoRA-hoz ez a legpontosabb alap - több VRAM kell.</li>
<li>A szerver a YuE2 és a SheetSage2 között automatikusan ürít (Beállítások).</li>
</ul>
<h3>LoRA</h3>
<p><b>AR LoRA</b>: a tervezést/szemantikát módosítja (pl. instrumentális adapter, full COT kell hozzá).
<b>NAR LoRA</b>: a hangzást (pl. „realaudio” v4). Csak nem-ComfyUI (unfused) .safetensors fájl jó.
LoRA váltás a modell újratöltésével jár (automatikus).</p>
<h3>Feladatsor</h3>
<p>Több generálás is sorba állítható (Darab mező is). A futó feladat csak <b>Azonnali megszakítás</b>-sal
(szerver újraindítás) állítható le.</p>
<h3>Projektek és tiszta lap</h3>
<p>Fent a <b>projekt sáv</b>: egy projekt = minden fül tartalma + a hozzá generált dalok.
<b>🆕 Új projekt</b> (Ctrl+N) – minden fül kiürül, új név. <b>💾 Mentés</b> (Ctrl+S) – egy .y2proj fájlba.
<b>📂 Megnyitás</b> (Ctrl+O) / <b>Legutóbbiak</b> – pontosan visszaáll. A ● jel: nem mentett változás.
Kilépéskor semmi sem vész el: a következő indításkor ott folytatod, ahol abbahagytad.</p>
<p><b>🧹 Tiszta lap</b> minden munkafülön: csak azt a fület üríti ki. A <b>Könyvtár</b> alapból csak az aktuális
projekt dalait mutatja; a <b>🧹 Könyvtár ürítése…</b> törli a dalokat (csak a listából, vagy a lemezről is).</p>
<h3>Átdolgozás – meglévő dal új stílusban</h3>
<ol>
<li><b>Tallózás…</b> – MP3/WAV/FLAC… Hosszú dalnál pipáld be a <b>Csak egy részlet</b>-et (60–90 mp ideális).</li>
<li><b>⚡ 2–4. lépés egyben</b> – ének leválasztása (HTDemucs), dallam + akkordok (SheetSage2),
dalszöveg-felismerés (Qwen3-ASR).</li>
<li>Az <b>Időzített felismerés</b> (alapból be) a levegővételeknél bontja sorokra a szöveget – így jobban illik a
dallamhoz. Az <b>⏱ Időzítés…</b> gomb megmutatja, melyik sor hányadik másodpercnél / ütemnél szól az eredetiben.
Időpontot a szövegbe ne írj: a YuE2 nem érti, elénekelné.</li>
<li><b>Nézd át</b> a felismert szöveget – javítsd, írj újat, vagy fordíttasd le (Ollama).</li>
<li>Pipálj be <b>egy vagy több stílust</b>, válassz énekhangot és az ének nyelvét.</li>
<li><b>▶ Átdolgozás indítása</b> – stílusonként N változat. Jobb oldalt dupla kattintással meghallgatod.</li>
</ol>
<p>Kell hozzá: <b>SheetSage2</b>, <b>HTDemucs</b> és <b>Qwen3-ASR</b> csomag (Telepítés fül). Ha valamelyik hiányzik,
azt a lépést kihagyja (a szöveget kézzel is beírhatod). A YuE2 a <b>dallamot és a hangulatot</b> követi –
nem pontos másolat, hanem új feldolgozás.</p>
<h3>DAW – sávszerkesztő</h3>
<ul>
<li><b>Betöltés:</b> ＋ Hangfájl…, ＋ Könyvtárból, vagy a Könyvtárban a 🎛 DAW-ba gomb.</li>
<li><b>Sávokra bontás:</b> ✂ gomb (vagy jobb klikk egy klipen) – 4 sáv: ének, dob, basszus, egyéb. A HTDemucs csomag kell
hozzá (Telepítés fül). Az eredeti sávot a program elnémítja.</li>
<li><b>Klipek:</b> húzással mozgatod (másik sávra is), a széleket húzva vágod, a tetején lévő fehér pöttyökkel
beúsztatsz/kiúsztatsz. <b>S</b> = vágás a lejátszófejnél, <b>Del</b> = törlés, <b>Ctrl+D</b> = duplikálás,
dupla kattintás = tulajdonságok (tempó %, hangmagasság, visszafelé, hangerő).</li>
<li><b>Sávfejléc:</b> M = némítás, S = szóló, A = hangerő-automáció (görbe a sávon: kattintás = pont, jobb klikk = törlés),
FX = effektlánc (EQ, kompresszor, zengető, visszhang, kórus, torzító…, kész láncok énekre/dobra/basszusra).</li>
<li><b>Mester:</b> hangerő és Mester FX (pl. limiter) a teljes keverésre.</li>
<li><b>Vonalzó:</b> kattintás = ugrás, húzás = hurok tartomány (🔁 gombbal be).</li>
<li><b>MIDI:</b> jobb klikk egy klipen → MIDI kotta (MuScriptor csomag kell).</li>
<li><b>Export:</b> keverés WAV/MP3/FLAC fájlba (felvehető a Könyvtárba), vagy sávonként külön.</li>
<li>Visszavonás: Ctrl+Z, újra: Ctrl+Y. A projekt magától mentődik, a Mentés gombbal nevesítheted.</li>
</ul>
<h3>Kotta</h3>
<ul>
<li>Forrás: az ABC-kotta fül, a Könyvtárban kijelölt dal terve (🎼 Kotta gomb), vagy egy MIDI fájl (pl. a DAW-ban
készült: jobb klikk egy klipen → ♫ MIDI kotta).</li>
<li>MIDI-nél: <b>Dallam</b> (egy sor, énekhez) vagy <b>Zongora</b> (két sor) típus, ritmus rács, hangnem.</li>
<li>Transzponálás, méret, cím, szerző, szólam választás, akkordjelek ki/be.</li>
<li><b>🖨 Nyomtatás</b>, <b>PDF</b>, <b>MIDI</b> és <b>ABC</b> mentés, <b>▶ Lejátszás</b> a hangjegyek kiemelésével
(a hangszerhangokhoz internet kell), megnyitás <b>MuseScore</b>-ban.</li>
<li>A kottamegjelenítő (abcjs) első használatkor letöltődik.</li>
</ul>
<h3>Kalauz</h3>
<p>Lent a <b>🧭 Kalauz</b> gombbal vagy fent a gombokkal lépésről lépésre bemutatót indíthatsz: a sötét rétegen a
kiemelt elemre figyelj (rá is kattinthatsz). Lépés: Következő / → ; kilépés: Esc.</p>
<h3>Licenc</h3>
<p>YuE2 súlyok és a Mothersuperior LoRA-k: CC BY-NC 4.0 - nem kereskedelmi felhasználás. audio.cpp: MIT.</p>
"""


def _translate_after_init(*classes):
    """A később létrejövő ablakrészek (Kalauz, sávfejléc, párbeszédek) feliratait is lefordítjuk."""
    for cls in classes:
        orig = cls.__init__

        def init(self, *a, __orig=orig, **k):
            __orig(self, *a, **k)
            translate_tree(self)
        cls.__init__ = init


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_QSS)
    global TOOLTIP_FILTER
    TOOLTIP_FILTER = TooltipFilter()
    lang = load_settings().get("ui_lang", "hu")
    if lang != "hu":
        if load_language(lang):
            install_translation_hooks()
            _translate_after_init(TourOverlay, TrackHeader, FxDialog, ClipDialog)
        else:
            QMessageBox.warning(None, APP_NAME, f"A(z) '{lang}' nyelvi fájl hiányzik vagy hibás (lang mappa). "
                                                "A program magyarul indul.")
    try:
        w = MainWindow()
        translate_tree(w)
    except Exception:
        QMessageBox.critical(None, APP_NAME, traceback.format_exc())
        raise
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
