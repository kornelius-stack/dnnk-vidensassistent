#!/usr/bin/env python3
"""
Build search-index.json for DNNK Vidensassistent.

For each transcription:
  - Decodes title and category from filename
  - Matches against dnnk.dk event pages (title, description, date, speakers)
  - Finds YouTube URL via dnnk.dk category pages
  - Generates AI summary + keywords via Claude Haiku
  - Only processes NEW files (skips already-indexed ones)

PDF-dokumenter (filer med navnepræfikset "PDF_", lagt i samme mappe af
PDF-scraperen) behandles særskilt:
  - Titel/kategori/kilde-URL/dato parses fra filens header (=== DNNK PDF Dokument ===)
  - Ingen YouTube- eller dnnk.dk-event-matching
  - AI-resumé + keywords genereres som for webinarer, men med dokument-prompt
  - Entries får type="pdf" (webinarer får type="webinar"; manglende type
    betyder webinar af hensyn til bagudkompatibilitet)

Run:  python build_search_index.py
Env:  ANTHROPIC_API_KEY  (required)
      GITHUB_TOKEN        (optional but recommended to avoid rate limits)
"""

from __future__ import annotations

import json
import os
import re
import time
import requests
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
from urllib.parse import unquote, quote, urlparse

# anthropic og yt_dlp er kun nødvendige for behandling af NYE filer.
# Metadata-refresh af eksisterende entries skal kunne køre uden dem.
try:
    import anthropic
except ImportError:
    anthropic = None
try:
    import yt_dlp
except ImportError:
    yt_dlp = None

# Sættes til True når Anthropic-forbrugsloftet rammes — resten af kørslen
# springer AI-kald over i stedet for at fejle (jobbet skal lande grønt).
AI_BUDGET_EXHAUSTED = False

# ── Config ────────────────────────────────────────────────────────────────────

TRANSCRIPTIONS_REPO = "klimatilpasning/dnnk-transcriptor"
# Korpusset ligger i transcriptions/ i transcriptor-repoet; kan overstyres via env
BASE_FOLDER = os.environ.get("BASE_FOLDER", "transcriptions")

DNNK_CATEGORY_PAGES = [
    ("Godmorgen med DNNK", "https://www.dnnk.dk/god-morgen-med-dnnk/"),
    ("Tech Talk",          "https://www.dnnk.dk/tech-talks/"),
    ("Jura",               "https://www.dnnk.dk/jura-i-klimatilpasning/"),
    ("Masterclass",        "https://www.dnnk.dk/dnnk-masterclass/"),
    ("Konferencer",        "https://www.dnnk.dk/optagelser-fra-konferencer-og-temadage/"),
    # ("Øvrige", "https://www.dnnk.dk/ovrige-optagelser/") er fjernet:
    # siden giver 404 (juli 2026), og hverken /oevrige-optagelser/ eller
    # dnnk.dk's sitemap (wp-sitemap-posts-page-1.xml) har en afløser.
]

# Kategorier hvor videoerne ligger på UNDERSIDER (én side pr. event)
# i stedet for direkte på kategorisiden.
SUBPAGE_CATEGORIES = {"Masterclass"}

# External resource pages to scrape for cross-references
EXTERNAL_RESOURCE_PAGES = [
    # DNNK egen vidensbank — primær kilde
    ("DNNK Rapporter",         "https://www.dnnk.dk/vidensbank2/",                              "dnnk.dk"),
    ("DNNK Rapporter side 2",  "https://www.dnnk.dk/vidensbank2/page/2/",                       "dnnk.dk"),
    ("DNNK Rapporter side 3",  "https://www.dnnk.dk/vidensbank2/page/3/",                       "dnnk.dk"),
    ("DNNK Rapporter side 4",  "https://www.dnnk.dk/vidensbank2/page/4/",                       "dnnk.dk"),
    ("DNNK Rapporter side 5",  "https://www.dnnk.dk/vidensbank2/page/5/",                       "dnnk.dk"),
    ("DNNK Rapporter side 6",  "https://www.dnnk.dk/vidensbank2/page/6/",                       "dnnk.dk"),
    ("DNNK Horizon & LIFE",    "https://www.dnnk.dk/horizon-og-life/",                          "dnnk.dk"),
    ("DNNK Nyheder",           "https://www.dnnk.dk/nyheder/",                                  "dnnk.dk"),
    # Klimatilpasning.dk — sekundær
    ("Klimatilpasning.dk",     "https://www.klimatilpasning.dk/publikationer/",                 "klimatilpasning.dk"),
    ("Klimatilpasning vejl.",  "https://www.klimatilpasning.dk/kommuner-og-forsyning/proces-og-vejledning/", "klimatilpasning.dk"),
]

# URL-patterns og titler der ALDRIG må optræde som ressource
EXCLUDE_PATTERNS = [
    r"/privatlivspolitik",
    r"/kontakt",
    r"/cookies",
    r"/login",
    r"/medlem",
    r"/wp-admin",
    r"/wp-content",
    r"/wp-login",
    r"/feed",
    r"#",
    r"/page/\d+/?$",     # selve paginerings-links
    r"/\d+-\d+/?$",      # numerisk junk som /8293-2
]
EXCLUDE_TITLE_KEYWORDS = [
    "privatlivspolitik", "cookies", "kontakt os", "log ind",
    "tilmeld", "nyhedsbrev", "om os", "medlemskab",
]

def is_junk_resource(url: str, title: str) -> bool:
    """Filter out navigation/junk URLs and titles"""
    url_low = url.lower()
    title_low = title.lower()
    for pat in EXCLUDE_PATTERNS:
        if re.search(pat, url_low):
            return True
    for kw in EXCLUDE_TITLE_KEYWORDS:
        if kw in title_low:
            return True
    if len(title) < 15 or len(title) > 200:
        return True
    return False

def detect_resource_type(url: str, title: str) -> str:
    """Detect what type of resource this is for icon display"""
    u = url.lower(); t = title.lower()
    if any(k in u for k in ["/horizon", "/life", "interreg"]): return "🌐 EU-projekt"
    if any(k in t for k in ["håndbog", "haandbog", "guide", "vejledning"]): return "📋 Vejledning"
    if any(k in t for k in ["værktøj", "vaerktoj", "atlas", "kort"]):       return "🔧 Værktøj"
    if any(k in t for k in ["case", "kommune", "projekt"]):                 return "📍 Case"
    if any(k in t for k in ["lov", "bekendtg", "direktiv", "paragraf"]):    return "⚖️ Lovgivning"
    if any(k in t for k in ["rapport", "hvidbog", "white paper", "analyse"]): return "📊 Rapport"
    return "📄 Dokument"

HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; DNNK-indexer/1.0)"}

DNNK_YOUTUBE_CHANNEL = "https://www.youtube.com/@dnnk-detnationalenetvrkfor946/videos"

# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_youtube_id(url: str) -> str | None:
    if not url:
        return None
    m = re.search(
        r"(?:youtube\.com/(?:watch\?v=|embed/)|youtu\.be/)([a-zA-Z0-9_-]{11})",
        url,
    )
    return m.group(1) if m else None


# Høsteren skriver selv video-id'et to steder i hver transskription:
# i filnavnet (transcribe.py l.176, fetch_youtube_subs.py l.145) og som
# "Video ID:"-linje i headeren. Indekset læste ingen af dem og gættede i
# stedet via fuzzy titelmatch — hvilket er umuligt, når titlen ER filnavnet.
# 33 af 37 sådanne poster endte derfor uden link til deres egen video.
_FN_ID = re.compile(r"_([A-Za-z0-9_-]{11})_\d{8}_\d{6}\.txt$")
_HDR_ID = re.compile(r"^Video ID:\s*([A-Za-z0-9_-]{11})\s*$", re.M)
# En titel der stadig bærer høsterens dato+klokkeslæt er filnavnet, ikke en titel
_STOEJTITEL = re.compile(r"\b\d{8}[\s_]+\d{6}\b")


def id_from_transcript(filename: str, content: str = "") -> str | None:
    """Video-id'et som høsteren selv skrev. Autoritativt — intet gætværk."""
    m = _HDR_ID.search(content[:600]) if content else None
    if m:
        return m.group(1)
    m = _FN_ID.search(filename)
    return m.group(1) if m else None


_OEMBED_CACHE: dict[str, str | None] = {}


def youtube_title(video_id: str) -> str | None:
    """Rigtig videotitel via oEmbed. Virker også for unlisted videoer, hvor
    fetch_youtube_channel() ikke kan se dem — og det er langt de fleste."""
    if video_id in _OEMBED_CACHE:
        return _OEMBED_CACHE[video_id]
    title = None
    try:
        r = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": f"https://youtu.be/{video_id}", "format": "json"},
            timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200:
            title = (r.json().get("title") or "").strip() or None
    except Exception:
        pass
    _OEMBED_CACHE[video_id] = title
    return title


def decode_filename(filename: str) -> str:
    """'Godmorgen_med_DNNK__Aarhus__erfaringer.mp3.txt' → clean title."""
    name = filename
    for ext in (".mp3.txt", ".aac.txt", ".wav.txt", ".txt"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    name = unquote(name)
    # Klip høsterens id+timestamp-suffiks af, så titlen ikke bliver til støj
    # som "Tech Talks 2vrQr211cwE 20260806 171134".
    name = re.sub(r"_[A-Za-z0-9_-]{11}_\d{8}_\d{6}$", "", name)
    name = re.sub(r"_{2,}", ": ", name)   # double underscores → colon
    name = name.replace("_", " ")
    name = re.sub(r"\s+", " ", name).strip()
    return name


def detect_category(filename: str) -> str:
    f = filename.lower()
    if "godmorgen" in f:
        return "Godmorgen med DNNK"
    if "tech" in f and "talk" in f:
        return "Tech Talk"
    if "masterclass" in f:
        return "Masterclass"
    if "jura" in f:
        return "Jura"
    return "Øvrige"


# ── PDF-dokumenter fra PDF-scraperen ─────────────────────────────────────────

PDF_FILENAME_PREFIX = "PDF_"

_PDF_HEADER_START = re.compile(r"^=+\s*DNNK PDF Dokument\s*=+$")
_PDF_HEADER_END = re.compile(r"^={10,}$")
_PDF_HEADER_FIELD = re.compile(r"^(Titel|Kategori|Kilde|Indekseret):\s*(.*)$")


def is_pdf_filename(filename: str) -> bool:
    return filename.startswith(PDF_FILENAME_PREFIX)


def parse_pdf_document(content: str) -> tuple[dict | None, str]:
    """
    Parsér headeren fra en PDF-scrapet tekstfil:

        === DNNK PDF Dokument ===
        Titel: <titel>
        Kategori: <kategori>
        Kilde: <pdf-url>
        Indekseret: <iso-dato>
        ==================================================
        <udtrukket tekst>

    Returnerer (meta, body) hvor meta har nøglerne title/category/source_url/date
    (date som YYYY-MM-DD eller None). Hvis headeren mangler eller er ugyldig,
    returneres (None, content) så kalderen kan falde tilbage til filnavnet.
    """
    lines = content.splitlines()
    if not lines or not _PDF_HEADER_START.match(lines[0].strip()):
        return None, content

    fields: dict[str, str] = {}
    body_start = len(lines)
    for i in range(1, len(lines)):
        stripped = lines[i].strip()
        if _PDF_HEADER_END.match(stripped):
            body_start = i + 1
            break
        m = _PDF_HEADER_FIELD.match(stripped)
        if m:
            fields[m.group(1).lower()] = m.group(2).strip()

    # Dato: kun YYYY-MM-DD-delen af ISO-datoen (fx '2026-07-01T09:30:00')
    date = None
    m = re.match(r"(\d{4}-\d{2}-\d{2})", fields.get("indekseret", ""))
    if m:
        date = m.group(1)

    meta = {
        "title": fields.get("titel") or None,
        "category": fields.get("kategori") or None,
        "source_url": fields.get("kilde") or None,
        "date": date,
    }
    body = "\n".join(lines[body_start:]).strip()
    return meta, body


_SERIES_PREFIX = re.compile(
    r"^\s*(?:tech[\s_-]*talk(?:\s*\d+)?|god\s*morgen\s+med\s+dnnk|dnnk[\s_-]*masterclass(?:\s+om)?"
    r"|webinar(?:\s+\d+)?(?:\s+om\s+jura)?)\s*[:–—-]*\s*",
    re.I,
)

# Kalibreret mod det faktiske indeks (juli 2026): under ~0.75 er næsten alle
# kandidater falske positiver (mange optagelser — fx konference-oplæg — findes
# slet ikke på kategorisiderne). Et forkert dnnk_url/dato er værre end null.
MATCH_THRESHOLD = 0.75
MATCH_THRESHOLD_SHORT = 0.85  # korte titler matcher for let — kræv mere


def title_similarity(a: str, b: str) -> float:
    """
    Robust similarity that handles missing æ/ø/å in filenames.
    Fælles seriepræfikser ('Tech Talk:', 'Godmorgen med DNNK:') strippes først —
    ellers scorer to urelaterede webinarer i samme serie kunstigt højt.
    Uses four strategies and returns the best score:
    1. Standard normalization
    2. ASCII normalization (æ→ae, ø→oe, å→aa) + remove spaces
    3. Consonant-only comparison (most robust against missing vowels)
    4. Truncation-tolerant comparison (filenames are cut at ~60 chars)
    """
    a = _SERIES_PREFIX.sub("", a)
    b = _SERIES_PREFIX.sub("", b)

    def norm_std(s):
        s = s.lower()
        s = re.sub(r"[^a-z0-9æøå ]", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    def norm_ascii(s):
        s = s.lower()
        s = s.replace("æ", "ae").replace("ø", "oe").replace("å", "aa")
        return re.sub(r"[^a-z0-9]", "", s)

    def consonants(s):
        s = norm_ascii(s)
        return "".join(c for c in s if c.isalpha() and c not in "aeiou")

    sim1 = SequenceMatcher(None, norm_std(a), norm_std(b)).ratio()
    sim2 = SequenceMatcher(None, norm_ascii(a), norm_ascii(b)).ratio()
    sim3 = SequenceMatcher(None, consonants(a), consonants(b)).ratio()
    # 4. Trunkerings-tolerant: Transkriptor klipper mp3-navne ved ~60 tegn,
    #    så sammenlign kun op til den kortestes længde (kræver rimelig længde
    #    for ikke at give falske positiver på korte strenge).
    sim4 = 0.0
    na, nb = norm_ascii(a), norm_ascii(b)
    m = min(len(na), len(nb))
    if m >= 25:
        sim4 = SequenceMatcher(None, na[:m], nb[:m]).ratio() * 0.98
    return max(sim1, sim2, sim3, sim4)

# ── GitHub ────────────────────────────────────────────────────────────────────

def get_github_headers() -> dict:
    h = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def is_junk_filename(filename: str) -> bool:
    """Skip timestamp-only or otherwise meaningless filenames."""
    name = filename
    for ext in (".mp3.txt", ".aac.txt", ".wav.txt", ".m4a.txt", ".txt"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    # All-digit filename (unix timestamp etc)
    if re.fullmatch(r"\d{8,}", name.replace("_", "")):
        return True
    # Too short to be a real title
    if len(name) < 10:
        return True
    # Just a UUID
    if re.fullmatch(r"[a-f0-9-]{30,}", name.lower()):
        return True
    return False

def get_all_transcription_files() -> list[dict]:
    url = (
        f"https://api.github.com/repos/{TRANSCRIPTIONS_REPO}"
        "/git/trees/main?recursive=1"
    )
    resp = requests.get(url, headers=get_github_headers(), timeout=30)
    resp.raise_for_status()
    tree = resp.json().get("tree", [])

    files = []
    for item in tree:
        path = item["path"]
        if (
            path.startswith(BASE_FOLDER)
            and path.endswith(".txt")
            and item["type"] == "blob"
        ):
            filename = os.path.basename(path)
            # PDF-filer får titel m.m. fra headeren — filnavnet må ikke afvise dem
            if not is_pdf_filename(filename) and is_junk_filename(filename):
                continue
            folder = os.path.basename(os.path.dirname(path))
            safe_path = quote(path, safe="/")
            files.append(
                {
                    "path": path,
                    "filename": filename,
                    "folder": folder,
                    "raw_url": (
                        f"https://raw.githubusercontent.com"
                        f"/{TRANSCRIPTIONS_REPO}/main/{safe_path}"
                    ),
                }
            )
    return files

# ── DNNK scraping ─────────────────────────────────────────────────────────────

def fetch_youtube_channel() -> list[dict]:
    """
    Fetch all videos from DNNK's YouTube channel using yt-dlp.
    Returns list of {title, youtube_id, youtube_url}.
    """
    if yt_dlp is None:
        print("  Warning – yt_dlp er ikke installeret; springer YouTube-kanal over")
        return []
    try:
        print("  Henter videoer fra DNNK YouTube-kanal …")
        ydl_opts = {
            "quiet": True,
            "extract_flat": True,
            "playlist_items": "1-500",
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(DNNK_YOUTUBE_CHANNEL, download=False)
            videos = []
            for entry in (info or {}).get("entries", []):
                vid_id = entry.get("id", "")
                title  = entry.get("title", "")
                if vid_id and title:
                    raw_date = entry.get("upload_date", "") or ""
                    upload_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}" if len(raw_date) == 8 else None
                    videos.append({
                        "title":       title,
                        "youtube_id":  vid_id,
                        "youtube_url": f"https://youtube.com/watch?v={vid_id}",
                        "upload_date": upload_date,
                    })
            print(f"  Fandt {len(videos)} videoer på YouTube-kanalen")
            return videos
    except Exception as exc:
        print(f"  Warning – YouTube-kanal ikke tilgængelig: {exc}")
        return []

def _title_from_table_row(tr, link_el) -> str:
    """
    dnnk.dk lister Godmorgen-webinarer og Tech Talks i <table>-rækker.
    Titlen står som ren tekst i en søskende-celle til link-cellen — IKKE i en
    heading — så heading-strategien fandt kun linkteksten ('Se webinaret her >').
    Tag den længste celle (titelcellen), og fjern dato-præfiks og
    oplægsholder-suffiks ('v. Navn, Organisation').
    """
    link_td = link_el.find_parent("td")
    texts = []
    for td in tr.find_all("td"):
        if td is link_td:
            continue
        t = re.sub(r"\s+", " ", td.get_text(" ", strip=True)).strip()
        if t:
            texts.append(t)
    if not texts:
        return ""
    best = max(texts, key=len)
    best = re.sub(r"^\s*\d{2}\.\d{2}\.\d{4}\s*:?\s*", "", best)  # ledende dato
    best = re.sub(r"\sv\.\s.*$", "", best)                        # oplægsholder-suffiks
    return best.strip(" -–:")


def _date_from_text(text: str) -> str | None:
    """Find DD.MM.YYYY i tekst og returnér som YYYY-MM-DD."""
    m = re.search(r"(\d{2})\.(\d{2})\.(\d{4})", text)
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    return None


def scrape_category_subpages(cat_name: str, cat_url: str, soup, seen_yt: set,
                             max_pages: int = 30) -> list[dict]:
    """
    Crawl event-UNDERSIDER fra en kategoriside (fx Masterclass): videoerne
    ligger ikke på kategorisiden, men på én underside pr. event
    (fx /dnnk-masterclass-om-klimamodeller-...-v-dmi/).
    Henter YouTube-links, titel (h1) og evt. dato fra hver underside.
    """
    events = []
    cat_path = urlparse(cat_url).path.strip("/").lower()
    sub_urls = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/"):
            href = "https://www.dnnk.dk" + href
        parsed = urlparse(href)
        if not parsed.netloc.lower().endswith("dnnk.dk"):
            continue
        path = parsed.path.strip("/").lower()
        # Undersider genkendes på at kategoriens navn indgår i stien
        # (fx 'masterclass'), uden at det er selve kategorisiden.
        if "masterclass" not in path or path == cat_path:
            continue
        clean = f"https://www.dnnk.dk/{parsed.path.strip('/')}/"
        if clean not in sub_urls:
            sub_urls.append(clean)

    for sub_url in sub_urls[:max_pages]:
        try:
            resp = requests.get(sub_url, headers=HTTP_HEADERS, timeout=15)
            resp.raise_for_status()
            ssoup = BeautifulSoup(resp.text, "html.parser")
            h1 = ssoup.find("h1")
            title = h1.get_text(strip=True) if h1 else ""
            if not title or len(title) < 8:
                continue
            for tag in ssoup.find_all(["nav", "header", "footer", "script", "style"]):
                tag.decompose()
            date = _date_from_text(ssoup.get_text())
            yt_ids = []
            for el in ssoup.find_all("a", href=re.compile(r"youtu")):
                yt_ids.append(extract_youtube_id(el["href"]))
            for el in ssoup.find_all("iframe", src=re.compile(r"youtu")):
                yt_ids.append(extract_youtube_id(el.get("src", "")))
            for yt_id in yt_ids:
                if not yt_id or yt_id in seen_yt:
                    continue
                seen_yt.add(yt_id)
                events.append({
                    "title":       title,
                    "category":    cat_name,
                    "event_url":   sub_url,
                    "youtube_id":  yt_id,
                    "youtube_url": f"https://youtube.com/watch?v={yt_id}",
                    "date":        date,
                })
        except Exception as exc:
            print(f"    Warning – kunne ikke hente underside {sub_url}: {exc}")
        time.sleep(0.5)  # høflig pause mellem undersider

    print(f"    {cat_name}: {len(sub_urls[:max_pages])} undersider crawlet")
    return events


def scrape_dnnk_events() -> list[dict]:
    """
    Scrape all DNNK category pages.
    Godmorgen/Tech Talks/Jura lister webinarer direkte på kategorisiden;
    Masterclass har én underside pr. event (crawles via scrape_category_subpages).
    Returns list of {title, category, event_url, youtube_id, youtube_url, date}.
    """
    events = []

    for cat_name, cat_url in DNNK_CATEGORY_PAGES:
        try:
            print(f"  Scraping {cat_url} …")
            resp = requests.get(cat_url, headers=HTTP_HEADERS, timeout=15)
            resp.raise_for_status()  # en 404/500-fejlside må ikke parses som "0 events" i stilhed
            soup = BeautifulSoup(resp.text, "html.parser")

            # Remove boilerplate
            for tag in soup.find_all(["nav", "header", "footer", "script", "style"]):
                tag.decompose()

            # Strategy: find every YouTube link, then extract title and date from context
            seen_yt = set()
            for yt_a in soup.find_all("a", href=re.compile(r"youtu")):
                yt_id = extract_youtube_id(yt_a["href"])
                if not yt_id or yt_id in seen_yt:
                    continue
                seen_yt.add(yt_id)

                title = ""
                date = None

                # Primær strategi: dnnk.dk bruger tabeller (Godmorgen, Tech Talks)
                # hvor titlen står i en søskende-<td> — ikke i en heading.
                tr = yt_a.find_parent("tr")
                if tr:
                    title = _title_from_table_row(tr, yt_a)
                    date = _date_from_text(tr.get_text(" ", strip=True))

                # Sekundær strategi: gå op og find heading/fed tekst (Jura-siden)
                if not title:
                    parent = yt_a.find_parent(["li", "tr", "article", "div", "section"])
                    for _ in range(4):  # max 4 levels up
                        if not parent:
                            break
                        # Look for heading or bold text as title
                        for tag in ["h1", "h2", "h3", "h4", "strong", "b"]:
                            el = parent.find(tag)
                            if el:
                                t = el.get_text(strip=True)
                                if len(t) > 8 and t != yt_a.get_text(strip=True):
                                    title = t
                                    break
                        if not date:
                            date = _date_from_text(parent.get_text())
                        if title:
                            break
                        parent = parent.find_parent(["li", "tr", "article", "div", "section"])

                # Fallback: use link text if no heading found — men aldrig rene
                # navigationslinktekster som 'Se webinaret her >'
                if not title:
                    t = yt_a.get_text(strip=True)
                    if not re.search(r"^se\b|klik her|læs mere|>$", t.lower()):
                        title = t
                if not title or len(title) < 5:
                    continue

                events.append({
                    "title":       title,
                    "category":    cat_name,
                    "event_url":   cat_url,   # no individual event page — use category URL
                    "youtube_id":  yt_id,
                    "youtube_url": f"https://youtube.com/watch?v={yt_id}",
                    "date":        date,
                })

            # Also check iframes (embedded players)
            for iframe in soup.find_all("iframe", src=re.compile(r"youtu")):
                yt_id = extract_youtube_id(iframe.get("src", ""))
                if not yt_id or yt_id in seen_yt:
                    continue
                seen_yt.add(yt_id)
                parent = iframe.find_parent(["li", "tr", "article", "div"])
                title = ""
                date = None
                if parent:
                    if parent.name == "tr":
                        title = _title_from_table_row(parent, iframe)
                    if not title:
                        for tag in ["h1", "h2", "h3", "h4", "strong"]:
                            el = parent.find(tag)
                            if el:
                                title = el.get_text(strip=True)
                                break
                    date = _date_from_text(parent.get_text())
                if not title:
                    continue
                events.append({
                    "title":       title,
                    "category":    cat_name,
                    "event_url":   cat_url,
                    "youtube_id":  yt_id,
                    "youtube_url": f"https://youtube.com/watch?v={yt_id}",
                    "date":        date,
                })

            # Kategorier hvor events ligger på undersider (fx Masterclass)
            if cat_name in SUBPAGE_CATEGORIES:
                events.extend(scrape_category_subpages(cat_name, cat_url, soup, seen_yt))

            time.sleep(0.5)
        except Exception as exc:
            print(f"  Warning – could not scrape {cat_url}: {exc}")

    print(f"  Found {len(events)} events on dnnk.dk")
    return events


def get_event_description(event_url: str) -> str | None:
    """Fetch invitationstekst from a dnnk.dk/event/... page."""
    if not event_url:
        return None
    try:
        resp = requests.get(event_url, headers=HTTP_HEADERS, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove boilerplate
        for tag in soup.find_all(["nav", "header", "footer", "script", "style"]):
            tag.decompose()

        # Try common content containers
        for selector in [
            {"class": re.compile(r"entry-content|post-content|event-description", re.I)},
            {"class": re.compile(r"content", re.I)},
        ]:
            el = soup.find("div", **selector)
            if el:
                text = el.get_text(separator=" ", strip=True)
                text = re.sub(r"\s+", " ", text)
                return text[:600].strip()
    except Exception:
        pass
    return None


# Ord der er så generiske i DNNK-sammenhæng at de ikke beviser et match
_GENERIC_TOKENS = {
    "klimatilpasning", "klimatilpasningen", "klimatilpasningsprojekter",
    "webinar", "dnnk", "danmark", "dansk", "danske", "optagelse", "temadag",
}


def _shared_significant_token(a: str, b: str) -> bool:
    """Deler de to titler mindst ét betydende ord (>=5 tegn, ikke-generisk)?
    Substring-tjek mod den sammenkædede modpart, så æ/ø/å-hullede ord
    ('nseregion' fra 'gr nseregion') stadig kan genfindes i 'graenseregion'."""
    def norm(s: str) -> str:
        s = s.lower().replace("æ", "ae").replace("ø", "oe").replace("å", "aa")
        return re.sub(r"[^a-z0-9 ]", " ", s)
    na, nb = norm(a), norm(b)
    ta = [w for w in na.split() if len(w) >= 5 and w not in _GENERIC_TOKENS]
    tb = [w for w in nb.split() if len(w) >= 5 and w not in _GENERIC_TOKENS]
    na_j, nb_j = na.replace(" ", ""), nb.replace(" ", "")
    return any(w in nb_j for w in ta) or any(w in na_j for w in tb)


_SPEAKER_SUFFIX = re.compile(
    r"\s+(?:v\.|ved)\s+[A-ZÆØÅ][\wÆØÅæøå]+\s+[A-ZÆØÅ].*$"
)


def strip_speaker_suffix(title: str) -> str:
    """Fjern en afsluttende oplægsholder-angivelse fra en video-/event-titel.

    DNNK's YouTube-titler hedder ofte "<emne> v. <Navn>, <organisation>"
    ("Bidragsmodeller v. Helle Tegner Anker, Professor ved KU"), mens
    transskriptionens filnavn er klippet ved ~60 tegn og derfor kun hedder
    "Bidragsmodeller_v". Navnet trækker similarity langt ned, så et korrekt
    match blev afvist. Kræver to store begyndelsesbogstaver efter v./ved, så
    almindeligt dansk "ved" ikke rammes ("Interessentinddragelse ved
    klimatilpasning" forbliver urørt).
    """
    return _SPEAKER_SUFFIX.sub("", title).strip(" ,-–")


def _match_acceptable(a: str, b: str, score: float) -> bool:
    """Accept-regel for et titelmatch — kalibreret mod falske positiver."""
    a_s = _SERIES_PREFIX.sub("", a)
    b_s = _SERIES_PREFIX.sub("", b)
    # Tal-konflikt: 'Tørke og Hede i Danmark 1' må ikke matche '... 6'
    da = {str(int(d)) for d in re.findall(r"\d+", a_s)}
    db = {str(int(d)) for d in re.findall(r"\d+", b_s)}
    if da and db and not (da & db):
        return False
    # I gråzonen skal parret dele mindst ét betydende ord — SequenceMatcher
    # alene kan score to urelaterede titler ens på ordstumper og småord
    if score < 0.82 and not _shared_significant_token(a_s, b_s):
        return False
    def _alen(s: str) -> int:
        return len(re.sub(r"[^a-z0-9æøå]", "", s.lower()))
    thresh = MATCH_THRESHOLD if min(_alen(a_s), _alen(b_s)) >= 25 else MATCH_THRESHOLD_SHORT
    return score >= thresh


def find_best_event(title: str, events: list[dict]) -> tuple[dict | None, float]:
    """Returnerer (event, score) så confidence ikke skal genberegnes af kalderen.
    Bedste kandidat der opfylder accept-reglen vinder — en høj men afvist
    kandidat (fx tal-konflikt) må ikke skygge for en korrekt nr. 2."""
    scored = []
    for ev in events:
        if not ev.get("title"):
            continue
        scored.append((title_similarity(title, ev["title"]), ev))
    scored.sort(key=lambda x: -x[0])
    for score, ev in scored[:5]:
        if _match_acceptable(title, ev["title"], score):
            return ev, score
    return None, 0.0


def refresh_event_metadata(index: list[dict], dnnk_events: list[dict]) -> int:
    """
    Backfill: genkør KUN event-matchingen for eksisterende entries (ingen
    AI-kald) og opdatér date/dnnk_url/description/match_confidence.
    Uden dette får entries fra før et scraper-fix aldrig deres metadata —
    skip-logikken ('filename in existing_map') rører dem aldrig igen.
    Returnerer antal opdaterede entries.
    """
    if not dnnk_events:
        return 0
    category_urls = {url.rstrip("/") for _, url in DNNK_CATEGORY_PAGES}
    desc_cache: dict[str, str | None] = {}
    updated = 0
    for entry in index:
        # PDF-dokumenter har ingen dnnk.dk-event — spring dem over
        if entry.get("type") == "pdf":
            continue

        # Backfill af høsterens eget video-id. SKAL ligge før 'if not best:
        # continue' nedenfor: netop de entries der mangler id'et har en
        # støjtitel ("Tech Talks 2vrQr211cwE 20260806 171134"), så de finder
        # aldrig et event og ville aldrig nå længere ned i løkken.
        # Tælles direkte op i 'updated' — build_index gemmer kun 'if updated'.
        fn_id = id_from_transcript(entry.get("filename", ""))
        if fn_id and entry.get("youtube_id") != fn_id:
            entry["youtube_id"] = fn_id
            entry["youtube_url"] = f"https://youtube.com/watch?v={fn_id}"
            updated += 1
        # Titlen på netop disse entries er filnavnet ("Tech Talks 2vrQr211cwE
        # 20260806 171134") — ubrugelig i en søgning. Hent den rigtige.
        # Genkend på høsterens timestamp (8 cifre + 6 cifre), ikke på id'et:
        # et id med underscore ("uLiX_MX5qA4") står som "uLiX MX5qA4" i titlen
        # og ville aldrig matche sig selv.
        if fn_id and _STOEJTITEL.search(entry.get("title") or ""):
            ægte = youtube_title(fn_id)
            if ægte:
                entry["title"] = ægte
                updated += 1

        # Match både på det afkodede filnavn og på den (evt. AI-rettede) titel
        candidates = [decode_filename(entry["filename"])]
        if entry.get("title") and entry["title"] not in candidates:
            candidates.append(entry["title"])
        best, best_score = None, 0.0
        for cand in candidates:
            ev, score = find_best_event(cand, dnnk_events)
            if ev and score > best_score:
                best, best_score = ev, score
        if not best:
            continue

        changed = False
        if best.get("event_url") and entry.get("dnnk_url") != best["event_url"]:
            entry["dnnk_url"] = best["event_url"]
            changed = True
        if best.get("date") and entry.get("date") != best["date"]:
            entry["date"] = best["date"]
            changed = True
        # Et event-match er den autoritative kilde til videoen (titel og id
        # kommer fra samme tabelrække/underside på dnnk.dk). Overskriv derfor
        # også et FORKERT gemt id — ikke kun et tomt. Uden dette blev gamle
        # fejlmatch fra kanal-fallbacket aldrig rettet igen.
        # ... men et id fra filnavnet stammer fra selve hentningen og slår
        # også event-matchet. Rør det ikke.
        if (not fn_id and best.get("youtube_id")
                and entry.get("youtube_id") != best["youtube_id"]):
            entry["youtube_id"] = best["youtube_id"]
            entry["youtube_url"] = best["youtube_url"]
            changed = True
        if entry.get("match_confidence") != round(best_score, 3):
            entry["match_confidence"] = round(best_score, 3)
            changed = True
        # Beskrivelse hentes kun fra rigtige event-UNDERSIDER — kategorisider
        # giver bare listetekst ('Dato, Titel og Oplægsholdere …') som beskrivelse.
        ev_url = (best.get("event_url") or "").rstrip("/")
        if ev_url and ev_url not in category_urls and not entry.get("description"):
            if ev_url not in desc_cache:
                desc_cache[ev_url] = get_event_description(best["event_url"])
                time.sleep(0.5)
            if desc_cache[ev_url]:
                entry["description"] = desc_cache[ev_url]
                changed = True
        if changed:
            updated += 1
    return updated

# ── AI summary ────────────────────────────────────────────────────────────────

def generate_summary(client: anthropic.Anthropic, title: str, content: str, description: str | None = None, doc_type: str = "webinar") -> dict | None:
    """Returnerer dict med resumé-felter, eller None hvis AI-kaldet fejlede.
    None betyder: gem IKKE entry'en — filen prøves igen ved næste kørsel.
    doc_type: 'webinar' (default) eller 'pdf' (rapport/dokument — anden prompt,
    samme JSON-schema, så downstream-koden er uændret)."""
    excerpt = content[:5000]
    invitation_block = (
        f"Invitationstekst fra dnnk.dk (verificeret af DNNK, brug som primær kilde):\n{description}\n\n"
        if description else ""
    )
    if doc_type == "pdf":
        prompt = (
            f"Rapport/dokument titel: {title}\n\n"
            f"{invitation_block}"
            f"Uddrag af dokumentets tekst:\n{excerpt}\n\n"
            "Dette er en rapport eller et dokument om klimatilpasning – IKKE et webinar.\n"
            "Svar KUN med valid JSON – ingen forklaring:\n"
            '{"corrected_title": "Korrekt dansk titel med æ/ø/å",\n'
            ' "summary": "2-3 sætninger om indhold og vigtigste pointer (dansk)",\n'
            ' "keywords": ["nøgleord1", "nøgleord2", ...],\n'
            ' "speakers": [{"name": "Fornavn Efternavn", "org": "Organisation"}, ...],\n'
            ' "places": ["Stednavn1", "Stednavn2", ...]}\n\n'
            "Krav:\n"
            "- corrected_title: Behold titlen uændret medmindre den har åbenlyse fejl "
            "(fx manglende æ/ø/å).\n"
            "- summary: 2-3 sætninger på dansk om dokumentets indhold og vigtigste pointer/anbefalinger.\n"
            "- keywords: 8-12 ord (emner, steder, teknologier, metoder, aktører)\n"
            "- speakers: Forfattere eller udgivende organisationer hvis de fremgår tydeligt, max 4. Tom hvis uklart.\n"
            "- places: Konkrete danske eller udenlandske stednavne nævnt i dokumentet "
            "(byer, kommuner, fjorde, regioner, vandløb). Max 6. Kun navngivne steder, ikke generiske som 'kysten'."
        )
    else:
        prompt = (
            f"Webinar titel (kan have manglende æ/ø/å): {title}\n\n"
            f"{invitation_block}"
            f"Transskription (uddrag, brug som supplement):\n{excerpt}\n\n"
            "Svar KUN med valid JSON – ingen forklaring:\n"
            '{"corrected_title": "Korrekt dansk titel med æ/ø/å",\n'
            ' "summary": "2-3 sætninger om indhold og vigtigste pointer (dansk)",\n'
            ' "keywords": ["nøgleord1", "nøgleord2", ...],\n'
            ' "speakers": [{"name": "Fornavn Efternavn", "org": "Organisation"}, ...],\n'
            ' "places": ["Stednavn1", "Stednavn2", ...]}\n\n'
            "Krav:\n"
            "- corrected_title: Ret manglende eller forkerte æ/ø/å i titlen baseret på kontekst. "
            "Behold titlen uændret hvis den allerede er korrekt.\n"
            "- summary: Basér primært på invitationsteksten hvis den findes. "
            "Supplér med pointer fra transskriptionen som invitationen ikke dækker.\n"
            "- summary: 2-3 sætninger på dansk\n"
            "- keywords: 8-12 ord (emner, steder, teknologier, metoder, aktører)\n"
            "- speakers: Identificer faktiske oplægsholdere (ikke moderator), max 4. Tom hvis uklart.\n"
            "- places: Konkrete danske eller udenlandske stednavne nævnt i webinaret "
            "(byer, kommuner, fjorde, regioner, vandløb). Max 6. Kun navngivne steder, ikke generiske som 'kysten'."
        )
    json_retries = 0
    for attempt in range(3):
        try:
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=850,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            raw = re.sub(r"^```json\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            try:
                return json.loads(raw)
            except json.JSONDecodeError as exc:
                # Ugyldig JSON fra modellen: giv den ét ekstra forsøg
                if json_retries < 1:
                    json_retries += 1
                    print(f"    Warning – ugyldig JSON fra modellen, prøver igen: {exc}")
                    continue
                print(f"    Warning – ugyldig JSON fra modellen igen: {exc}")
                return None
        except (anthropic.RateLimitError, anthropic.APIStatusError) as e:
            if hasattr(e, 'status_code') and e.status_code not in (429, 529):
                # Ikke-midlertidig API-fejl (fx 400 ved månedligt forbrugsloft).
                # Må ALDRIG vælte det natlige job — filen springes over og
                # prøves igen næste kørsel. Ved forbrugsloft stopper vi helt
                # med at forsøge flere filer i denne kørsel.
                global AI_BUDGET_EXHAUSTED
                if "usage limits" in str(e).lower():
                    AI_BUDGET_EXHAUSTED = True
                    print("    ::warning::Anthropic-forbrugsloftet er nået — "
                          "nye filer indekseres, når loftet nulstilles.")
                else:
                    print(f"    Warning – API-fejl ({getattr(e, 'status_code', '?')}): {e}")
                return None
            wait = 60 * (attempt + 1)
            print(f"    Rate limit – venter {wait}s …")
            time.sleep(wait)
        except Exception as exc:
            print(f"    Warning – AI generation failed: {exc}")
            return None
    return None

# ── External resources ────────────────────────────────────────────────────────

def scrape_external_resources() -> list[dict]:
    """
    Scrape DNNK rapporter og klimatilpasning.dk publikationer/vejledninger.
    Returns list of {title, url, source}.
    """
    resources = []

    for label, base_url, source in EXTERNAL_RESOURCE_PAGES:
        try:
            print(f"  Scraping ekstern kilde: {base_url} …")
            page_url = base_url
            pages_scraped = 0
            while page_url and pages_scraped < 8:
                resp = requests.get(page_url, headers=HTTP_HEADERS, timeout=15)
                soup = BeautifulSoup(resp.text, "html.parser")

                # Find article/resource links — try multiple selectors
                found = 0
                for a in soup.find_all("a", href=True):
                    href = a["href"]
                    title = a.get_text(strip=True)
                    if not title or len(title) < 10:
                        continue
                    # Skip navigation/boilerplate links
                    if any(skip in title.lower() for skip in ["log ind", "søg", "menu", "læs mere", "detaljer", "→", "←"]):
                        continue
                    if not href.startswith("http"):
                        href = "https://www." + source + href if href.startswith("/") else href
                    # Only keep links on same domain (rigtigt domænecheck,
                    # ikke substring — 'dnnk.dk' må ikke matche fx en sti)
                    netloc = urlparse(href).netloc.lower()
                    if not (netloc == source or netloc.endswith("." + source)):
                        continue
                    # Skip category/index pages — only individual resources
                    if href.rstrip("/") == base_url.rstrip("/"):
                        continue
                    if is_junk_resource(href, title): continue
                    resources.append({"title": title, "url": href, "source": label, "type": detect_resource_type(href, title)})
                    found += 1

                # Follow pagination
                next_link = soup.find("a", string=re.compile(r"→|næste|next", re.I))
                if next_link and next_link.get("href") and found > 0:
                    next_href = next_link["href"]
                    if not next_href.startswith("http"):
                        next_href = "https://www." + source + next_href
                    page_url = next_href if next_href != page_url else None
                else:
                    page_url = None
                pages_scraped += 1
                time.sleep(0.5)

        except Exception as exc:
            print(f"  Warning – kunne ikke scrape {base_url}: {exc}")

    # Deduplicate by URL
    seen = set()
    unique = []
    for r in resources:
        if r["url"] not in seen:
            seen.add(r["url"])
            unique.append(r)

    print(f"  Fandt {len(unique)} eksterne ressourcer")
    return unique


def keyword_overlap(kw_list: list[str], text: str) -> int:
    """Count how many keywords from kw_list appear in text (case-insensitive)."""
    text_lower = text.lower()
    return sum(1 for kw in kw_list if kw.lower() in text_lower)


def find_related_resources(keywords: list[str], title: str, resources: list[dict], top_n: int = 3) -> list[dict]:
    """Find top_n external resources matching by weighted keyword + title overlap.
    Requires at least 2 overlapping keywords for inclusion."""
    if not keywords or not resources:
        return []
    scored = []
    title_low = title.lower()
    for r in resources:
        kw_score = keyword_overlap(keywords, r["title"]) * 3  # nøgleord vægter 3x
        # Bonus hvis ressourcetitlen indeholder et stort ord fra webinartitlen
        title_words = [w for w in title_low.split() if len(w) > 5]
        title_bonus = sum(1 for w in title_words if w in r["title"].lower())
        total = kw_score + title_bonus
        if kw_score >= 6:   # mindst 2 nøgleords-overlap (2 × 3 = 6)
            scored.append((total, r))
    scored.sort(key=lambda x: -x[0])
    return [r for _, r in scored[:top_n]]


def compute_related_webinars(index: list[dict], top_n: int = 3) -> None:
    """
    For each entry in index, find top_n related webinars by keyword overlap.
    Mutates index in place by adding 'related_webinars' key.
    PDF-entries deltager på lige fod (keyword-overlap er ens); de har ingen
    youtube_url/dnnk_url, så source_url og type medtages i krydslinket.
    """
    for entry in index:
        kw = entry.get("keywords", [])
        if not kw:
            entry["related_webinars"] = []
            continue
        scored = []
        entry_key = entry.get("path") or entry["filename"]
        for other in index:
            if (other.get("path") or other["filename"]) == entry_key:
                continue
            other_kw = other.get("keywords", [])
            other_text = " ".join(other_kw) + " " + other.get("title", "")
            score = keyword_overlap(kw, other_text)
            if score >= 2:  # require at least 2 overlapping keywords
                scored.append((score, {
                    "title": other["title"],
                    "filename": other["filename"],
                    "type": other.get("type", "webinar"),
                    "youtube_url": other.get("youtube_url"),
                    "dnnk_url": other.get("dnnk_url"),
                    "source_url": other.get("source_url"),
                }))
        scored.sort(key=lambda x: -x[0])
        entry["related_webinars"] = [r for _, r in scored[:top_n]]


# ── Main ──────────────────────────────────────────────────────────────────────

def load_existing_index() -> list[dict]:
    rebuild = os.environ.get("REBUILD", "").lower() in ("1", "true", "yes")
    if rebuild:
        print("  REBUILD=true – starter forfra")
        return []
    if os.path.exists("search-index.json"):
        with open("search-index.json", encoding="utf-8-sig") as f:
            return json.load(f)
    return []


def save_index(index: list[dict]) -> None:
    with open("search-index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def build_index():
    print("Loading existing index …")
    existing = load_existing_index()
    # Dedupe pr. path (ikke kun filnavn) — samme filnavn kan ligge i flere mapper.
    # Ældre entries uden path falder tilbage til filename.
    existing_map = {e.get("path") or e["filename"]: e for e in existing}
    print(f"  {len(existing)} entries already indexed")

    print("Fetching transcription file list from GitHub …")
    all_files = get_all_transcription_files()
    new_files = [f for f in all_files if f["path"] not in existing_map and f["filename"] not in existing_map]
    print(f"  {len(all_files)} total files, {len(new_files)} new")

    # ── Ryd op efter slettede transskriptioner ───────────────────────────────
    # Uden dette bliver en entry hængende for evigt, når filen fjernes fra
    # transskriptions-repoet (fx ved dublet-oprydning) — indekset kunne kun
    # vokse. Sikkerhedsventil: spring over hvis fillisten ser mistænkeligt
    # tom ud (GitHub-fejl eller rate limit må ikke tømme indekset).
    pruned = 0
    if len(all_files) >= 50:
        live = {f["path"] for f in all_files} | {f["filename"] for f in all_files}
        stale = [e for e in existing if (e.get("path") or e["filename"]) not in live]
        if stale:
            pruned = len(stale)
            for e in stale:
                print(f"  Fjerner entry uden fil: {e.get('title', e['filename'])[:60]}")
            existing = [e for e in existing if e not in stale]
            existing_map = {e.get("path") or e["filename"]: e for e in existing}
            print(f"  {len(stale)} entries fjernet, {len(existing)} tilbage")
    else:
        print(f"::warning::kun {len(all_files)} filer hentet fra GitHub - "
              "springer oprydning af slettede entries over")

    print("Scraping dnnk.dk for event metadata …")
    dnnk_events = scrape_dnnk_events()
    if not dnnk_events:
        # GitHub Actions-annotation så et strukturskifte på dnnk.dk opdages i workflow-loggen
        print("::warning::scrape_dnnk_events() fandt 0 events - dnnk.dk-strukturen kan vaere aendret; tjek kategorisiderne og parsing-heuristikken i scrape_dnnk_events()")

    index = list(existing)
    updated = 0

    # ── Metadata-refresh af EKSISTERENDE entries (default, ingen API-kald) ────
    # Kører før og uafhængigt af AI-delen, så date/dnnk_url/description kan
    # backfilles selv når ANTHROPIC_API_KEY mangler.
    if index and dnnk_events:
        print("Opdaterer event-metadata for eksisterende entries …")
        updated = refresh_event_metadata(index, dnnk_events)
        with_url = sum(1 for e in index if e.get("dnnk_url"))
        with_date = sum(1 for e in index if e.get("date"))
        print(f"  {updated} entries opdateret — {with_url}/{len(index)} har dnnk_url, {with_date}/{len(index)} har dato")
        if updated:
            save_index(index)

    # En ren oprydning (ingen metadata-ændringer, ingen nye filer) skal også
    # gemmes — ellers ville de fjernede entries dukke op igen næste kørsel.
    if pruned and not updated:
        save_index(index)

    if not new_files:
        print("Ingen nye filer at behandle.")
        return

    # ── Nye filer kræver AI-resumé (anthropic + ANTHROPIC_API_KEY) ───────────
    if anthropic is None or not os.environ.get("ANTHROPIC_API_KEY"):
        print(f"::warning::ANTHROPIC_API_KEY mangler (eller anthropic-pakken er ikke installeret) - "
              f"springer {len(new_files)} nye filer over. Kun metadata-refresh af eksisterende entries blev udfoert.")
        return
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    print("Henter DNNK YouTube-kanal …")
    youtube_videos = fetch_youtube_channel()

    print("Scraping eksterne ressourcer (vidensbank, klimatilpasning.dk) …")
    ext_resources = scrape_external_resources()

    for i, file_info in enumerate(new_files):
        filename = file_info["filename"]
        is_pdf = is_pdf_filename(filename)
        title = decode_filename(filename)
        category = detect_category(filename)

        print(f"[{i+1}/{len(new_files)}] {title[:70]} …")

        # Fetch transcription text
        try:
            resp = requests.get(file_info["raw_url"], timeout=15)
            resp.raise_for_status()
            resp.encoding = "utf-8"  # rå .txt fra GitHub er altid UTF-8
            content = resp.text
        except Exception as exc:
            print(f"  Error fetching content: {exc}")
            continue

        matched = None
        match_confidence = 0.0
        event_url = youtube_id = youtube_url = description = date = None
        source_url = None

        if is_pdf:
            # PDF-dokument: metadata fra headeren — INGEN event-/YouTube-matching
            meta, body = parse_pdf_document(content)
            if meta:
                title = meta["title"] or title
                category = meta["category"] or "Rapporter"
                source_url = meta["source_url"]
                date = meta["date"]
                content = body or content
                print(f"  → PDF-dokument: {title[:60]}")
            else:
                category = "Rapporter"
                print(f"  Warning – kunne ikke parse PDF-header i {filename}; "
                      "bruger filnavnet som titel")
        else:
            # Høsterens eget id slår al titelmatching: det er skrevet af den
            # kode der hentede videoen, ikke gættet ud fra en tekststreng.
            youtube_id = id_from_transcript(filename, content)
            if youtube_id:
                youtube_url = f"https://youtube.com/watch?v={youtube_id}"
                vid = next((v for v in youtube_videos
                            if v.get("youtube_id") == youtube_id), None)
                if vid:
                    # Rigtig titel i stedet for det afkodede filnavn
                    if vid.get("title"):
                        title = vid["title"]
                    if vid.get("upload_date"):
                        date = vid["upload_date"]
                print(f"  → id fra transskription: {youtube_id}")
            # Match with dnnk.dk event
            matched, match_confidence = find_best_event(title, dnnk_events)

        if matched:
            event_url = matched.get("event_url")
            # Et id fra transskriptionen må ikke overskrives af et gæt
            if not youtube_id:
                youtube_id = matched.get("youtube_id")
                youtube_url = matched.get("youtube_url")
            # Behold en dato vi allerede har, hvis eventet ikke selv har en
            date = matched.get("date") or date
            # Use DNNK title when match is confident — fixes æ/ø/å lost in filename encoding
            # (find_best_event returnerer kun matches der opfylder _match_acceptable)
            if matched.get("title"):
                title = matched["title"]
            print(f"  → matched ({match_confidence:.2f}): {matched['title'][:60]}")
            # Beskrivelse kun fra rigtige event-undersider — kategorisider
            # giver bare listetekst som "beskrivelse"
            category_urls = {url.rstrip("/") for _, url in DNNK_CATEGORY_PAGES}
            if event_url and event_url.rstrip("/") not in category_urls:
                description = get_event_description(event_url)
                time.sleep(0.5)

        # Fallback 1: match against YouTube channel video titles (ikke for PDF)
        if not is_pdf and not youtube_id and youtube_videos:
            best_yt_score = 0.0
            best_yt = None
            for vid in youtube_videos:
                # Sammenlign også uden oplægsholder-suffiks: filnavnet er
                # klippet ved ~60 tegn, så "Bidragsmodeller_v" skal kunne
                # matche "Bidragsmodeller v. Helle Tegner Anker, KU".
                cand_titles = {vid["title"], strip_speaker_suffix(vid["title"])}
                score = max(title_similarity(title, ct) for ct in cand_titles)
                if score > best_yt_score:
                    best_yt_score = score
                    best_yt = vid
            # Samme accept-regel som for dnnk.dk-events. Det gamle løse krav
            # (score >= 0.45, ingen anden validering) gjorde, at "bedste" match
            # reelt var støj: 179 entries endte med at dele 44 video-id'er, så
            # "se webinaret" pegede på det forkerte klip for det meste af
            # arkivet. Et manglende link er bedre end et forkert.
            if best_yt and _match_acceptable(title, best_yt["title"], best_yt_score):
                youtube_id  = best_yt["youtube_id"]
                youtube_url = best_yt["youtube_url"]
                if not date and best_yt.get("upload_date"):
                    date = best_yt["upload_date"]
                print(f"  → YouTube match ({best_yt_score:.2f}): {best_yt['title'][:60]}")
            elif best_yt:
                print(f"  → YouTube-match afvist ({best_yt_score:.2f}): {best_yt['title'][:60]}")

        # Fallback 2 FJERNET: et YouTube-link der optræder INDE i en
        # transskription er som regel noget oplægsholderen henviste til — ikke
        # optagelsen af webinaret selv. Den regel gav bl.a. jura-webinarer et
        # link til "Horizon og LIFE projekter om klimatilpasning". Et manglende
        # link er bedre end et forkert.

        # Generate AI summary
        if AI_BUDGET_EXHAUSTED:
            # Forbrugsloftet er nået — ingen grund til at forsøge flere filer
            # i denne kørsel; de samles op automatisk, når loftet nulstilles.
            print(f"  Springer {filename} over (Anthropic-forbrugsloft — prøves igen senere)")
            continue
        print("  → generating summary …")
        ai = generate_summary(client, title, content, description,
                              doc_type="pdf" if is_pdf else "webinar")
        if ai is None:
            # Gem IKKE entry med tomt resumé — spring over, så filen prøves igen næste kørsel
            print(f"  Warning – AI-resumé fejlede for {filename}; springer over (prøves igen næste kørsel)")
            continue
        if ai.get("corrected_title") and len(ai["corrected_title"]) > 5:
            title = ai["corrected_title"]

        related_resources = find_related_resources(ai.get("keywords", []), title, ext_resources)

        index.append(
            {
                "filename": filename,
                "path": file_info["path"],
                "folder": file_info["folder"],
                "type": "pdf" if is_pdf else "webinar",
                "title": title,
                "category": category,
                "date": date,
                "summary": ai.get("summary", ""),
                "keywords": ai.get("keywords", []),
                "speakers": ai.get("speakers", []),
                "places": ai.get("places", []),
                "youtube_id": youtube_id,
                "youtube_url": youtube_url,
                "dnnk_url": event_url,
                "source_url": source_url,  # PDF-kilde-URL; None for webinarer
                "match_confidence": round(match_confidence, 3) if matched else None,
                "description": description,
                "related_resources": related_resources,
                "related_webinars": [],  # filled in after all entries are built
            }
        )

        time.sleep(0.3)

        # Gem løbende hver 10. entry så vi ikke mister arbejde ved fejl
        if len(index) % 10 == 0:
            save_index(index)

    # Drop ressourcer der matcher 40%+ af alle entries (for generiske til at være nyttige)
    print("Filtrerer for generiske ressourcer …")
    resource_counts = {}
    for entry in index:
        for r in entry.get("related_resources", []):
            resource_counts[r["url"]] = resource_counts.get(r["url"], 0) + 1
    threshold = len(index) * 0.40
    too_common = {url for url, cnt in resource_counts.items() if cnt > threshold}
    if too_common:
        print(f"  Fjerner {len(too_common)} for generiske ressourcer:")
        for url in too_common:
            print(f"    - {url} (matchede {resource_counts[url]} entries)")
    for entry in index:
        entry["related_resources"] = [r for r in entry.get("related_resources", []) if r["url"] not in too_common]

    # Compute related webinars across entire index
    print("Beregner krydsreferencer mellem webinarer …")
    compute_related_webinars(index)

    # Sort chronologically by folder, then title
    index.sort(key=lambda x: (x.get("folder", ""), x.get("title", "")))

    save_index(index)

    print(f"\nDone. {len(index)} entries written to search-index.json")


if __name__ == "__main__":
    build_index()
