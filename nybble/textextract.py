"""Text extraction from the file formats personal writing actually lives in.

Every format degrades gracefully: .docx and .odt are parsed with stdlib zipfile
plus a small XML walk, HTML with html.parser, PDF only if pypdf is installed.
The importer must work on a machine where `pip install` is not an option.
"""

from __future__ import annotations

import io
import re
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".org", ".text", ".log"}
DOC_SUFFIXES = {".docx", ".odt", ".rtf", ".pdf", ".tex", ".html", ".htm", ".epub"}
ALL_SUFFIXES = TEXT_SUFFIXES | DOC_SUFFIXES

MAX_BYTES = 8 * 1024 * 1024  # skip anything bigger; it is not prose


def extract(path: Path) -> Optional[str]:
    """Return the readable text of `path`, or None if unsupported/unreadable."""
    try:
        if not path.is_file() or path.stat().st_size > MAX_BYTES:
            return None
    except OSError:
        return None

    suffix = path.suffix.lower()
    try:
        if suffix in TEXT_SUFFIXES:
            return read_text(path)
        if suffix == ".docx":
            return from_docx(path)
        if suffix == ".odt":
            return from_odt(path)
        if suffix == ".rtf":
            return from_rtf(path)
        if suffix == ".pdf":
            return from_pdf(path)
        if suffix in (".html", ".htm"):
            return from_html(read_text(path) or "")
        if suffix == ".tex":
            return from_latex(read_text(path) or "")
        if suffix == ".epub":
            return from_epub(path)
    except Exception:
        return None
    return None


def read_text(path: Path) -> Optional[str]:
    """Decode a text file, trying the encodings real-world files actually use."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    return decode(raw)


def decode(raw: bytes) -> Optional[str]:
    if not raw:
        return None
    for enc in ("utf-8", "utf-8-sig", "utf-16"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    try:
        import chardet  # optional
        guess = chardet.detect(raw[:100_000])
        if guess.get("encoding"):
            return raw.decode(guess["encoding"], "replace")
    except Exception:
        pass
    return raw.decode("latin-1", "replace")


# --- OOXML / ODF --------------------------------------------------------------

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def from_docx(path: Path) -> Optional[str]:
    with zipfile.ZipFile(path) as z:
        if "word/document.xml" not in z.namelist():
            return None
        root = ET.fromstring(z.read("word/document.xml"))
    paras = []
    for p in root.iter(f"{_W}p"):
        runs = [t.text or "" for t in p.iter(f"{_W}t")]
        line = "".join(runs).strip()
        if line:
            paras.append(line)
    return "\n\n".join(paras) or None


def from_odt(path: Path) -> Optional[str]:
    with zipfile.ZipFile(path) as z:
        if "content.xml" not in z.namelist():
            return None
        root = ET.fromstring(z.read("content.xml"))
    text_ns = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}"
    paras = []
    for p in root.iter():
        if p.tag in (f"{text_ns}p", f"{text_ns}h"):
            line = "".join(p.itertext()).strip()
            if line:
                paras.append(line)
    return "\n\n".join(paras) or None


def from_epub(path: Path) -> Optional[str]:
    chunks = []
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if name.lower().endswith((".xhtml", ".html", ".htm")):
                text = from_html(decode(z.read(name)) or "")
                if text:
                    chunks.append(text)
    return "\n\n".join(chunks) or None


# --- RTF ----------------------------------------------------------------------

_RTF_CONTROL = re.compile(r"\\[a-zA-Z]+-?\d* ?|\\'[0-9a-fA-F]{2}|[{}]")


def from_rtf(path: Path) -> Optional[str]:
    try:
        from striprtf.striprtf import rtf_to_text  # optional, much better
        return rtf_to_text(read_text(path) or "", errors="ignore") or None
    except Exception:
        pass
    raw = read_text(path) or ""
    text = _RTF_CONTROL.sub(" ", raw)
    return clean(text) or None


# --- PDF ----------------------------------------------------------------------

def from_pdf(path: Path, max_pages: int = 80) -> Optional[str]:
    try:
        from pypdf import PdfReader
    except ImportError:
        return None
    try:
        reader = PdfReader(str(path))
        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")
            except Exception:
                return None
        pages = []
        for page in reader.pages[:max_pages]:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                continue
        return clean("\n\n".join(pages)) or None
    except Exception:
        return None


# --- HTML ---------------------------------------------------------------------

class _HTMLText(HTMLParser):
    SKIP = {"script", "style", "head", "nav", "noscript", "svg"}
    BREAK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
             "blockquote", "section", "article"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip_depth += 1
        elif tag in self.BREAK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag in self.BREAK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip_depth:
            self.parts.append(data)

    def text(self) -> str:
        return clean("".join(self.parts))


def from_html(html: str) -> Optional[str]:
    if not html:
        return None
    parser = _HTMLText()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        return None
    return parser.text() or None


# --- LaTeX --------------------------------------------------------------------

_TEX_ENV_DROP = re.compile(
    r"\\begin\{(equation|align|figure|table|tikzpicture|lstlisting|verbatim|"
    r"minted|algorithm)\*?\}.*?\\end\{\1\*?\}", re.DOTALL)
_TEX_COMMENT = re.compile(r"(?<!\\)%.*$", re.MULTILINE)
_TEX_CMD_ARG = re.compile(r"\\(?:emph|textbf|textit|texttt|section|subsection|"
                          r"subsubsection|title|chapter|paragraph)\*?\{([^{}]*)\}")
_TEX_CMD = re.compile(r"\\[a-zA-Z@]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?")


def from_latex(tex: str) -> Optional[str]:
    if not tex:
        return None
    body = tex
    if "\\begin{document}" in body:
        body = body.split("\\begin{document}", 1)[1]
    body = body.split("\\end{document}", 1)[0]
    body = _TEX_COMMENT.sub("", body)
    body = _TEX_ENV_DROP.sub(" ", body)
    body = _TEX_CMD_ARG.sub(r"\1", body)
    body = _TEX_CMD.sub(" ", body)
    body = body.replace("{", "").replace("}", "").replace("\\\\", "\n")
    return clean(body) or None


# --- markdown -----------------------------------------------------------------

_MD_CODE = re.compile(r"```.*?```", re.DOTALL)
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MD_IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_FRONTMATTER = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)


def strip_markdown(md: str, keep_code: bool = False) -> str:
    """Reduce markdown to the prose. Used for notes where the prose is the signal."""
    if not md:
        return ""
    out = _FRONTMATTER.sub("", md)
    if not keep_code:
        out = _MD_CODE.sub(" ", out)
    out = _MD_IMG.sub(" ", out)
    out = _MD_LINK.sub(r"\1", out)
    out = re.sub(r"^#{1,6}\s*", "", out, flags=re.MULTILINE)
    out = re.sub(r"[*_`~]{1,3}", "", out)
    return clean(out)


def frontmatter_title(md: str) -> Optional[str]:
    m = re.search(r"^title:\s*[\"']?(.+?)[\"']?\s*$", md[:2000], re.MULTILINE)
    if m:
        return m.group(1).strip()
    m = re.search(r"^#\s+(.+)$", md, re.MULTILINE)
    return m.group(1).strip() if m else None


# --- shared -------------------------------------------------------------------

_WS = re.compile(r"[ \t\r\f\v]+")
_BLANKS = re.compile(r"\n{3,}")


def clean(text: str) -> str:
    """Normalize whitespace and unicode without destroying paragraph structure."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ").replace("\ufeff", "")
    # Drop control characters that survive bad decodes.
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t"
                   or unicodedata.category(ch)[0] != "C")
    text = _WS.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _BLANKS.sub("\n\n", text)
    return text.strip()


def looks_like_prose(text: str, min_words: int = 20) -> bool:
    """Filter out config files, logs, and minified junk that match .txt/.md."""
    if not text:
        return False
    words = text.split()
    if len(words) < min_words:
        return False
    # Prose has spaces and sentence punctuation; a lockfile or CSV does not.
    letters = sum(ch.isalpha() for ch in text)
    if letters / max(len(text), 1) < 0.5:
        return False
    avg_word = sum(len(w) for w in words) / len(words)
    return 2.0 <= avg_word <= 18.0
