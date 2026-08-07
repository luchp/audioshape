import re
import bibtexparser
from pathlib import Path

ALLOWED_COMMANDS = {
    "cite",
    "emph",
    "textbf",
    "textit",
    "text",
    "url",
    "ref",
    "eqref",
}

def _surname(name: str) -> str:
    """Extract a display surname from a single BibTeX author name."""
    name = name.strip()
    if "," in name:
        return name.split(",")[0].strip()
    parts = name.split()
    return parts[-1].strip() if parts else name


def load_bibtex(bib_file_path: Path):
    """Load bib entries keyed by citation key, case-insensitively.

    Each value is a dict with a formatted "author" string (e.g. "Smith and
    Jones" or "Smith et al.") and the raw "year" string, so callers can
    build author-year style citations.
    """
    bib_db = {}
    with open(bib_file_path, "r", encoding="utf-8") as f:
        bib_database = bibtexparser.load(f)
        for entry in bib_database.entries:
            key = entry.get("ID")
            if not key:
                continue
            authors = entry.get("author", "")
            year = entry.get("year", "").strip()
            fmt_author = ""
            if authors:
                names = [_surname(n) for n in authors.split(" and ")]
                if len(names) == 1:
                    fmt_author = names[0]
                elif len(names) == 2:
                    fmt_author = f"{names[0]} and {names[1]}"
                else:
                    fmt_author = f"{names[0]} et al."
            bib_db[key.lower()] = {"author": fmt_author, "year": year}
    return bib_db


def _format_year_list(years: list[str]) -> list[str]:
    """De-duplicate a list of years, sorting numerically where possible."""
    uniq = []
    for y in years:
        if y and y not in uniq:
            uniq.append(y)
    try:
        uniq.sort(key=lambda y: int(re.match(r"\d+", y).group()))
    except (AttributeError, ValueError):
        pass
    return uniq

def extract_and_clean_abstract(tex_file_path: Path, bib_file_path: Path|None) -> str:
    content = tex_file_path.read_text(encoding="utf-8")

    # 1. Find \section*{Abstract}
    start_match = re.search(
        r"\\section\*\s*\{\s*Abstract\s*\}", content, flags=re.IGNORECASE
    )
    if not start_match:
        raise ValueError("Could not find '\\section*{Abstract}' in the file.")

    tail_content = content[start_match.end():]
    # 2. Process line by line; stop at first non-allowed command
    valid_lines = []
    for line in tail_content.splitlines():
        stripped = line.strip()
        if stripped.startswith("%") or not stripped:
            continue

        commands_on_line = re.findall(r"\\([a-zA-Z]+)", stripped)
        disallowed = [
            cmd for cmd in commands_on_line if cmd not in ALLOWED_COMMANDS
        ]
        if disallowed:
            break
        valid_lines.append(line)
    raw_abstract_text = " ".join(valid_lines)
    # 3. Resolve \cite{...} using BibTeX (or remove if no bib provided)
    bib_db = load_bibtex(bib_file_path) if bib_file_path else {}

    def cite_replacer(m):
        keys = [k.strip() for k in m.group(1).split(",")]
        missing = [k for k in keys if k.lower() not in bib_db]
        if missing:
            raise ValueError("Could not resolve citation keys: " + ", ".join(missing))
        entries = [bib_db[k.lower()] for k in keys]

        # Merge consecutive keys that share the same author into one group,
        # so e.g. \cite{steinwolf_a,steinwolf_b} collapses to "(Steinwolf y1, y2)".
        groups = []
        for e in entries:
            if groups and groups[-1]["author"] == e["author"]:
                if e["year"]:
                    groups[-1]["years"].append(e["year"])
            else:
                groups.append(
                    {"author": e["author"], "years": [e["year"]] if e["year"] else []}
                )

        # If the author was already spelled out in the prose right before the
        # \cite (a common LaTeX idiom, e.g. "Steinwolf~\cite{key}"), don't
        # repeat the name -- just append the year(s) to avoid duplication.
        prefix = re.sub(r"[~\s]+$", "", m.string[: m.start()])
        author_already_named = (
            len(groups) == 1
            and groups[0]["author"]
            and prefix.lower().endswith(groups[0]["author"].lower())
        )

        if author_already_named:
            years = _format_year_list(groups[0]["years"])
            return f" ({', '.join(years)})" if years else ""

        parts = []
        for g in groups:
            years = _format_year_list(g["years"])
            if g["author"] and years:
                parts.append(f"{g['author']} {', '.join(years)}")
            elif g["author"]:
                parts.append(g["author"])
            elif years:
                parts.append(", ".join(years))
        return " (" + "; ".join(parts) + ")" if parts else ""

    text = re.sub(r"\\cite\{([^}]+)\}", cite_replacer, raw_abstract_text)

    # 4. Dynamically unwrap content from ALLOWED_COMMANDS (excluding 'cite')
    unwrap_cmds = ALLOWED_COMMANDS - {"cite"}
    pattern_cmds = "|".join(sorted(unwrap_cmds, key=len, reverse=True))

    # Replaces \cmd{content} with content for any command in ALLOWED_COMMANDS
    text = re.sub(rf"\\(?:{pattern_cmds})\{{([^}}]+)\}}", r"\1", text)

    # 5. Clean up inline math and formatting
    text = re.sub(r"\$([^$]+)\$", r"\1", text)
    text = text.replace("---", " — ").replace("--", "–")
    text = text.replace("~", " ")
    text = re.sub(r"\s+([.,;:)])", r"\1", text)  # drop space before punctuation
    text = re.sub(r"\s+", " ", text).strip()

    return text



