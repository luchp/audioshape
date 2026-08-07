#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import extract_abstract

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
CODE_METADATA_FILE = SCRIPTS_DIR / "metadata.json"


class ReleaseAbort(RuntimeError):
    """Raised when release preconditions are not met."""

import platform
import subprocess

@dataclass(frozen=True)
class PlatformMetadata:
    python_version: str
    uv_version: str
    os_version: str

@dataclass(frozen=True)
class CodeMetadata: 
    affiliation: str
    author_family_names: str
    author_given_names: str
    copyright: str
    email: str
    keywords: list[str]
    license_name: str
    message: str
    repository: str
    summary: str
    title: str
    version: str
    website: str


@dataclass(frozen=True)
class PaperMetadata:
    author: str
    keywords: list[str]
    license_name: str
    paper_id: str
    repository: str
    summary: str
    texmain: str
    title: str
    version: str
    year: str


@dataclass(frozen=True)
class ReleaseState:
    local_code_tag_commit: str | None
    remote_code_tag_commit: str | None
    local_paper_tag_commit: str | None
    remote_paper_tag_commit: str | None
    paper_release_exists: bool


@dataclass(frozen=True)
class PublishActions:
    create_code_tag_local: bool
    push_code_tag: bool
    create_paper_tag_local: bool
    push_paper_tag: bool
    create_paper_release: bool
    upload_assets: bool

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and publish a paper release")
    parser.add_argument(
        "-p",
        "--paper-id",
        required=True,
        help="Paper id under scripts/papers/<paper-id>/",
    )
    parser.add_argument(
        "--code-version",
        help="Code release version used for code tag/CITATION.cff (defaults to code_metadata.json)",
    )
    parser.add_argument(
        "--execute-publish",
        action="store_true",
        help="Actually create/push tags and create/upload GitHub release assets",
    )
    return parser.parse_args()


def run_checked(cmd: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        cmd,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        details = "\n".join(
            s for s in [result.stdout.strip(), result.stderr.strip()] if s
        )
        raise ReleaseAbort(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n{details}"
        )
    return result.stdout.strip()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ReleaseAbort(f"Missing required metadata file: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ReleaseAbort(f"Expected JSON object in: {path}")
    return payload


def required_str(meta: dict[str, Any], key: str, source: Path) -> str:
    value = meta.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ReleaseAbort(f"Metadata field '{key}' must be a non-empty string in {source}")
    return value.strip()


def normalize_keywords(value: Any, source: Path) -> list[str]:
    if isinstance(value, list):
        keywords = [k.strip() for k in value if isinstance(k, str) and k.strip()]
    elif isinstance(value, str):
        keywords = [k.strip() for k in value.split(",") if k.strip()]
    else:
        raise ReleaseAbort(
            f"Metadata field 'keywords' must be a comma-separated string or list in {source}"
        )
    if not keywords:
        raise ReleaseAbort(f"Metadata field 'keywords' must not be empty in {source}")
    return keywords

def load_platform_metadata():
    uvv =  run_checked(["uv", "--version"], cwd=REPO_ROOT)
    return PlatformMetadata(
        python_version=platform.python_version(),
        uv_version=uvv.split()[1],
        os_version=f"{platform.system()} {platform.release()} ({platform.machine()})",
    )

def load_code_metadata(path: Path, code_version_override: str | None) -> CodeMetadata:
    data = load_json(path)
    version = code_version_override or required_str(data, "version", path)
    return CodeMetadata(
        affiliation=required_str(data, "affiliation", path),
        author_family_names=required_str(data, "author_family_names", path),
        author_given_names=required_str(data, "author_given_names", path),
        copyright=required_str(data, "copyright", path),
        email=required_str(data, "email", path),
        keywords=normalize_keywords(data.get("keywords"), path),
        license_name=required_str(data, "license", path),
        message=required_str(data, "message", path),
        repository=required_str(data, "repository", path),
        summary=required_str(data, "summary", path),
        title=required_str(data, "title", path),
        version=version,
        website=required_str(data, "website", path),
    )


def load_paper_metadata(code_metadata: CodeMetadata, paper_id: str) -> PaperMetadata:
    path = SCRIPTS_DIR / "papers" / paper_id / "metadata.json"
    data = load_json(path)
    texmain_raw = required_str(data, "texmain", path)
    texmain = texmain_raw if texmain_raw.endswith(".tex") else f"{texmain_raw}.tex"
    author = " ".join([code_metadata.author_given_names,
                       code_metadata.author_family_names])
    return PaperMetadata(
        author=author,
        keywords=normalize_keywords(data.get("keywords"), path),
        license_name=required_str(data, "license", path),
        paper_id=paper_id,
        repository=code_metadata.repository,
        summary=required_str(data, "summary", path),
        texmain=texmain,
        title=required_str(data, "title", path),
        version=required_str(data, "version", path),
        year=required_str(data, "year", path),
    )


def build_tags(code_version: str, paper_id: str, paper_version: str) -> tuple[str, str]:
    code_tag = f"v{code_version}" if not code_version.startswith("v") else code_version
    paper_tag = f"paper-{paper_id}-v{paper_version.lstrip('v')}"
    return code_tag, paper_tag


def ensure_repo_clean(force: bool) -> None:
    status = run_checked(["git", "status", "--porcelain"], cwd=REPO_ROOT)
    if status:
        msg = f"Repository has outstanding changes. Commit or stash before running release.\n{status}"
        if force:
            raise ReleaseAbort(msg)
        else:
            print(msg, flush=True)


def head_sha() -> str:
    return run_checked(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)


def local_tag_commit(tag: str) -> str | None:
    result = subprocess.run(
        ["git", "rev-list", "-n", "1", tag],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


def remote_tag_commit(tag: str) -> str | None:
    output = run_checked(
        ["git", "ls-remote", "--tags", "origin", f"refs/tags/{tag}", f"refs/tags/{tag}^{{}}"],
        cwd=REPO_ROOT,
    )
    if not output:
        return None
    peeled = None
    direct = None
    for line in output.splitlines():
        if not line.strip():
            continue
        sha, ref = line.split("\t", 1)
        if ref.endswith("^{}"):
            peeled = sha
        elif ref.endswith(f"/{tag}"):
            direct = sha
    return peeled or direct


def github_request(
    method: str,
    url: str,
    *,
    token: str,
    payload: dict[str, Any] | None = None,
    content_type: str = "application/json",
) -> tuple[int, dict[str, Any]]:
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "python-urllib",
    }
    body: bytes | None = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = content_type
    req = urllib.request.Request(url, method=method, headers=headers, data=body)
    try:
        with urllib.request.urlopen(req) as response:
            status = response.status
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        if exc.code == 404:
            return 404, {}
        raise ReleaseAbort(f"GitHub API {method} {url} failed ({exc.code}): {raw}") from exc
    payload_dict = json.loads(raw) if raw else {}
    if not isinstance(payload_dict, dict):
        raise ReleaseAbort(f"Unexpected GitHub response from {url}")
    return status, payload_dict


def github_get_json(url: str, *, token: str) -> tuple[int, Any]:
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "python-urllib",
    }
    req = urllib.request.Request(url, method="GET", headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            status = response.status
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        if exc.code == 404:
            return 404, None
        raise ReleaseAbort(f"GitHub API GET {url} failed ({exc.code}): {raw}") from exc
    return status, (json.loads(raw) if raw else None)


def github_release_exists(repository: str, tag: str, token: str) -> bool:
    # The /releases/tags/{tag} endpoint has been observed returning server-side
    # 503s, so page through the (reliable) list endpoint and match on tag_name.
    page = 1
    while True:
        url = (
            f"https://api.github.com/repos/{repository}/releases"
            f"?per_page=100&page={page}"
        )
        status, data = github_get_json(url, token=token)
        if status == 404:
            return False
        if status != 200:
            raise ReleaseAbort(
                f"Unexpected status while listing releases for {repository}: {status}"
            )
        if not isinstance(data, list) or not data:
            return False
        for release in data:
            if isinstance(release, dict) and release.get("tag_name") == tag:
                return True
        if len(data) < 100:
            return False
        page += 1


def collect_release_state(
    repository: str,
    code_tag: str,
    paper_tag: str,
    token: str,
) -> ReleaseState:
    return ReleaseState(
        local_code_tag_commit=local_tag_commit(code_tag),
        remote_code_tag_commit=remote_tag_commit(code_tag),
        local_paper_tag_commit=local_tag_commit(paper_tag),
        remote_paper_tag_commit=remote_tag_commit(paper_tag),
        paper_release_exists=github_release_exists(repository, paper_tag, token),
    )


def resolve_tag_commit(local_commit: str | None, remote_commit: str | None, tag_name: str) -> str | None:
    if local_commit and remote_commit and local_commit != remote_commit:
        raise ReleaseAbort(
            f"Tag '{tag_name}' differs between local ({local_commit}) and origin ({remote_commit})."
        )
    return remote_commit or local_commit


def determine_publish_actions(
    *,
    head_commit: str,
    state: ReleaseState,
    execute_publish: bool,
) -> PublishActions:
    code_commit = resolve_tag_commit(
        state.local_code_tag_commit, state.remote_code_tag_commit, "code"
    )
    paper_commit = resolve_tag_commit(
        state.local_paper_tag_commit, state.remote_paper_tag_commit, "paper"
    )

    if state.paper_release_exists and paper_commit is None:
        raise ReleaseAbort("Paper release exists but paper tag does not exist.")
    if state.paper_release_exists and code_commit is None:
        raise ReleaseAbort("Paper release exists but code tag does not exist.")

    for label, commit in (("code", code_commit), ("paper", paper_commit)):
        if commit is not None and commit != head_commit:
            raise ReleaseAbort(
                f"Existing {label} tag points to {commit}, but HEAD is {head_commit}. "
                "Refusing to overwrite existing release identity."
            )

    if state.paper_release_exists and paper_commit != head_commit:
        raise ReleaseAbort(
            "Paper release already exists for a different commit. Refusing to continue."
        )

    if not execute_publish:
        return PublishActions(
            create_code_tag_local=False,
            push_code_tag=False,
            create_paper_tag_local=False,
            push_paper_tag=False,
            create_paper_release=False,
            upload_assets=False,
        )

    code_remote_missing = state.remote_code_tag_commit is None
    paper_remote_missing = state.remote_paper_tag_commit is None
    code_local_missing = state.local_code_tag_commit is None
    paper_local_missing = state.local_paper_tag_commit is None
    release_missing = not state.paper_release_exists

    return PublishActions(
        create_code_tag_local=code_remote_missing and code_local_missing,
        push_code_tag=code_remote_missing,
        create_paper_tag_local=paper_remote_missing and paper_local_missing,
        push_paper_tag=paper_remote_missing,
        create_paper_release=release_missing,
        upload_assets=release_missing,
    )


def describe_publish_actions(actions: PublishActions, *, code_tag: str, paper_tag: str) -> list[str]:
    planned: list[str] = []
    if actions.create_code_tag_local:
        planned.append(f"create local annotated tag {code_tag}")
    if actions.push_code_tag:
        planned.append(f"push tag {code_tag} to origin")
    if actions.create_paper_tag_local:
        planned.append(f"create local annotated tag {paper_tag}")
    if actions.push_paper_tag:
        planned.append(f"push tag {paper_tag} to origin")
    if actions.create_paper_release:
        planned.append(f"create GitHub release for {paper_tag}")
    if actions.upload_assets:
        planned.append("upload paper PDF and PROVENANCE.txt as release assets")
    if not planned:
        planned.append("no publish actions are required; tags and release already exist")
    return planned


def run_tests() -> None:
    run_checked(["uv", "run", "pytest"], cwd=REPO_ROOT)


def run_figures(paper_id: str) -> None:
    run_checked(
        ["uv", "run", "--extra", "examples", "python", "scripts/make_figures.py", "-p", paper_id],
        cwd=REPO_ROOT,
    )


def run_paper_figures(paper_script: Path) -> None:
    run_checked(
        ["uv", "run", "--extra", "examples", "python", str(paper_script)],
        cwd=REPO_ROOT,
    )


def build_paper_pdf(paper: PaperMetadata) -> Path:
    paper_dir = REPO_ROOT / "papers" / paper.paper_id
    tex_path = paper_dir / paper.texmain
    if not tex_path.is_file():
        raise ReleaseAbort(f"TeX entry file does not exist: {tex_path}")
    tex_file_name = tex_path.name
    tex_stem = tex_path.stem
    pdf_path = paper_dir / f"{tex_stem}.pdf"
    # delete to pdf if it exists
    pdf_path.unlink(missing_ok=True)
    # run latex
    pdflatex = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_file_name]
    run_checked(pdflatex, cwd=paper_dir)
    run_checked(["bibtex", tex_stem], cwd=paper_dir)
    run_checked(pdflatex, cwd=paper_dir)
    run_checked(pdflatex, cwd=paper_dir)
    # check that we generated a pdf
    if not pdf_path.is_file() or pdf_path.stat().st_size == 0:
        raise ReleaseAbort(f"Expected PDF was not generated: {pdf_path}")
    # build the release path
    last_name = paper.author.split()[-1].capitalize()
    title = "".join(t.capitalize() for t in paper.title.split())
    pdf_out_path = paper_dir / f"{last_name}_{title}_v{paper.version}.pdf"
    pdf_path.copy(pdf_out_path)
    return pdf_out_path

def write_citation_cff(code: CodeMetadata, released_at: datetime) -> Path:
    keywords_block = "\n".join(f'  - "{k}"' for k in code.keywords)
    content = (
        "cff-version: 1.2.0\n"
        f'message: "{code.message}"\n'
        f'title: "{code.title}"\n'
        f'version: "{code.version}"\n'
        f'date-released: "{released_at.date().isoformat()}"\n'
        "authors:\n"
        f'  - family-names: "{code.author_family_names}"\n'
        f'    given-names: "{code.author_given_names}"\n'
        f'    affiliation: "{code.affiliation}"\n'
        "identifiers:\n"
        "  - type: url\n"
        f'    value: "https://github.com/{code.repository}"\n'
        f'license: "{code.license_name}"\n'
        f'abstract: "{code.summary}"\n'
        "keywords:\n"
        f"{keywords_block}\n"
    )
    path = REPO_ROOT / "CITATION.cff"
    path.write_text(content, encoding="utf-8")
    return path

def write_abstract(paper: PaperMetadata) ->Path:
    paper_dir = REPO_ROOT / "papers" / paper.paper_id
    tex_path = paper_dir / paper.texmain
    bib_path = tex_path.with_name("refs.bib")
    if not tex_path.is_file():
        raise ReleaseAbort(f"TeX entry file does not exist: {tex_path}")
    if not bib_path.is_file():
        raise ReleaseAbort(f"BIB file does not exist: {bib_path}")
    abstract = extract_abstract.extract_and_clean_abstract(tex_path, bib_path)
    abs_path = tex_path.with_name(tex_path.stem + "_abstract.txt")
    abs_path.write_text(abstract, encoding="utf-8")
    return abs_path

def write_provenance(
    *,
    platfrm: PlatformMetadata,
    paper: PaperMetadata,
    code_tag: str,
    paper_tag: str,
    commit: str,
    released_at: datetime,
    pdf_path: Path,
    command: list[str],
) -> Path:
    paper_dir = REPO_ROOT / "papers" / paper.paper_id
    path = paper_dir / "PROVENANCE.txt"
    body = (
        f"paper_id: {paper.paper_id}\n"
        f"title: {paper.title}\n"
        f"paper_version: {paper.version}\n"
        f"code_tag: {code_tag}\n"
        f"paper_tag: {paper_tag}\n"
        f"repository: {paper.repository}\n"
        f"commit: {commit}\n"
        f"built_utc: {released_at.isoformat()}\n"
        f"build_command: {' '.join(command)}\n"
        f"paper_pdf: {pdf_path.relative_to(REPO_ROOT)}\n"
        f"python: {platfrm.python_version}\n"
        f"uv: {platfrm.uv_version}\n"
        f"os: {platfrm.os_version}\n"
    )
    path.write_text(body, encoding="utf-8")
    return path

def create_annotated_tag(tag: str, message: str) -> None:
    run_checked(["git", "tag", "-a", tag, "-m", message], cwd=REPO_ROOT)

def push_tag(tag: str) -> None:
    run_checked(["git", "push", "origin", tag], cwd=REPO_ROOT)


def create_github_release(repository: str, tag: str, body: str, token: str) -> dict[str, Any]:
    url = f"https://api.github.com/repos/{repository}/releases"
    payload = {
        "tag_name": tag,
        "name": tag,
        "body": body,
        "draft": True,
        "prerelease": False,
    }
    status, data = github_request("POST", url, token=token, payload=payload)
    if status != 201:
        raise ReleaseAbort(f"Failed to create release for tag {tag}. Status: {status}")
    return data


def upload_release_asset(upload_url_template: str, asset_path: Path, token: str) -> None:
    upload_url = upload_url_template.split("{", 1)[0]
    query = urllib.parse.urlencode({"name": asset_path.name})
    url = f"{upload_url}?{query}"
    content_type, _ = mimetypes.guess_type(str(asset_path))
    if not content_type:
        content_type = "application/octet-stream"

    data = asset_path.read_bytes()
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": content_type,
    }
    req = urllib.request.Request(url, method="POST", headers=headers, data=data)
    try:
        with urllib.request.urlopen(req) as response:
            if response.status not in (201, 200):
                raise ReleaseAbort(
                    f"Asset upload failed for {asset_path.name} with status {response.status}"
                )
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise ReleaseAbort(
            f"GitHub asset upload failed for {asset_path.name} ({exc.code}): {details}"
        ) from exc

def ensure_required_tools() -> None:
    print("Checking required tools...", flush=True)
    for tool in ("git", "uv", "pdflatex", "bibtex"):
        print(f" - {tool}", flush=True)
        result = subprocess.run(
            [tool, "--version"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise ReleaseAbort(f"Required tool is not available: {tool}")
    print("Required tools available.", flush=True)

def load_secrets():
    return load_json(SCRIPTS_DIR / "secrets.json")

def main() -> None:
    args = parse_args()
    released_at = datetime.now(timezone.utc)
    command = ["python", "scripts/publish_release.py", "-p", args.paper_id]
    if args.code_version:
        command.extend(["--code-version", args.code_version])
    if args.execute_publish:
        command.append("--execute-publish")

    print("Starting release workflow...", flush=True)
    ensure_required_tools()
    print("Checking repository state...", flush=True)
    ensure_repo_clean(args.execute_publish)
    print("Reading commit hash...", flush=True)
    commit = head_sha()
    print(f"Current commit: {commit}", flush=True)

    print("Loading metadata...", flush=True)
    platfrm = load_platform_metadata()
    code = load_code_metadata(CODE_METADATA_FILE, args.code_version)
    paper = load_paper_metadata(code, args.paper_id)
    #
    # get current state and print planned actions
    secrets = load_secrets()
    token = secrets["GITHUB_TOKEN"]
    if not token:
        raise ReleaseAbort("GITHUB_TOKEN must be set to check release existence safely.")

    print("Determining release tags and state...", flush=True)
    code_tag, paper_tag = build_tags(code.version, args.paper_id, paper.version)
    state = collect_release_state(
        repository=paper.repository,
        code_tag=code_tag,
        paper_tag=paper_tag,
        token=token,
    )
    actions = determine_publish_actions(
        head_commit=commit,
        state=state,
        execute_publish=args.execute_publish,
    )
    if not args.execute_publish:
        planned_actions = determine_publish_actions(
            head_commit=commit,
            state=state,
            execute_publish=True,
        )
        print("Dry run: --execute-publish not set. Planned publish actions:")
        for action in describe_publish_actions(
            planned_actions, code_tag=code_tag, paper_tag=paper_tag
        ):
            print(f" - {action}")
    # Final checks, we can't rebuild the figures because they are not idempotent.
    # some figures/tables need benchmarks, and there are always (minor) timing differences
    # so we only check if the scrip is there, build_paper_pdf will fail if the output is missing
    print("Running tests...", flush=True)
    paper_script = SCRIPTS_DIR / "papers" / args.paper_id / "make_figures.py"
    if not paper_script.is_file():
        raise ReleaseAbort(f"Missing paper figure script: {paper_script}")
    run_tests()
    print("Tests complete.", flush=True)
    print("Building paper PDF...", flush=True)
    pdf_path = build_paper_pdf(paper)
    abs_path = write_abstract(paper)
    print(f"Built PDF: {pdf_path}", flush=True)
    print(f"Wrote abstract: {abs_path}", flush=True)
    print("Writing provenance and citation metadata...", flush=True)
    prov_path = write_provenance(
        platfrm=platfrm,
        paper=paper,
        code_tag=code_tag,
        paper_tag=paper_tag,
        commit=commit,
        released_at=released_at,
        pdf_path=pdf_path,
        command=command,
    )
    # Actually create and push tags and release
    if actions.create_code_tag_local:
        print(f"Creating local code tag {code_tag}...", flush=True)
        create_annotated_tag(code_tag, f"Code release {code_tag} ({commit})")
    if actions.push_code_tag:
        print(f"Pushing code tag {code_tag}...", flush=True)
        push_tag(code_tag)
    if actions.create_paper_tag_local:
        print(f"Creating local paper tag {paper_tag}...", flush=True)
        create_annotated_tag(paper_tag, f"Paper release {paper_tag} ({commit})")
    if actions.push_paper_tag:
        print(f"Pushing paper tag {paper_tag}...", flush=True)
        push_tag(paper_tag)
    if actions.create_paper_release:
        print(f"Creating GitHub release {paper_tag}...", flush=True)
        release_data = create_github_release(
            repository=paper.repository,
            tag=paper_tag,
            body=prov_path.read_text(encoding="utf-8"),
            token=token,
        )
        if actions.upload_assets:
            print("Uploading release assets...", flush=True)
            upload_release_asset(release_data["upload_url"], pdf_path, token)
            upload_release_asset(release_data["upload_url"], prov_path, token)
            print("Release assets uploaded.", flush=True)

    print("Release workflow complete.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except ReleaseAbort as exc:
        raise SystemExit(f"ABORTING RELEASE: {exc}") from exc
