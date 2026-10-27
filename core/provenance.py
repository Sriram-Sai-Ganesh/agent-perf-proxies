"""Record the code, settings, environment, and start/end times of a run.

``start`` writes the initial record and a patch of tracked and untracked changes.
``finish`` adds the end time. Git must be available in a checkout with a commit.
Stages sharing an output folder should use distinct provenance filenames.
"""

import getpass
import importlib.metadata
import json
import os
import socket
import subprocess
import sys
from datetime import UTC, datetime


def git(*args: str, verbatim: bool = False) -> str:
    """Run a git command in the repository holding this file.

    Args:
        *args: The arguments after ``git``.
        verbatim: Give the output as git wrote it, for a diff ``git apply`` must accept and a status list whose
            first line's leading space means a change is not staged.

    Returns:
        Its output, stripped unless ``verbatim``.

    Raises:
        SystemExit: When git is missing or the command fails: a record naming no commit cannot say what made the
            files it describes, so the run stops rather than write one.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        done = subprocess.run(["git", "-C", here, *args], capture_output=True, text=True, check=False)
    except OSError as error:
        raise SystemExit(f"provenance needs git, to record the code that made these files: {error}") from error
    if done.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed in {here}: {done.stderr.strip()}")
    return done.stdout if verbatim else done.stdout.strip()


PACKAGES = ("vllm", "scienceworld", "alfworld", "jdk4py", "httpx", "pyarrow", "trl", "peft")
"""Packages whose versions can change what a run produces, recorded where installed."""


def environment() -> dict:
    """Say which Python ran, with which packages, launched how.

    Returns:
        ``python``, the interpreter's path, which tells a container image from a uv environment; ``packages``,
        the installed version of each of ``PACKAGES`` present; and ``launch``, the ``LAUNCH_*`` variables the
        cluster scripts set: the full container command, the user and the host; and ``slurm``, every ``SLURM_*``
        variable of the job.
    """
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    launch = {k: v for k, v in os.environ.items() if k.startswith("LAUNCH_")}
    slurm = {k: v for k, v in sorted(os.environ.items()) if k.startswith("SLURM_")}
    return {"python": sys.executable, "packages": versions, "launch": launch, "slurm": slurm}


def start(out_dir: str, settings: dict, name: str = "provenance.json") -> dict:
    """Write the record at the start of a run.

    Args:
        out_dir: The folder the run writes into.
        settings: The arguments and settings the run was given.
        name: The file to write, for a folder more than one step writes into.

    Returns:
        The record, to be handed to ``finish``.
    """
    record = {
        "user": getpass.getuser(),
        "host": socket.gethostname(),
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "status": git("status", "--porcelain", verbatim=True),
        "command": " ".join(sys.argv),
        **environment(),
        "settings": settings,
        "started": datetime.now(UTC).isoformat(),
        "finished": None,
        "file": name,
    }
    _write(out_dir, record, name)
    # the uncommitted changes themselves, beside the record, so the files can be traced to the code that made them
    with open(os.path.join(out_dir, name.removesuffix(".json") + ".diff"), "wb") as f:
        f.write(git("diff", "HEAD", "--binary", verbatim=True).encode())
        f.write(untracked_files())
    return record


def untracked_files() -> bytes:
    """Write every file git does not track, and does not ignore, as a diff that creates it.

    ``git diff HEAD`` leaves such files out, so a script not yet added to git, which may be the one that made the
    files a record describes, would be in no diff.

    Returns:
        One new-file diff per file, paths from the repository root, in the form ``git apply`` takes.

    Raises:
        SystemExit: When git fails on a file.
    """
    top = git("rev-parse", "--show-toplevel")
    names = git("ls-files", "--others", "--exclude-standard", "--full-name", "-z", ":/", verbatim=True).split("\0")
    diffs = []
    for name in filter(None, names):
        # git diff --no-index exits 1 when the two differ, which a new file always does
        done = subprocess.run(
            ["git", "-C", top, "diff", "--no-index", "--binary", "--", "/dev/null", name],
            capture_output=True,
            check=False,
        )
        if done.returncode not in (0, 1):
            raise SystemExit(f"git diff of the untracked file {name} failed: {done.stderr.decode().strip()}")
        diffs.append(done.stdout)
    return b"".join(diffs)


def finish(out_dir: str, record: dict) -> None:
    """Write the record again at the end of a run, with the finish time.

    Args:
        out_dir: The folder the run writes into.
        record: What ``start`` returned.
    """
    record["finished"] = datetime.now(UTC).isoformat()
    _write(out_dir, record, str(record.get("file", "provenance.json")))


def _write(out_dir: str, record: dict, name: str) -> None:
    """Write the record.

    Args:
        out_dir: The folder to write into.
        record: The record.
        name: The file to write.
    """
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, name), "w") as f:
        json.dump(record, f, indent=2, sort_keys=True)
        f.write("\n")
