# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Execute the docs notebooks and refresh their stored outputs in place."""

from __future__ import annotations

import argparse
import datetime
import glob
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError, DeadKernelError
from nbformat import NotebookNode

REPO_ROOT = Path(__file__).parent.parent
DOCS_DIR = REPO_ROOT / "docs"
NOTEBOOK_DIRS = ["explanations", "how-to-guides", "tutorials"]
DEFAULT_TIMEOUT = 300
KERNELSPEC = {
    "display_name": "Python 3 (ipykernel)",
    "language": "python",
    "name": "python3",
}
THREAD_ENV_VARS = [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "RAYON_NUM_THREADS",
]
N_SLOWEST = 10


@dataclass
class Result:
    """Outcome of executing one notebook."""

    path: Path
    status: str  # "ok", "failed", or "skipped"
    seconds: float
    detail: str = ""  # failure summary, or reason for skipping


def display_path(path: Path) -> str:
    """Return the path relative to the repo root, for printing."""
    return os.path.relpath(path, REPO_ROOT)


def is_excluded(path: Path) -> bool:
    """Return whether a path inside the repo lies in a hidden or private dir.

    This keeps the copies nbsphinx generates under docs/_build, and stale ones
    under .ipynb_checkpoints, out of the selection. Paths outside the repo are
    never excluded, so an explicitly given one is always honored.
    """
    try:
        relative = path.relative_to(REPO_ROOT)
    except ValueError:
        return False
    return any(part.startswith((".", "_")) for part in relative.parts)


def discover_notebooks(patterns: list[str]) -> list[Path]:
    """Return the notebooks selected by the given paths, dirs, or globs."""
    if not patterns:
        patterns = [str(DOCS_DIR / name) for name in NOTEBOOK_DIRS]
    paths: set[Path] = set()
    for pattern in patterns:
        path = Path(pattern)
        if path.is_file():
            matches = [path]
        elif path.is_dir():
            matches = sorted(path.rglob("*.ipynb"))
        else:
            matches = [Path(match) for match in glob.glob(pattern, recursive=True)]
            matches = [match for match in matches if match.suffix == ".ipynb"]
            if not matches:
                raise FileNotFoundError(f"No notebooks matched {pattern}")
        paths.update(
            resolved
            for resolved in (match.resolve() for match in matches)
            if not is_excluded(resolved)
        )
    return sorted(paths, key=display_path)


def clear_outputs(nb: NotebookNode) -> int:
    """Clear stored output state; return the number of code cells.

    Widget state is stored output too, and is not re-created because the client
    runs with store_widget_state=False.
    """
    nb.metadata.pop("widgets", None)
    n_code_cells = 0
    for cell in nb.cells:
        # Wall-clock timestamps left behind by tools that run with
        # record_timing=True are pure diff noise; drop them.
        cell.get("metadata", {}).pop("execution", None)
        if cell.cell_type != "code":
            continue
        n_code_cells += 1
        cell.outputs = []
        cell.execution_count = None
    return n_code_cells


def canonicalize_metadata(nb: NotebookNode) -> None:
    """Pin the kernelspec so it does not vary between contributors.

    language_info is left alone: the kernel reports it during execution, and it
    records which interpreter actually produced the outputs.
    """
    nb.metadata["kernelspec"] = dict(KERNELSPEC)


def write_notebook(nb: NotebookNode, path: Path) -> None:
    """Write the notebook, via a temp file so an interrupt cannot truncate it."""
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f"{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as file:
        temp_path = Path(file.name)
        try:
            nbformat.write(nb, file)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
    os.replace(temp_path, path)


def available_cpus() -> int:
    """Return the number of CPUs this process may actually run on.

    os.cpu_count() reports the whole machine, which overshoots when the process
    is restricted to a subset of the CPUs, as in a container.
    """
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def thread_limits_in_env() -> str:
    """Return the thread-limit variables already set, for reporting."""
    return ", ".join(
        f"{name}={os.environ[name]}" for name in THREAD_ENV_VARS if name in os.environ
    )


def thread_env(jobs: int) -> dict[str, str]:
    """Return thread-limit environment variables to share cores between jobs.

    ffsim's Rayon kernels and the BLAS behind NumPy, SciPy, and pyscf each grab
    every core by default, so running notebooks concurrently without limiting
    them oversubscribes the machine.

    If any of these variables is already set, the caller has a thread policy
    and none are touched. Filling in only the gaps would mean running some
    libraries at a different width than the caller asked for, and reporting a
    per-kernel thread count that is not the effective one.
    """
    if jobs <= 1 or thread_limits_in_env():
        return {}
    threads = max(1, available_cpus() // jobs)
    return {name: str(threads) for name in THREAD_ENV_VARS}


def execute_notebook(path: Path, timeout: int, dry_run: bool) -> Result:
    """Execute one notebook in a fresh kernel and rewrite it in place."""
    nb = nbformat.read(path, as_version=4)
    n_code_cells = clear_outputs(nb)
    canonicalize_metadata(nb)
    if not n_code_cells:
        # Nothing to execute, but the metadata is still brought into line.
        if not dry_run:
            write_notebook(nb, path)
        return Result(path, "skipped", 0.0, "no code cells")

    # Recorded as execution proceeds so that a failure can name the cell.
    current_cell = 0

    def note_cell(cell: NotebookNode, cell_index: int) -> None:
        nonlocal current_cell
        current_cell = cell_index

    client = NotebookClient(
        nb,
        timeout=timeout or None,
        allow_errors=False,
        record_timing=False,
        store_widget_state=False,
        # Merge consecutive prints into one stream output, the way Jupyter
        # stores them, instead of one output per flush.
        coalesce_streams=True,
        resources={"metadata": {"path": str(path.parent)}},
        on_cell_execute=note_cell,
        # Keep the kernel's own startup chatter off the progress lines. Output
        # produced by the cells themselves is captured by nbclient regardless.
        extra_arguments=["--IPKernelApp.log_level=ERROR"],
    )
    start = time.monotonic()
    try:
        client.execute()
    except (CellExecutionError, CellTimeoutError, DeadKernelError) as error:
        elapsed = time.monotonic() - start
        detail = f"cell {current_cell}: {summarize_error(error)}"
        return Result(path, "failed", elapsed, detail)
    elapsed = time.monotonic() - start
    if not dry_run:
        write_notebook(nb, path)
    return Result(path, "ok", elapsed)


def summarize_error(error: Exception) -> str:
    """Return a one-line summary of a notebook execution failure."""
    if isinstance(error, CellExecutionError):
        return f"{error.ename}: {error.evalue}".strip().splitlines()[0]
    lines = [line for line in str(error).splitlines() if line.strip()]
    return lines[0] if lines else type(error).__name__


def format_progress(index: int, total: int, path: Path) -> str:
    """Return the leading part of a notebook's progress line."""
    return f"[{index:>{len(str(total))}}/{total}] {display_path(path)} ... "


def format_outcome(result: Result) -> str:
    """Return the trailing part of a notebook's progress line."""
    label = {"ok": "ok", "failed": "FAILED", "skipped": "skipped"}[result.status]
    detail = result.detail if result.status == "skipped" else f"{result.seconds:.1f}s"
    return f"{label} ({detail})"


def run_serial(paths: list[Path], args: argparse.Namespace) -> list[Result]:
    """Execute the notebooks one at a time, printing progress as they run."""
    results = []
    for index, path in enumerate(paths, start=1):
        print(format_progress(index, len(paths), path), end="", flush=True)
        result = execute_notebook(path, args.timeout, args.dry_run)
        results.append(result)
        print(format_outcome(result), flush=True)
        if result.status == "failed" and args.fail_fast:
            print("Stopping early because --fail-fast was given.")
            break
    return results


def run_parallel(paths: list[Path], args: argparse.Namespace) -> list[Result]:
    """Execute the notebooks concurrently, reporting each as it finishes.

    Unlike the serial runner this cannot print a line before starting a
    notebook, since several are in flight at once.
    """
    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [
            executor.submit(execute_notebook, path, args.timeout, args.dry_run)
            for path in paths
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                format_progress(index, len(paths), result.path)
                + format_outcome(result),
                flush=True,
            )
    return results


def print_summary(results: list[Result], elapsed: float) -> None:
    """Print the timing table, any failures, and the final counts."""
    failed = [result for result in results if result.status == "failed"]
    executed = [result for result in results if result.status == "ok"]
    skipped = [result for result in results if result.status == "skipped"]

    slowest = sorted(executed, key=lambda result: result.seconds, reverse=True)
    if slowest:
        print(f"\nSlowest notebooks (of {len(executed)}):")
        for result in slowest[:N_SLOWEST]:
            print(f"  {result.seconds:8.1f}s  {display_path(result.path)}")

    if failed:
        print("\nFailures:")
        for result in failed:
            print(f"  {display_path(result.path)}")
            print(f"    {result.detail}")

    total = datetime.timedelta(seconds=round(elapsed))
    print(
        f"\n{len(executed)} succeeded, {len(failed)} failed, "
        f"{len(skipped)} skipped in {total}"
    )
    if not failed:
        print("\nReview the refreshed outputs with:\n  git diff --stat docs")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Execute the docs notebooks and refresh their stored outputs."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        metavar="PATH",
        help="notebooks, directories, or globs to execute "
        f"(default: docs/{{{','.join(NOTEBOOK_DIRS)}}})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        metavar="SECONDS",
        help=f"per-cell timeout, 0 for no limit (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help="notebooks to execute concurrently, sharing the cores between "
        "them (default: 1)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop at the first failure instead of reporting them all at the end",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="execute the notebooks but do not write them back",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the selected notebooks and exit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be at least 1")

    try:
        paths = discover_notebooks(args.paths)
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error
    if args.list:
        for path in paths:
            print(display_path(path))
        return
    if not paths:
        raise SystemExit("No notebooks selected")

    # Kernels are subprocesses, so they inherit these.
    env = thread_env(args.jobs)
    os.environ.update(env)
    if env:
        print(f"Limiting each kernel to {next(iter(env.values()))} threads.")
    elif args.jobs > 1:
        print(f"Using the thread limits already set: {thread_limits_in_env()}.")

    verb = "Checking" if args.dry_run else "Refreshing"
    print(f"{verb} {len(paths)} notebooks with {args.jobs} job(s).")
    start = time.monotonic()
    results = run_serial(paths, args) if args.jobs == 1 else run_parallel(paths, args)
    print_summary(results, time.monotonic() - start)

    if any(result.status == "failed" for result in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
