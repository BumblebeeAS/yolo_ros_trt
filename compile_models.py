#!/usr/bin/env python3

import re
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from ultralytics import YOLO

console = Console()
err_console = Console(stderr=True)
app = typer.Typer(help="Interactively pick and compile YOLO models to TensorRT engine format")

DEFAULT_MODEL_DIRS = [
    "/workspaces/isaac_ros-dev/src/ml_models/yolov11_segment",
    "/workspaces/isaac_ros-dev/src/ml_models/yolov26_segment",
]


def export_model_to_engine(model_file_path: str) -> None:
    """Export the YOLO model to TensorRT engine format.
    Args:
        model_file_path (str): The path of the model file.
    """
    model = YOLO(model_file_path)
    model.export(format="engine", device=0, half=True)


def check_compiled_files_exist(pt_file_path: str) -> dict[str, bool]:
    """Check if corresponding .onnx and .engine files exist for a .pt file.

    Args:
        pt_file_path (str): Path to the .pt file

    Returns:
        Dict with 'onnx' and 'engine' keys indicating if files exist
    """
    pt_path = Path(pt_file_path)
    base_name = pt_path.stem  # filename without extension

    onnx_path = pt_path.parent / f"{base_name}.onnx"
    engine_path = pt_path.parent / f"{base_name}.engine"

    return {"onnx": onnx_path.exists(), "engine": engine_path.exists()}


def parse_filename(filename: str) -> dict[str, Any] | None:
    """Parse filename to extract version, category, date, and index.

    Expected format: yolov{version}_{category}_{yyyymmdd}[_{index}].{ext}

    Args:
        filename (str): The filename to parse

    Returns:
        Dict with version, category, date, index, and extension, or None if the
        filename doesn't follow the naming convention
    """
    pattern = r"(yolov\d+[a-z]?)_(.+?)_(\d{8})(?:_(\d+))?\.(pt|onnx|engine)$"
    match = re.match(pattern, filename)

    if match:
        version, category, date_str, index_str, ext = match.groups()
        return {
            "version": version,
            "category": category,
            "date": int(date_str),
            "index": int(index_str) if index_str is not None else 0,
            "extension": ext,
            "filename": filename,
        }
    return None


def discover_models(model_dirs: list[Path]) -> list[dict[str, Any]]:
    """Discover every .pt model across the given directories.

    Files that don't follow the yolov{version}_{category}_{date} naming
    convention are still included (with version/category left as None) so
    nothing is silently hidden from the picker.

    Args:
        model_dirs (list[Path]): Directories containing model files

    Returns:
        List of dicts, one per .pt file, with parsed metadata and compiled status
    """
    models = []
    for model_dir in model_dirs:
        pt_files = list(model_dir.glob("*.pt"))
        if not pt_files:
            console.print(f"[yellow]No .pt files found in {model_dir}[/yellow]")

        for pt_file in pt_files:
            parsed = parse_filename(pt_file.name)
            compiled_files = check_compiled_files_exist(str(pt_file))
            models.append(
                {
                    "path": str(pt_file),
                    "dir": model_dir.name,
                    "filename": pt_file.name,
                    "version": parsed["version"] if parsed else None,
                    "category": parsed["category"] if parsed else None,
                    "date": parsed["date"] if parsed else None,
                    "index": parsed["index"] if parsed else None,
                    "compiled": compiled_files["onnx"] and compiled_files["engine"],
                }
            )

    return models


def find_latest_paths(models: list[dict[str, Any]]) -> set[str]:
    """Return the path of the latest model for each (version, category) pair.

    Models that don't match the naming convention (version/category unknown)
    are excluded since there's nothing to compare them against.
    """
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for model in models:
        if model["version"] is None:
            continue
        key = (model["version"], model["category"])
        current = latest.get(key)
        if current is None or (model["date"], model["index"]) > (
            current["date"],
            current["index"],
        ):
            latest[key] = model

    return {model["path"] for model in latest.values()}


def sort_models(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort models for display: grouped by directory/category, newest first within each group."""
    return sorted(
        models,
        key=lambda m: (
            m["dir"],
            m["category"] or m["filename"],
            -(m["date"] or 0),
            -(m["index"] or 0),
        ),
    )


def render_model_table(models: list[dict[str, Any]], preselected: set[str]) -> None:
    """Print a numbered table of models. Row numbers are used to pick models by index."""
    table = Table(title="Available Models")
    table.add_column("#", style="bold", justify="right")
    table.add_column("Version", style="magenta", no_wrap=True)
    table.add_column("Category", style="cyan", no_wrap=True)
    table.add_column("Directory", style="blue")
    table.add_column("Model File", style="green")
    table.add_column("Status")

    for i, model in enumerate(models, start=1):
        marker = "[bold yellow]*[/bold yellow]" if model["path"] in preselected else ""
        status = (
            "[dim]compiled[/dim]" if model["compiled"] else "[bold]not compiled[/bold]"
        )
        table.add_row(
            f"{marker}{i}",
            model["version"] or "-",
            model["category"] or "-",
            model["dir"],
            model["filename"],
            status,
        )

    console.print(table)
    console.print("[dim]* = pre-selected (latest, not yet compiled)[/dim]")


def parse_selection(selection: str, max_index: int) -> set[int]:
    """Parse a selection string like '1,3,5-7' or 'all' into a set of 1-based indices.

    Args:
        selection (str): Raw user input
        max_index (int): Highest valid index (inclusive)

    Returns:
        Set of valid indices within [1, max_index]
    """
    selection = selection.strip()
    if not selection:
        return set()
    if selection.lower() == "all":
        return set(range(1, max_index + 1))

    indices: set[int] = set()
    for part in selection.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            indices.update(range(int(start_str), int(end_str) + 1))
        else:
            indices.add(int(part))

    return {i for i in indices if 1 <= i <= max_index}


@app.command()
def compile_latest(
    model_dirs: list[str] = typer.Argument(
        DEFAULT_MODEL_DIRS,
        help="Directory paths containing the model files",
    ),
    categories: list[str] | None = typer.Option(
        None, "--category", "-c", help="Only show these categories in the picker"
    ),
    versions: list[str] | None = typer.Option(
        None,
        "--version",
        "-V",
        help="Only show these YOLO versions in the picker, e.g. yolov11",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Also pre-check models that are already compiled",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Show what would be compiled without actually compiling",
    ),
):
    """Pick which YOLO models to compile to TensorRT engine format, interactively."""

    try:
        model_paths = [Path(model_dir) for model_dir in model_dirs]

        for model_path in model_paths:
            if not model_path.exists():
                err_console.print(
                    f"[bold red]Error:[/bold red] Directory {model_path} does not exist"
                )
                raise typer.Exit(1)

            if not model_path.is_dir():
                err_console.print(
                    f"[bold red]Error:[/bold red] {model_path} is not a directory"
                )
                raise typer.Exit(1)

        console.print(
            Panel(
                f"[bold]Searching for models in:[/bold] {', '.join(str(p) for p in model_paths)}",
                title="YOLO Model Compiler",
                border_style="blue",
            )
        )

        models = discover_models(model_paths)

        if versions:
            models = [m for m in models if m["version"] in versions]
        if categories:
            models = [m for m in models if m["category"] in categories]

        if not models:
            console.print("[yellow]No models found[/yellow]")
            raise typer.Exit(1)

        latest_paths = find_latest_paths(models)
        preselected = {
            m["path"]
            for m in models
            if m["path"] in latest_paths and (force or not m["compiled"])
        }

        sorted_models = sort_models(models)
        render_model_table(sorted_models, preselected)

        default_selection = ",".join(
            str(i) for i, m in enumerate(sorted_models, start=1) if m["path"] in preselected
        )

        console.print(
            "\n[dim]Enter numbers to compile, e.g. '1,3,5-7' or 'all'. Blank cancels.[/dim]"
        )
        selection = Prompt.ask(
            "Select models to compile", default=default_selection
        )
        indices = parse_selection(selection, len(sorted_models))

        if not indices:
            console.print("[yellow]No models selected[/yellow]")
            raise typer.Exit(0)

        selected_paths = [sorted_models[i - 1]["path"] for i in sorted(indices)]
        models_by_path = {m["path"]: m for m in models}

        # Create compilation summary table
        table = Table(
            title=f"{'🔍 Compilation Preview' if dry_run else '⚙️ Model Compilation'}"
        )
        table.add_column("Version", style="magenta", no_wrap=True)
        table.add_column("Category", style="cyan", no_wrap=True)
        table.add_column("Model File", style="green")
        table.add_column("Action", style="bold")
        table.add_column("Reason", style="dim")

        console.print()  # Add spacing

        success_count = 0
        for path in selected_paths:
            model = models_by_path[path]
            version = model["version"] or "-"
            category = model["category"] or "-"
            model_filename = model["filename"]

            if dry_run:
                reason = "Recompiling" if model["compiled"] else "Not yet compiled"
                table.add_row(
                    version,
                    category,
                    model_filename,
                    "[yellow]Would compile[/yellow]",
                    reason,
                )
                continue

            console.print(
                f"\n[bold blue]Compiling {version} {category}:[/bold blue] [green]{model_filename}[/green]"
            )
            try:
                with console.status(
                    f"[bold green]Compiling {version} {category} model..."
                ):
                    export_model_to_engine(path)
                table.add_row(
                    version,
                    category,
                    model_filename,
                    "[bold green]✓ Success[/bold green]",
                    "Compiled to .onnx and .engine",
                )
                success_count += 1
            except Exception as e:
                table.add_row(
                    version,
                    category,
                    model_filename,
                    "[bold red]✗ Failed[/bold red]",
                    str(e),
                )
                err_console.print(
                    f"[bold red]Failed to compile {version} {category} model:[/bold red] {str(e)}"
                )

        console.print(table)

        # Summary
        if dry_run:
            console.print(
                f"\n[bold blue]Dry run complete.[/bold blue] {len(selected_paths)} models would be compiled."
            )
        elif success_count == len(selected_paths):
            console.print(
                f"\n[bold green]🎉 All {success_count} selected models compiled successfully![/bold green]"
            )
        else:
            console.print(
                f"\n[yellow]⚠️ {success_count}/{len(selected_paths)} selected models compiled successfully[/yellow]"
            )

    except typer.Exit:
        raise
    except Exception as e:
        err_console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
