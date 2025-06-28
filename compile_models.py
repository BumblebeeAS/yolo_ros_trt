#!/usr/bin/env python3

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import typer
from ultralytics import YOLO
from rich import print
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()
err_console = Console(stderr=True)
app = typer.Typer(help="Compile latest YOLOv11 models to TensorRT engine format")

def export_model_to_engine(model_file_path: str) -> None:
    """Export the YOLO model to TensorRT engine format.
    Args:
        model_file_path (str): The path of the model file.
    """
    model = YOLO(model_file_path)
    model.export(format="engine")

def check_compiled_files_exist(pt_file_path: str) -> Dict[str, bool]:
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

    return {
        'onnx': onnx_path.exists(),
        'engine': engine_path.exists()
    }

def parse_filename(filename: str) -> Optional[Dict[str, Any]]:
    """Parse filename to extract category, date, and index.

    Expected format: yolov11s_{category}_{yyyymmdd}_{index}.{ext}

    Args:
        filename (str): The filename to parse

    Returns:
        Dict with category, date, index, and extension, or None if parsing fails
    """
    # Pattern to match yolov11s_{category}_{date}_{index}.{ext}
    pattern = r'yolov11s_(.+?)_(\d{8})_(\d+)\.(pt|onnx|engine)$'
    match = re.match(pattern, filename)

    if match:
        category, date_str, index_str, ext = match.groups()
        return {
            'category': category,
            'date': int(date_str),
            'index': int(index_str),
            'extension': ext,
            'filename': filename
        }
    return None

def find_latest_models(model_dir: Path) -> Dict[str, str]:
    """Find the latest model file for each category.

    Args:
        model_dir (Path): Directory containing model files

    Returns:
        Dict mapping category names to their latest model file paths
    """
    # Dictionary to store the latest model for each category
    latest_models = {}

    # Get all .pt files (we only want to compile .pt files, not .onnx or .engine)
    pt_files = list(model_dir.glob("*.pt"))

    if not pt_files:
        console.print(f"[yellow]No .pt files found in {model_dir}[/yellow]")
        return {}

    # Parse all filenames and group by category
    category_files = {}

    for pt_file in pt_files:
        parsed = parse_filename(pt_file.name)
        if parsed and parsed['extension'] == 'pt':
            category = parsed['category']
            if category not in category_files:
                category_files[category] = []
            category_files[category].append({
                'path': str(pt_file),
                'date': parsed['date'],
                'index': parsed['index'],
                'filename': parsed['filename']
            })

    # Find the latest file for each category
    for category, files in category_files.items():
        # Sort by date (descending), then by index (descending)
        latest_file = max(files, key=lambda x: (x['date'], x['index']))
        latest_models[category] = latest_file['path']

        console.print(f"[blue]Latest[/blue] [bold cyan]{category}[/bold cyan] model: [green]{latest_file['filename']}[/green]")

    return latest_models

@app.command()
def compile_latest(
    model_dir: str = typer.Argument("/workspaces/isaac_ros-dev/src/ml_models/yolov11_segment", help="Directory path containing the model files"),
    categories: Optional[List[str]] = typer.Option(None, "--category", "-c", help="Specific categories to compile (default: all)"),
    force: bool = typer.Option(False, "--force", "-f", help="Force recompilation even if .onnx/.engine files exist"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Show what would be compiled without actually compiling")
):
    """Compile the latest YOLOv11 models for each category to TensorRT engine format."""

    try:
        # Convert to Path object and validate
        model_path = Path(model_dir)

        if not model_path.exists():
            err_console.print(f"[bold red]Error:[/bold red] Directory {model_path} does not exist")
            raise typer.Exit(1)

        if not model_path.is_dir():
            err_console.print(f"[bold red]Error:[/bold red] {model_path} is not a directory")
            raise typer.Exit(1)

        console.print(Panel(f"[bold]Searching for models in:[/bold] {model_path}",
                          title="YOLOv11 Model Compiler", border_style="blue"))

        # Find latest models
        latest_models = find_latest_models(model_path)

        if not latest_models:
            console.print("[yellow]No models found to compile[/yellow]")
            raise typer.Exit(1)

        # Filter by categories if specified
        if categories:
            filtered_models = {cat: path for cat, path in latest_models.items() if cat in categories}
            if not filtered_models:
                err_console.print(f"[bold red]No models found for specified categories:[/bold red] {categories}")
                raise typer.Exit(1)
            latest_models = filtered_models

        # Check which models need compilation
        models_to_compile = {}
        skipped_models = {}

        for category, model_path in latest_models.items():
            compiled_files = check_compiled_files_exist(model_path)

            if force or not (compiled_files['onnx'] and compiled_files['engine']):
                models_to_compile[category] = model_path
            else:
                skipped_models[category] = model_path

        # Create compilation summary table
        table = Table(title=f"{'🔍 Compilation Preview' if dry_run else '⚙️ Model Compilation'}")
        table.add_column("Category", style="cyan", no_wrap=True)
        table.add_column("Model File", style="green")
        table.add_column("Action", style="bold")
        table.add_column("Reason", style="dim")

        console.print()  # Add spacing

        # Add skipped models to table
        for category, model_path in skipped_models.items():
            model_filename = Path(model_path).name
            table.add_row(
                category,
                model_filename,
                "[yellow]⏭️ Skipped[/yellow]",
                "Files exist (use --force to recompile)"
            )

        # Compile each model that needs compilation
        success_count = 0
        for category, model_path in models_to_compile.items():
            model_filename = Path(model_path).name

            if dry_run:
                compiled_files = check_compiled_files_exist(model_path)
                if force:
                    reason = "Force recompilation"
                elif not compiled_files['onnx']:
                    reason = "Missing .onnx file"
                elif not compiled_files['engine']:
                    reason = "Missing .engine file"
                else:
                    reason = "Missing both files"

                table.add_row(category, model_filename, "[yellow]Would compile[/yellow]", reason)
            else:
                console.print(f"\n[bold blue]Compiling {category}:[/bold blue] [green]{model_filename}[/green]")
                try:
                    with console.status(f"[bold green]Compiling {category} model..."):
                        export_model_to_engine(model_path)
                    table.add_row(category, model_filename, "[bold green]✓ Success[/bold green]", "Compiled to .onnx and .engine")
                    success_count += 1
                except Exception as e:
                    table.add_row(category, model_filename, f"[bold red]✗ Failed[/bold red]", str(e))
                    err_console.print(f"[bold red]Failed to compile {category} model:[/bold red] {str(e)}")

        console.print(table)

        # Summary
        if dry_run:
            total_actions = len(models_to_compile) + len(skipped_models)
            console.print(f"\n[bold blue]Dry run complete.[/bold blue] {len(models_to_compile)} models would be compiled, {len(skipped_models)} would be skipped.")
        else:
            total_models = len(models_to_compile)
            if total_models == 0:
                console.print(f"\n[bold green]🎉 All models already compiled![/bold green] {len(skipped_models)} models skipped (use --force to recompile)")
            elif success_count == total_models:
                console.print(f"\n[bold green]🎉 All {success_count} new models compiled successfully![/bold green]")
                if skipped_models:
                    console.print(f"[dim]{len(skipped_models)} models were already compiled and skipped[/dim]")
            else:
                console.print(f"\n[yellow]⚠️ {success_count}/{total_models} models compiled successfully[/yellow]")
                if skipped_models:
                    console.print(f"[dim]{len(skipped_models)} models were already compiled and skipped[/dim]")

    except Exception as e:
        err_console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(1)



if __name__ == "__main__":
    app()
