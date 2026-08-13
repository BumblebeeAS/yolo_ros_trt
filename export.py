import typer
from ultralytics import YOLO


def export_model_to_engine(model_file_path: str) -> None:
    """Export the YOLO model to TensorRT engine format.

    Args:
        model_file_path (str): The path of the model file.
    """
    model = YOLO(model_file_path)
    model.export(format="engine")


if __name__ == "__main__":
    typer.run(export_model_to_engine)
