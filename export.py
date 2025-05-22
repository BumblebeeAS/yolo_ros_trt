import typer
from ultralytics import YOLO


def export_model_to_engine(model_file_name: str) -> None:
    """Export the YOLO model to TensorRT engine format.

    Args:
        model_file_name (str): The name of the model file.
    """
    model = YOLO(f"models/{model_file_name}")
    model.export(format="engine")


if __name__ == "__main__":
    typer.run(export_model_to_engine)
