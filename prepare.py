from pathlib import Path

import pandas as pd


def prepare_LARD(lard_directory: Path):
    """Prepare the LARD dataset for training.

    ---
    Args:
        lard_directory (Path): Path to the LARD dataset directory.
            This should be the directory where the csv is stored.
    """
    csv = lard_directory / "LARD_train.csv"
    df = pd.read_csv(csv, sep=";")

    images = []
    corners = []
    for image, x1, y1, x2, y2, x3, y3, x4, y4 in df[
        ["image", "x_A", "y_A", "x_B", "y_B", "x_C", "y_C", "x_D", "y_D"]
    ].values:
        images.append(image)
        # Store corners as a list of tuples.
        corners.append([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])

    bbox_directory = lard_directory / "bboxes"
    bbox_directory.mkdir(exist_ok=True)

    all_content = []
    for image, corners_ in zip(images, corners):
        # Compute bbox xyxy coordinates.
        bbox = [
            min(corners_, key=lambda x: x[0])[0],
            min(corners_, key=lambda x: x[1])[1],
            max(corners_, key=lambda x: x[0])[0],
            max(corners_, key=lambda x: x[1])[1],
        ]

        # Compute filenames.
        bbox_filename = Path(image.replace("/", "."))
        bbox_filename = bbox_directory / bbox_filename.with_suffix(".txt")
        image_filename = lard_directory / image

        bbox_filename = bbox_filename.absolute()
        image_filename = image_filename.absolute()

        # Write bbox file.
        with open(bbox_filename, "w") as f:
            f.write(f"0 {' '.join(map(str, bbox))}")

        all_content.append(f"{image_filename} {bbox_filename}")

    # Write the content of the all.txt file.
    with open(lard_directory / "all.txt", "w") as f:
        f.write("\n".join(all_content))


if __name__ == "__main__":
    PATH_TO_LARD = Path("./.data/LARD/lard_dataset/lard/1.0.0/")
    prepare_LARD(PATH_TO_LARD)
