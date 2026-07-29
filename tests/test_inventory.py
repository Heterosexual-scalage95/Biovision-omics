from pathlib import Path
import pandas as pd
from biovision_omics.file_inventory import inventory_directory


def test_inventory_csv(tmp_path: Path):
    pd.DataFrame({"cell_id": [1, 2], "x": [3, 4], "y": [5, 6]}).to_csv(tmp_path / "cell_coordinates.csv", index=False)
    inventory = inventory_directory(tmp_path)
    assert len(inventory) == 1
    assert "spatial" in inventory.iloc[0]["categories"]
    assert inventory.iloc[0]["preview_columns"] == 3
