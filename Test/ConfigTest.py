import json
from pathlib import Path

from utilities.Config import load_config


def test_load_config_resolves_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "setup.json"
    config_path.write_text(
        json.dumps(
            {
                "absolute_path": str(tmp_path),
                "root_dir": "data",
                "architecture": "transformer",
                "num_epochs": 3,
                "image_size": [128, 128],
            }
        )
    )

    config = load_config(config_path)
    assert config.root == tmp_path / "data"
    assert config.videos == tmp_path / "data" / "videos"
    assert config.architecture == "transformer"
    assert config.num_epochs == 3
    assert config.image_size == (128, 128)


def test_load_config_uses_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "setup.json"
    config_path.write_text(json.dumps({"absolute_path": str(tmp_path)}))

    config = load_config(config_path)
    assert config.architecture == "unet"
    assert config.num_epochs == 10
    assert config.image_size is None
