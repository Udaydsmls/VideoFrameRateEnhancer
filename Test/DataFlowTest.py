from pathlib import Path

from FolderOperations.DataFlow import DataFlowResult, start_data_flow


def test_data_flow_extracts_each_video(synthetic_video: Path, tmp_path: Path) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    for name in ("alpha", "beta"):
        (videos / f"{name}.mp4").write_bytes(synthetic_video.read_bytes())

    result = start_data_flow(videos, tmp_path / "frames", scale_factor=1.0)
    assert isinstance(result, DataFlowResult)
    assert set(result.extracted) == {"alpha", "beta"}
    assert all(count > 0 for count in result.extracted.values())
    assert result.failed == {}
    assert result.success


def test_data_flow_skips_existing(synthetic_video: Path, tmp_path: Path) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "alpha.mp4").write_bytes(synthetic_video.read_bytes())

    first = start_data_flow(videos, tmp_path / "frames")
    second = start_data_flow(videos, tmp_path / "frames")
    assert first.extracted["alpha"] == second.extracted["alpha"]


def test_data_flow_handles_missing_videos_dir(tmp_path: Path) -> None:
    result = start_data_flow(tmp_path / "missing", tmp_path / "frames")
    assert result.extracted == {}
    assert not result.success


def test_data_flow_reports_no_supported_files(tmp_path: Path) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "notes.txt").write_text("hello")
    result = start_data_flow(videos, tmp_path / "frames")
    assert result.extracted == {}
    assert result.failed == {}
