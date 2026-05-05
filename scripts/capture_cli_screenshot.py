"""Run main.py through its menu and render the output as a PNG screenshot."""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "assets" / "cli_screenshot.png"
FONT_PATH = Path(r"C:\Windows\Fonts\consola.ttf")

BACKGROUND = (24, 24, 27)
FOREGROUND = (235, 235, 235)
PROMPT_COLOR = (147, 197, 253)
PADDING = 28
LINE_HEIGHT = 22
FONT_SIZE = 16


def _capture_output() -> str:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    proc = subprocess.Popen(
        [sys.executable, "-u", "main.py"],
        cwd=str(ROOT),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )

    chunks: list[str] = []

    def pump() -> None:
        assert proc.stdout is not None
        while True:
            data = proc.stdout.read1(4096)
            if not data:
                break
            chunks.append(data.decode("utf-8", errors="replace"))

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()

    def wait_for(token: str, timeout: float = 30.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if any(token in chunk for chunk in chunks):
                return
            time.sleep(0.1)
        raise TimeoutError(
            f"Did not see '{token}' in output within {timeout}s\n"
            f"Captured so far:\n{''.join(chunks)}"
        )

    def respond(token: str, echo: str) -> None:
        wait_for(token)
        chunks.append(echo + "\n")
        assert proc.stdin is not None
        proc.stdin.write(echo.encode("utf-8") + b"\n")
        proc.stdin.flush()

    try:
        respond("Press Enter to continue", "")
        respond("Architecture", "")
        respond("data flow", "n")
        wait_for("Enter your choice")
        time.sleep(0.3)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()

    return "".join(chunks)


def _render(text: str, output_path: Path) -> None:
    font = ImageFont.truetype(str(FONT_PATH), FONT_SIZE)
    lines = text.replace("\r\n", "\n").rstrip().split("\n")
    width = PADDING * 2 + max(
        (int(font.getlength(line)) for line in lines),
        default=400,
    )
    height = PADDING * 2 + LINE_HEIGHT * len(lines)
    width = max(width, 720)

    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    y = PADDING
    for line in lines:
        color = PROMPT_COLOR if line.lstrip().startswith(("Enter", "Architecture", "Press Enter", "Do you want")) else FOREGROUND
        draw.text((PADDING, y), line, font=font, fill=color)
        y += LINE_HEIGHT

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Wrote {output_path} ({width}x{height})")


def main() -> int:
    output = _capture_output()
    print(output)
    print("---")
    _render(output, OUTPUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
