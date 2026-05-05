from typing import Sequence


def ask_yes_no(prompt: str) -> bool:
    """Repeat ``prompt`` until the user answers yes or no."""
    while True:
        answer = input(prompt).strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Please enter 'y' or 'n'.")


def ask_choice(prompt: str, options: Sequence[str]) -> int:
    """Show ``options`` numbered from 1 and return the chosen index (1-based)."""
    if not options:
        raise ValueError("ask_choice requires at least one option")
    print(prompt)
    for i, option in enumerate(options, start=1):
        print(f"  {i}. {option}")
    while True:
        answer = input("Enter your choice: ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return int(answer)
        print(f"Please enter a number from 1 to {len(options)}.")


def ask_from_set(prompt: str, options: Sequence[str], default: str | None = None) -> str:
    """Ask for a value from ``options``; an empty answer returns ``default`` when set."""
    options_lower = [o.lower() for o in options]
    if default is not None and default.lower() not in options_lower:
        raise ValueError(f"default '{default}' is not in options {list(options)}")
    rendered = ", ".join(options)
    suffix = f" [{default}]" if default else ""
    while True:
        answer = input(f"{prompt}{suffix} ({rendered}): ").strip().lower()
        if not answer and default is not None:
            return default
        if answer in options_lower:
            return options[options_lower.index(answer)]
        print(f"Unknown value. Choose one of: {rendered}")
