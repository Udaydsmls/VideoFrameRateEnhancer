from io import StringIO

import pytest

from utilities.Prompts import ask_choice, ask_from_set, ask_yes_no


def _stdin(monkeypatch, *answers: str) -> None:
    monkeypatch.setattr("sys.stdin", StringIO("\n".join(answers) + "\n"))


def test_ask_yes_no_accepts_yes(monkeypatch, capsys) -> None:
    _stdin(monkeypatch, "y")
    assert ask_yes_no("> ") is True


def test_ask_yes_no_accepts_no(monkeypatch) -> None:
    _stdin(monkeypatch, "no")
    assert ask_yes_no("> ") is False


def test_ask_yes_no_reprompts_on_invalid(monkeypatch, capsys) -> None:
    _stdin(monkeypatch, "maybe", "yes")
    assert ask_yes_no("> ") is True
    captured = capsys.readouterr().out
    assert "Please enter 'y' or 'n'." in captured


def test_ask_choice_returns_index(monkeypatch) -> None:
    _stdin(monkeypatch, "2")
    assert ask_choice("Pick:", ["a", "b", "c"]) == 2


def test_ask_choice_reprompts_on_out_of_range(monkeypatch) -> None:
    _stdin(monkeypatch, "9", "0", "abc", "1")
    assert ask_choice("Pick:", ["only"]) == 1


def test_ask_choice_rejects_empty_options() -> None:
    with pytest.raises(ValueError):
        ask_choice("Pick:", [])


def test_ask_from_set_returns_default_on_empty(monkeypatch) -> None:
    _stdin(monkeypatch, "")
    assert ask_from_set("Arch", ["unet", "mamba"], default="unet") == "unet"


def test_ask_from_set_reprompts_until_known(monkeypatch) -> None:
    _stdin(monkeypatch, "rnn", "transformer")
    assert ask_from_set("Arch", ["unet", "transformer"]) == "transformer"


def test_ask_from_set_rejects_invalid_default() -> None:
    with pytest.raises(ValueError):
        ask_from_set("Arch", ["unet"], default="rnn")
