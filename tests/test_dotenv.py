"""Unit tests for the minimal .env loader."""

from __future__ import annotations

from aca.dotenv import load_dotenv, parse_env


def test_parse_handles_comments_export_and_quotes():
    parsed = parse_env(
        "\n".join([
            "# a comment",
            "",
            "OPENAI_API_KEY=sk-plain",
            "export ANTHROPIC_API_KEY=sk-exported",
            'XAI_API_KEY=\"sk-quoted\"',
            "GEMINI_API_KEY='sk-single'",
            "  SPACED = value ",
            "novalue",  # no '=' -> ignored
        ])
    )
    assert parsed == {
        "OPENAI_API_KEY": "sk-plain",
        "ANTHROPIC_API_KEY": "sk-exported",
        "XAI_API_KEY": "sk-quoted",
        "GEMINI_API_KEY": "sk-single",
        "SPACED": "value",
    }


def test_load_sets_missing_keys(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-fromfile\n")
    applied = load_dotenv(tmp_path / ".env")
    assert applied == [tmp_path / ".env"]
    import os

    assert os.environ["OPENAI_API_KEY"] == "sk-fromfile"


def test_shell_env_takes_precedence_over_dotenv(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-shell")
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-fromfile\n")
    load_dotenv(tmp_path / ".env")
    import os

    assert os.environ["OPENAI_API_KEY"] == "sk-shell"  # not overridden


def test_earlier_file_wins_over_later(tmp_path, monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    primary = tmp_path / "data" / ".env"
    primary.parent.mkdir()
    primary.write_text("XAI_API_KEY=from-datadir\n")
    (tmp_path / ".env").write_text("XAI_API_KEY=from-cwd\n")
    load_dotenv(primary, tmp_path / ".env")
    import os

    assert os.environ["XAI_API_KEY"] == "from-datadir"


def test_missing_file_is_ignored(tmp_path):
    assert load_dotenv(tmp_path / "nope.env") == []


def test_allowlist_ignores_unrelated_secrets(tmp_path, monkeypatch):
    # The review's repro: running from a directory with another app's .env must not absorb its
    # secrets into this long-running daemon — only the requested key is ingested.
    for var in ("OPENAI_API_KEY", "DATABASE_URL", "STRIPE_SECRET_KEY"):
        monkeypatch.delenv(var, raising=False)
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=sk-aca\n"
        "DATABASE_URL=postgres://admin:secret@prod/app\n"
        "STRIPE_SECRET_KEY=sk_live_unrelated\n"
    )
    load_dotenv(tmp_path / ".env", allow={"OPENAI_API_KEY"})
    import os

    assert os.environ["OPENAI_API_KEY"] == "sk-aca"
    assert "DATABASE_URL" not in os.environ
    assert "STRIPE_SECRET_KEY" not in os.environ

