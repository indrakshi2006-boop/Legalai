"""A .env copied from .env.example with blank values behaves like the defaults."""

from pathlib import Path

from dotenv import dotenv_values

import providers
import retry
import utils

ROOT = Path(__file__).resolve().parents[1]


def test_env_example_has_names_only_and_blank_values_mean_defaults(monkeypatch):
    values = dotenv_values(ROOT / ".env.example")
    assert {"OPENAI_API_KEY", "GEMINI_API_KEY", "AEGIS_PROVIDER"} <= set(values)
    secrets = {name: value for name, value in values.items() if name.endswith("_KEY")}
    assert all(value == "" for value in secrets.values()), "no key value in the example file"
    for name, value in values.items():
        monkeypatch.setenv(name, value or "")
    config = providers.settings()
    assert config.name == "openai" and config.chat_model == "gpt-4o"
    assert config.key_present is False
    assert utils.comparison_timeout_seconds() == 60.0
    assert utils.max_upload_bytes() == 15 * 1024 * 1024
    assert utils.max_pdf_pages() == 40
    assert retry.configured_max_retries() == 3
    monkeypatch.setenv("AEGIS_PROVIDER", "gemini")
    gemini = providers.settings()
    assert gemini.chat_model == "gemini-3.8-flash"
    assert gemini.min_similarity is None and gemini.broad_k == 8
