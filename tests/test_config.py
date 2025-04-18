import os
import importlib

import config

def reload_config():
    importlib.reload(config)
    return config

def test_default_langchain_max_concurrency(monkeypatch):
    monkeypatch.delenv('LANGCHAIN_MAX_CONCURRENCY', raising=False)
    cfg = reload_config()
    assert isinstance(cfg.LANGCHAIN_MAX_CONCURRENCY, int)
    assert cfg.LANGCHAIN_MAX_CONCURRENCY == 5

def test_custom_langchain_max_concurrency(monkeypatch):
    monkeypatch.setenv('LANGCHAIN_MAX_CONCURRENCY', '10')
    cfg = reload_config()
    assert cfg.LANGCHAIN_MAX_CONCURRENCY == 10

def test_default_use_langfuse(monkeypatch):
    monkeypatch.delenv('USE_LANGFUSE', raising=False)
    cfg = reload_config()
    assert cfg.USE_LANGFUSE is False

def test_custom_use_langfuse(monkeypatch):
    monkeypatch.setenv('USE_LANGFUSE', 'true')
    cfg = reload_config()
    assert cfg.USE_LANGFUSE is True