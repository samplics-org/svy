# tests/svy/utils/test_logconfig.py
import logging

from svy.utils.logconfig import install_pretty_logging


def test_additive_by_default():
    root = logging.getLogger()
    before = len(root.handlers)
    install_pretty_logging()  # default: additive, no replace
    after = len(root.handlers)
    # It should *not* clear existing handlers; may add one only if none existed
    assert after == before  # our function respects existing handlers


def test_replace_when_requested():
    root = logging.getLogger()
    install_pretty_logging(replace_handlers=True)  # force replace
    # Now we expect exactly one handler (our pretty/plain one)
    assert len(root.handlers) == 1


def test_install_pretty_logging_returns_handler():
    h = install_pretty_logging(logger_name="svy.test", replace_handlers=True)
    assert h is not None
    assert h in logging.getLogger("svy.test").handlers
