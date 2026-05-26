"""Tests für die robuste Redis-Backend-URL-Ableitung (#1)."""

from src.worker import _make_backend_url


def test_simple_db0_to_db1():
    assert _make_backend_url("redis://localhost:6379/0", db=1) == "redis://localhost:6379/1"


def test_hostname_with_digit_zero_unaffected():
    # Früherer str.replace("/0","/1")-Bug hätte den Hostnamen zerstört
    assert _make_backend_url("redis://redis10.example.com:6379/0", db=1) == \
        "redis://redis10.example.com:6379/1"


def test_two_digit_db_number():
    # "/10" darf nicht zu "/11" werden
    assert _make_backend_url("redis://redis:6379/10", db=1) == "redis://redis:6379/1"


def test_password_with_digits_preserved():
    url = "redis://:p0ssw0rd@redis:6379/0"
    assert _make_backend_url(url, db=1) == "redis://:p0ssw0rd@redis:6379/1"


def test_custom_db_number():
    assert _make_backend_url("redis://localhost:6379/0", db=2) == "redis://localhost:6379/2"
