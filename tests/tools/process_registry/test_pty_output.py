"""Buffered-exit regression for the PTY reader discussed in #133468."""
from __future__ import annotations

from pathlib import Path
import shlex
import sys
import threading

import pytest

from tests.waiting import wait_until
from tools.process_registry import ProcessRegistry, ProcessSession


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("exit_code", [0, 23])
@pytest.mark.parametrize("reader_started", [False, True])
def test_drains_exited_child_before_publishing_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit_code: int, reader_started: bool,
) -> None:
    registry = ProcessRegistry()
    entered, release = threading.Event(), threading.Event()
    expected = "START\ncafé 漢🙂\nSTDERR\nFINAL\n\x1b["
    before, stderr, after = expected.partition("STDERR\n")
    program = (
        f"import os; os.write(1, {before.encode()!r}); "
        f"os.write(2, {stderr.encode()!r}); os.write(1, {after.encode()!r}); "
        f"os._exit({exit_code})"
    )
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(program)}"
    # Keep the real spawn, ownership, reader and completion; omit login-shell startup.
    monkeypatch.setattr(registry, "_scope_argv", lambda _s, cmd, *_args: shlex.split(cmd))
    track = registry._track_started

    def track_with_gate(session, target, name, extra_args=()):
        session.notify_on_complete = True
        read = session._pty.read
        # Deterministic UTF-8 and query-prefix splits, without slowing the child exit.
        session._pty.read = lambda size: read(1)
        if not reader_started:
            wait_until(lambda: not session._pty.isalive(), what="PTY child exit before first read")
        track(session, target, name, extra_args)

    monkeypatch.setattr(registry, "_track_started", track_with_gate)

    def pause_first_output(_session: ProcessSession, _text: str) -> None:
        if reader_started and not entered.is_set():
            entered.set()
            release.wait(10)

    monkeypatch.setattr(registry, "on_output", pause_first_output)
    session = registry.spawn_local(command, cwd=str(tmp_path), use_pty=True, owner_task_id="pty-owner")
    try:
        assert not session.pty_fallback, session.pty_fallback
        if reader_started:
            assert entered.wait(5), "reader never reached its output sink"
            wait_until(lambda: not session._pty.isalive(), what="PTY child exit with unread tail")
            assert registry.completion_queue.empty()
            release.set()
        assert session._completion_event.wait(5), "reader did not publish completion"
        assert session._reader_thread is not None
        session._reader_thread.join(5)
        assert not session._reader_thread.is_alive()
        assert session._pty.closed
        assert session.exit_code == exit_code
        assert session.output_buffer == expected.replace("\n", "\r\n")
        # The public result deliberately strips ANSI, including an incomplete escape prefix.
        displayed = "START\r\ncafé 漢🙂\r\nSTDERR\r\nFINAL\r\n"
        result = registry.wait(session.id, timeout=2)
        assert result["exit_code"] == exit_code
        assert result["output"] == displayed
        completion = registry.completion_queue.get(timeout=2)
        assert completion["owner_task_id"] == "pty-owner"
        assert completion["exit_code"] == exit_code
        assert completion["output"] == displayed
        assert registry.completion_queue.empty()
    finally:
        release.set()
        registry.kill_all(source="test")
        if session._reader_thread is not None:
            session._reader_thread.join(5)
