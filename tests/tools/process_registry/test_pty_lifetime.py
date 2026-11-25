"""Bounded after-exit draining alongside the reader cleanup in #132358 and #133468."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import time

import pytest

from tests.waiting import wait_until
from tools.process_registry import ProcessRegistry, ProcessSession


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("busy", [False, True])
def test_exited_child_finishes_while_another_holder_keeps_slave_open(tmp_path: Path, busy: bool) -> None:
    from ptyprocess import PtyProcess

    tty_path = tmp_path / "tty"
    program = (
        "import os,sys; from pathlib import Path; "
        "Path(sys.argv[1]).write_text(os.ttyname(1)); "
        "os.read(0,1); os.write(1,b'TAIL\\n'); os._exit(23)"
    )
    pty = PtyProcess.spawn([sys.executable, "-c", program, str(tty_path)], echo=False)
    registry = ProcessRegistry()
    session = ProcessSession(id="proc_held_slave", command="held slave", task_id="test",
                             started_at=time.time(), pid=pty.pid, _pty=pty)
    holder = None
    try:
        wait_until(lambda: tty_path.exists() and tty_path.read_text(), what="child's terminal name")
        # Same open-file lifetime as an escaped descendant, without orphaning any test process.
        holder = os.open(tty_path.read_text(), os.O_RDWR | os.O_NOCTTY)
        pty.write(b"\n")
        wait_until(lambda: not pty.isalive(), what="PTY child exited with its slave still open")
        os.set_blocking(holder, False)
        read = pty.read
        reads = 0
        replies: list[bytes] = []
        queries = b"\x1b[5n" * 1024
        pending = queries

        def read_and_replenish(size: int = 1024) -> bytes:
            nonlocal reads, pending
            chunk = read(size)
            reads += 1
            if reads > 1024:
                raise RuntimeError("unbounded after-exit drain")
            if busy:
                # Refill the real slave before the next readiness check: continuously readable,
                # but all new bytes are stripped queries, so a text-only budget never terminates.
                try:
                    sent = os.write(holder, pending)
                    pending = pending[sent:] or queries
                except BlockingIOError:
                    pass  # A full output queue is already readable.
            return chunk

        def reject_reply(s: bytes, flush: bool = True) -> int:
            replies.append(s)
            raise AssertionError("no child remains to consume query replies")

        pty.read = read_and_replenish
        pty.write = reject_reply
        registry._track_started(session, registry._pty_reader_loop, "held-slave-reader")
        assert session._completion_event.wait(5), "reader waited for a surviving slave holder"
        assert session._reader_thread is not None
        session._reader_thread.join(5)
        assert not session._reader_thread.is_alive()
        assert session.output_buffer.startswith("TAIL\r\n")
        # A bounded raw drain may end on an incomplete query, which flush() preserves.
        assert session.output_buffer.removeprefix("TAIL\r\n") in ("", "\x1b", "\x1b[", "\x1b[5")
        assert 0 < reads <= 1024
        if busy:
            assert reads > 1, "the busy holder never exercised the final drain"
        assert not replies, "query replies could block after the child exits"
        assert session.exit_code == 23
        assert pty.closed
    finally:
        if holder is not None:
            os.close(holder)
        registry.kill_all(source="test")
        if session._reader_thread is not None:
            session._reader_thread.join(5)
        pty.close(force=True)
