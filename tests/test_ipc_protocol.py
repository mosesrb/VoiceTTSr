import json
import queue
import pytest
from unittest.mock import MagicMock


class MockTtsWorker:
    """Mock test worker to verify IPC message routing and queue draining."""
    def __init__(self):
        self._resp_queue = queue.Queue()
        self.logs = []
        self.chunks = []
        self.ready = False

    def on_log(self, text, level="info"):
        self.logs.append((text, level))

    def on_chunk(self, filepath):
        self.chunks.append(filepath)

    def handle_line(self, raw_line: str):
        raw = raw_line.strip()
        if not raw:
            return
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            self.on_log(raw, "info")
            return

        status = msg.get("status", "")
        if status == "ready":
            self.ready = True
        elif status == "log":
            self.on_log(msg.get("text", ""), msg.get("level", "info"))
        elif status == "chunk":
            if msg.get("file"):
                self.on_chunk(msg["file"])
        else:
            self._resp_queue.put(msg)

    def clear_queue(self):
        while not self._resp_queue.empty():
            try:
                self._resp_queue.get_nowait()
            except queue.Empty:
                break

    def get_response(self, timeout=0.1):
        try:
            return self._resp_queue.get(timeout=timeout)
        except queue.Empty:
            return {"status": "error", "message": "Response timeout."}


class TestIpcProtocol:
    def test_json_message_routing(self):
        worker = MockTtsWorker()

        # Log event
        worker.handle_line(json.dumps({"status": "log", "text": "Model loading", "level": "ok"}))
        assert len(worker.logs) == 1
        assert worker.logs[0] == ("Model loading", "ok")

        # Ready event
        worker.handle_line(json.dumps({"status": "ready"}))
        assert worker.ready is True

        # Streaming chunk event
        worker.handle_line(json.dumps({"status": "chunk", "file": "chunk_001.wav"}))
        assert len(worker.chunks) == 1
        assert worker.chunks[0] == "chunk_001.wav"

        # Done event
        worker.handle_line(json.dumps({"status": "done", "file": "output.wav"}))
        resp = worker.get_response(timeout=0.1)
        assert resp["status"] == "done"
        assert resp["file"] == "output.wav"

    def test_queue_draining(self):
        worker = MockTtsWorker()
        worker.handle_line(json.dumps({"status": "stale_1"}))
        worker.handle_line(json.dumps({"status": "stale_2"}))

        assert not worker._resp_queue.empty()
        worker.clear_queue()
        assert worker._resp_queue.empty()

        resp = worker.get_response(timeout=0.01)
        assert resp["status"] == "error"
        assert "timeout" in resp["message"].lower()
