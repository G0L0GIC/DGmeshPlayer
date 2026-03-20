from __future__ import annotations

import json
from typing import Any

from PySide6 import QtCore


class PlayerBridge(QtCore.QObject):
    eventEmitted = QtCore.Signal(str)

    def __init__(self, controller: Any):
        super().__init__()
        self.controller = controller

    @QtCore.Slot(result=str)
    def getInitialState(self) -> str:
        return json.dumps(self.controller.get_ui_state(), ensure_ascii=False)

    @QtCore.Slot(str, result=str)
    def dispatchCommand(self, command_json: str) -> str:
        try:
            command = json.loads(command_json)
        except json.JSONDecodeError as exc:
            return json.dumps(
                {
                    "ok": False,
                    "error": {
                        "code": "BAD_JSON",
                        "message": str(exc),
                    },
                },
                ensure_ascii=False,
            )

        try:
            result = self.controller.handle_ui_command(command)
            return json.dumps({"ok": True, "data": result}, ensure_ascii=False)
        except Exception as exc:
            return json.dumps(
                {
                    "ok": False,
                    "error": {
                        "code": "COMMAND_FAILED",
                        "message": str(exc),
                    },
                },
                ensure_ascii=False,
            )

    def emit_event(self, event_type: str, payload: dict[str, Any]) -> None:
        self.eventEmitted.emit(
            json.dumps(
                {
                    "type": event_type,
                    "payload": payload,
                    "version": 1,
                },
                ensure_ascii=False,
            )
        )
