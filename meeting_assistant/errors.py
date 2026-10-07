"""Errors carrying a user-facing message (shown verbatim in the UI)."""


class PipelineError(Exception):
    def __init__(self, message: str, stage: str = "pipeline"):
        super().__init__(message)
        self.stage = stage
        self.message = message

    def __str__(self) -> str:
        return self.message
