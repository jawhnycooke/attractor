"""Human-in-the-loop interaction protocols.

Provides an :class:`Interviewer` protocol and two implementations:

- :class:`CLIInterviewer` — interactive stdin/stdout using ``rich``.
- :class:`QueueInterviewer` — programmatic, for tests and automation.
"""

from __future__ import annotations

import asyncio
from typing import Protocol, runtime_checkable

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt


@runtime_checkable
class Interviewer(Protocol):
    """Protocol for human-in-the-loop interaction."""

    async def ask(self, prompt: str, options: list[str] | None = None) -> str:
        """Ask a free-form or multiple-choice question."""
        ...

    async def confirm(self, prompt: str) -> bool:
        """Ask for yes/no confirmation."""
        ...

    async def inform(self, message: str) -> None:
        """Display an informational message (no response expected)."""
        ...


class CLIInterviewer:
    """Interactive interviewer using ``rich`` for terminal formatting."""

    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()

    async def ask(self, prompt: str, options: list[str] | None = None) -> str:
        if options:
            self._console.print(Panel(prompt, title="Question"))
            for i, opt in enumerate(options, 1):
                self._console.print(f"  [{i}] {opt}")
            choice = await asyncio.to_thread(
                Prompt.ask,
                "Select an option",
                choices=[str(i) for i in range(1, len(options) + 1)],
                console=self._console,
            )
            return options[int(choice) - 1]
        else:
            self._console.print(Panel(prompt, title="Question"))
            return await asyncio.to_thread(
                Prompt.ask, "Your response", console=self._console
            )

    async def confirm(self, prompt: str) -> bool:
        self._console.print(Panel(prompt, title="Confirm"))
        return await asyncio.to_thread(Confirm.ask, "Proceed?", console=self._console)

    async def inform(self, message: str) -> None:
        self._console.print(Panel(message, title="Info", border_style="blue"))


class QueueInterviewer:
    """Programmatic interviewer fed by an asyncio Queue.

    Push responses into ``responses`` before the pipeline executes
    nodes that require human input.  Messages sent via ``inform``
    are collected in ``messages`` for later inspection.
    """

    def __init__(self) -> None:
        self.responses: asyncio.Queue[str] = asyncio.Queue()
        self.messages: list[str] = []

    async def ask(self, prompt: str, options: list[str] | None = None) -> str:
        return await self.responses.get()

    async def confirm(self, prompt: str) -> bool:
        response = await self.responses.get()
        return response.lower() in ("yes", "y", "true", "1")

    async def inform(self, message: str) -> None:
        self.messages.append(message)
