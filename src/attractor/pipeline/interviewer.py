"""Human-in-the-loop interaction protocols.

Provides an :class:`Interviewer` protocol and implementations:

- :class:`CLIInterviewer` — interactive stdin/stdout using ``rich``.
- :class:`QueueInterviewer` — programmatic, for tests and automation.
- :class:`AutoApproveInterviewer` — always approves, for CI/automation.
- :class:`CallbackInterviewer` — delegates to a user-provided callback.
- :class:`RecordingInterviewer` — wraps another interviewer and records interactions.

Question/Answer type system per spec.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Protocol, runtime_checkable

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

logger = logging.getLogger(__name__)


class QuestionType(str, enum.Enum):
    """Type of question determining UI and valid answer format."""

    YES_NO = "yes_no"
    MULTIPLE_CHOICE = "multiple_choice"
    FREEFORM = "freeform"
    CONFIRMATION = "confirmation"


class AnswerValue(str, enum.Enum):
    """Predefined answer values for structured responses."""

    YES = "yes"
    NO = "no"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


@dataclass
class Option:
    """A single choice in a multiple-choice question.

    Attributes:
        key: Accelerator key (e.g., "Y", "A", "1").
        label: Display text (e.g., "Yes, deploy to production").
    """

    key: str
    label: str


@dataclass
class Answer:
    """Structured response to a question.

    Attributes:
        value: The answer value — a string or AnswerValue enum member.
        selected_option: The full Option if this was a multiple-choice answer.
        text: Free text response for FREEFORM questions.
        metadata: Arbitrary context about the answer.
    """

    value: str | AnswerValue
    selected_option: Option | None = None
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Question:
    """Structured question presented to a human operator.

    Attributes:
        text: The question text displayed to the human.
        type: Determines the UI and valid answer format.
        options: Available choices for MULTIPLE_CHOICE questions.
        default: Fallback answer on timeout or skip.
        timeout_seconds: Maximum wait time before timeout.
        stage: Name of the originating pipeline stage.
        metadata: Arbitrary context about the question.
    """

    text: str
    type: QuestionType
    options: list[Option] = field(default_factory=list)
    default: Answer | None = None
    timeout_seconds: float | None = None
    stage: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InteractionRecord:
    """Record of a question-answer interaction.

    Attributes:
        question: The question that was asked.
        answer: The answer that was given.
        timestamp: UNIX epoch when the interaction occurred.
    """

    question: Question
    answer: Answer
    timestamp: float = field(default_factory=time.time)


@runtime_checkable
class Interviewer(Protocol):
    """Protocol for human-in-the-loop interaction.

    Implementations handle question presentation and answer collection
    using structured Question/Answer types.
    """

    async def ask(self, question: Question) -> Answer:
        """Ask a structured question and return a structured answer."""
        ...

    async def ask_multiple(self, questions: list[Question]) -> list[Answer]:
        """Ask multiple questions in batch."""
        ...

    async def inform(self, message: str, stage: str = "") -> None:
        """Display an informational message (no response expected).

        Args:
            message: The message to display.
            stage: The originating pipeline stage name.
        """
        ...


class CLIInterviewer:
    """Interactive interviewer using ``rich`` for terminal formatting."""

    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()

    async def ask(self, question: Question) -> Answer:
        """Present a question in the terminal and collect the answer."""
        title = f"Question [{question.stage}]" if question.stage else "Question"

        if question.type == QuestionType.MULTIPLE_CHOICE and question.options:
            self._console.print(Panel(question.text, title=title))
            for i, opt in enumerate(question.options, 1):
                self._console.print(f"  [{opt.key}] {opt.label}")
            choice = await asyncio.to_thread(
                Prompt.ask,
                "Select an option",
                choices=[opt.key for opt in question.options],
                console=self._console,
            )
            selected = next(
                (o for o in question.options if o.key == choice), question.options[0]
            )
            return Answer(value=selected.key, selected_option=selected)

        if question.type in (QuestionType.YES_NO, QuestionType.CONFIRMATION):
            self._console.print(Panel(question.text, title=title))
            result = await asyncio.to_thread(
                Confirm.ask, "Proceed?", console=self._console
            )
            return Answer(
                value=AnswerValue.YES if result else AnswerValue.NO,
            )

        # FREEFORM
        self._console.print(Panel(question.text, title=title))
        text = await asyncio.to_thread(
            Prompt.ask, "Your response", console=self._console
        )
        return Answer(value=text, text=text)

    async def ask_multiple(self, questions: list[Question]) -> list[Answer]:
        """Ask multiple questions sequentially."""
        return [await self.ask(q) for q in questions]

    async def inform(self, message: str, stage: str = "") -> None:
        """Display an informational message."""
        title = f"Info [{stage}]" if stage else "Info"
        self._console.print(Panel(message, title=title, border_style="blue"))


class QueueInterviewer:
    """Programmatic interviewer fed by an asyncio Queue.

    Push :class:`Answer` objects into ``responses`` before the pipeline
    executes nodes that require human input.  Messages sent via ``inform``
    are collected in ``messages`` for later inspection.
    """

    def __init__(self) -> None:
        self.responses: asyncio.Queue[Answer] = asyncio.Queue()
        self.messages: list[str] = []

    async def ask(self, question: Question) -> Answer:
        """Return the next pre-queued answer."""
        return await self.responses.get()

    async def ask_multiple(self, questions: list[Question]) -> list[Answer]:
        """Return N pre-queued answers."""
        return [await self.responses.get() for _ in questions]

    async def inform(self, message: str, stage: str = "") -> None:
        """Record the informational message."""
        self.messages.append(message)


# Type alias for callback-based interviewer
QuestionCallback = Callable[[Question], Coroutine[Any, Any, Answer]]


class AutoApproveInterviewer:
    """Interviewer that automatically approves all questions.

    For CI/automation pipelines where no human is present.

    Behavior:
    - YES_NO / CONFIRMATION: returns YES
    - MULTIPLE_CHOICE: selects the first option
    - FREEFORM: returns default if set, otherwise SKIPPED
    """

    async def ask(self, question: Question) -> Answer:
        """Auto-approve the question based on its type."""
        if question.type in (QuestionType.YES_NO, QuestionType.CONFIRMATION):
            return Answer(value=AnswerValue.YES)
        if question.type == QuestionType.MULTIPLE_CHOICE and question.options:
            opt = question.options[0]
            return Answer(value=opt.key, selected_option=opt)
        if question.default is not None:
            return question.default
        return Answer(value=AnswerValue.SKIPPED)

    async def ask_multiple(self, questions: list[Question]) -> list[Answer]:
        """Auto-approve all questions."""
        return [await self.ask(q) for q in questions]

    async def inform(self, message: str, stage: str = "") -> None:
        """No-op for automation."""
        logger.debug("AutoApprove inform [%s]: %s", stage, message)


class CallbackInterviewer:
    """Interviewer that delegates to a user-provided callback function."""

    def __init__(self, callback: QuestionCallback) -> None:
        self._callback = callback

    async def ask(self, question: Question) -> Answer:
        """Delegate to the callback."""
        return await self._callback(question)

    async def ask_multiple(self, questions: list[Question]) -> list[Answer]:
        """Delegate each question to the callback."""
        return [await self._callback(q) for q in questions]

    async def inform(self, message: str, stage: str = "") -> None:
        """No-op for callback interviewer."""


class RecordingInterviewer:
    """Wraps another Interviewer and records all question-answer pairs."""

    def __init__(self, inner: Interviewer) -> None:
        self._inner = inner
        self.records: list[InteractionRecord] = []

    async def ask(self, question: Question) -> Answer:
        """Delegate and record the interaction."""
        answer = await self._inner.ask(question)
        self.records.append(
            InteractionRecord(
                question=question, answer=answer, timestamp=time.time()
            )
        )
        return answer

    async def ask_multiple(self, questions: list[Question]) -> list[Answer]:
        """Delegate and record each interaction."""
        answers = await self._inner.ask_multiple(questions)
        for q, a in zip(questions, answers):
            self.records.append(
                InteractionRecord(question=q, answer=a, timestamp=time.time())
            )
        return answers

    async def inform(self, message: str, stage: str = "") -> None:
        """Delegate to the inner interviewer."""
        await self._inner.inform(message, stage)
