# Interviewer Protocol — Spec Compliance Gap

## Status: Not Implemented

This document describes the interviewer protocol changes required to align
`src/attractor/pipeline/interviewer.py` with the attractor specification.

---

## Current Implementation

The current `Interviewer` protocol has three methods:

```python
class Interviewer(Protocol):
    async def ask(self, prompt: str, options: list[str] | None = None) -> str: ...
    async def confirm(self, prompt: str) -> bool: ...
    async def inform(self, message: str) -> None: ...
```

Two implementations exist: `CLIInterviewer` (Rich-based terminal) and
`QueueInterviewer` (for tests).

### Problems

| ID  | Issue |
|-----|-------|
| I1  | No `Question`/`Answer` type system — `ask()` takes raw strings, not structured questions |
| I2  | Missing implementations: `AutoApproveInterviewer`, `CallbackInterviewer`, `RecordingInterviewer` |
| I3  | `ask()` returns `str` not `Answer`; `confirm()` is not in the spec (replaced by `QuestionType.YES_NO`) |

---

## Spec Requirements

### Interviewer Interface

```
INTERFACE Interviewer:
    FUNCTION ask(question: Question) -> Answer
    FUNCTION ask_multiple(questions: List<Question>) -> List<Answer>
    FUNCTION inform(message: String, stage: String) -> Void
```

Key differences from current code:
- `ask()` accepts a `Question` object, not a raw string
- `ask()` returns an `Answer` object, not a raw string
- `ask_multiple()` is a new batch method
- `inform()` takes a `stage` parameter for display context
- `confirm()` does not exist — use `QuestionType.YES_NO` or `QuestionType.CONFIRMATION`

### Question Model

```python
@dataclass
class Question:
    text: str                               # Displayed to the human
    type: QuestionType                      # Determines UI and valid answer format
    options: list[Option] = field(default_factory=list)  # For MULTIPLE_CHOICE
    default: Answer | None = None           # Fallback on timeout/skip
    timeout_seconds: float | None = None    # Max wait time
    stage: str = ""                         # Originating stage name
    metadata: dict[str, Any] = field(default_factory=dict)  # Arbitrary context
```

### QuestionType Enum

```python
class QuestionType(str, enum.Enum):
    YES_NO = "yes_no"                   # Binary affirmative/negative
    MULTIPLE_CHOICE = "multiple_choice" # Select one from a list
    FREEFORM = "freeform"               # Free text input
    CONFIRMATION = "confirmation"       # Yes/no confirmation (semantically distinct from YES_NO)
```

The distinction between `YES_NO` and `CONFIRMATION`:
- `YES_NO` is a neutral binary question ("Should we deploy to staging?")
- `CONFIRMATION` implies the action will proceed unless rejected ("Deploy to production?")

### Option Model

```python
@dataclass
class Option:
    key: str     # Accelerator key (e.g., "Y", "A", "1")
    label: str   # Display text (e.g., "Yes, deploy to production")
```

### Answer Model

```python
@dataclass
class Answer:
    value: str | AnswerValue            # Selected value or enum
    selected_option: Option | None = None  # Full option (MULTIPLE_CHOICE)
    text: str = ""                      # Free text response (FREEFORM)
```

### AnswerValue Enum

```python
class AnswerValue(str, enum.Enum):
    YES = "yes"
    NO = "no"
    SKIPPED = "skipped"       # Human skipped the question
    TIMEOUT = "timeout"       # No response within timeout window
```

---

## Required Implementations

### 1. AutoApproveInterviewer

For CI/automation pipelines where no human is present.

Behavior:
- `YES_NO` / `CONFIRMATION` questions: always returns `AnswerValue.YES`
- `MULTIPLE_CHOICE` questions: selects the **first** option
- `FREEFORM` questions: returns the `default` answer if set, otherwise returns `AnswerValue.SKIPPED`
- `inform()`: no-op (logs at DEBUG level)

```python
class AutoApproveInterviewer:
    async def ask(self, question: Question) -> Answer:
        if question.type in (QuestionType.YES_NO, QuestionType.CONFIRMATION):
            return Answer(value=AnswerValue.YES)
        if question.type == QuestionType.MULTIPLE_CHOICE and question.options:
            opt = question.options[0]
            return Answer(value=opt.key, selected_option=opt)
        if question.default is not None:
            return question.default
        return Answer(value=AnswerValue.SKIPPED)

    async def ask_multiple(self, questions: list[Question]) -> list[Answer]:
        return [await self.ask(q) for q in questions]

    async def inform(self, message: str, stage: str = "") -> None:
        pass  # no-op
```

### 2. CallbackInterviewer

Delegates to a user-provided callback function.

```python
QuestionCallback = Callable[[Question], Coroutine[Any, Any, Answer]]

class CallbackInterviewer:
    def __init__(self, callback: QuestionCallback) -> None:
        self._callback = callback

    async def ask(self, question: Question) -> Answer:
        return await self._callback(question)

    async def ask_multiple(self, questions: list[Question]) -> list[Answer]:
        return [await self._callback(q) for q in questions]

    async def inform(self, message: str, stage: str = "") -> None:
        pass  # callback interviewer ignores inform
```

### 3. RecordingInterviewer

Wraps another `Interviewer` and records all question-answer pairs.

```python
@dataclass
class InteractionRecord:
    question: Question
    answer: Answer
    timestamp: float  # time.time()

class RecordingInterviewer:
    def __init__(self, inner: Interviewer) -> None:
        self._inner = inner
        self.records: list[InteractionRecord] = []

    async def ask(self, question: Question) -> Answer:
        answer = await self._inner.ask(question)
        self.records.append(InteractionRecord(
            question=question, answer=answer, timestamp=time.time()
        ))
        return answer

    async def ask_multiple(self, questions: list[Question]) -> list[Answer]:
        answers = await self._inner.ask_multiple(questions)
        for q, a in zip(questions, answers):
            self.records.append(InteractionRecord(
                question=q, answer=a, timestamp=time.time()
            ))
        return answers

    async def inform(self, message: str, stage: str = "") -> None:
        await self._inner.inform(message, stage)
```

### 4. CLIInterviewer (Update)

The existing `CLIInterviewer` must be updated to:
- Accept `Question` objects instead of raw strings
- Return `Answer` objects instead of raw strings
- Handle all four `QuestionType` variants
- Support `timeout_seconds` via non-blocking reads
- Display `stage` in the panel title

### 5. QueueInterviewer (Update)

The existing `QueueInterviewer` must be updated to:
- Accept `Question` objects
- Return `Answer` objects from the queue
- Support `ask_multiple()` by popping N answers
- The queue type changes from `asyncio.Queue[str]` to `asyncio.Queue[Answer]`

---

## Integration Points

### wait.human Handler

The `WaitHumanHandler` in `handlers.py` is the primary consumer. It currently calls:

```python
answer = await interviewer.ask(prompt, options=edge_labels)
```

After the refactor, it should construct a `Question` object:

```python
options = [Option(key=str(i+1), label=lbl) for i, lbl in enumerate(edge_labels)]
question = Question(
    text=prompt,
    type=QuestionType.MULTIPLE_CHOICE if options else QuestionType.FREEFORM,
    options=options,
    stage=node.name,
)
answer = await interviewer.ask(question)
result.preferred_label = answer.selected_option.label if answer.selected_option else answer.value
```

### Engine inform() Calls

The engine currently does not call `inform()`. The spec requires the engine to
call `interviewer.inform(message, stage=node.name)` for status updates when
`auto_status=true` is set on a node.

---

## Backward Compatibility

The `Interviewer` protocol signature changes are breaking. All call sites must
be updated simultaneously:

1. `src/attractor/pipeline/interviewer.py` — protocol + all implementations
2. `src/attractor/pipeline/handlers.py` — `WaitHumanHandler.execute()`
3. `tests/pipeline/test_handlers.py` — handler tests using `QueueInterviewer`
4. `src/attractor/cli.py` — `CLIInterviewer` construction (no change needed)

The `confirm()` method should be removed entirely — its functionality is
subsumed by `QuestionType.YES_NO` and `QuestionType.CONFIRMATION`.
