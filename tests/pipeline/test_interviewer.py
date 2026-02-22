"""Tests for the interviewer protocol and implementations."""

from __future__ import annotations

import asyncio

import pytest

from attractor.pipeline.interviewer import (
    Answer,
    AnswerValue,
    AutoApproveInterviewer,
    CallbackInterviewer,
    InteractionRecord,
    Interviewer,
    Option,
    Question,
    QuestionType,
    QueueInterviewer,
    RecordingInterviewer,
)


class TestQuestionAnswerTypes:
    """Tests for Question, Answer, Option, and enum types."""

    def test_question_type_values(self) -> None:
        assert QuestionType.YES_NO == "yes_no"
        assert QuestionType.MULTIPLE_CHOICE == "multiple_choice"
        assert QuestionType.FREEFORM == "freeform"
        assert QuestionType.CONFIRMATION == "confirmation"

    def test_answer_value_values(self) -> None:
        assert AnswerValue.YES == "yes"
        assert AnswerValue.NO == "no"
        assert AnswerValue.SKIPPED == "skipped"
        assert AnswerValue.TIMEOUT == "timeout"

    def test_question_defaults(self) -> None:
        q = Question(text="Test?", type=QuestionType.FREEFORM)
        assert q.text == "Test?"
        assert q.type == QuestionType.FREEFORM
        assert q.options == []
        assert q.default is None
        assert q.timeout_seconds is None
        assert q.stage == ""
        assert q.metadata == {}

    def test_question_with_options(self) -> None:
        opts = [Option(key="1", label="Yes"), Option(key="2", label="No")]
        q = Question(
            text="Choose?",
            type=QuestionType.MULTIPLE_CHOICE,
            options=opts,
            stage="review",
        )
        assert len(q.options) == 2
        assert q.options[0].key == "1"
        assert q.options[0].label == "Yes"
        assert q.stage == "review"

    def test_answer_defaults(self) -> None:
        a = Answer(value=AnswerValue.YES)
        assert a.value == AnswerValue.YES
        assert a.selected_option is None
        assert a.text == ""
        assert a.metadata == {}

    def test_answer_with_option(self) -> None:
        opt = Option(key="A", label="Alpha")
        a = Answer(value="A", selected_option=opt)
        assert a.selected_option is opt

    def test_answer_with_text(self) -> None:
        a = Answer(value="custom text", text="custom text")
        assert a.text == "custom text"

    def test_option_fields(self) -> None:
        opt = Option(key="Y", label="Yes, deploy")
        assert opt.key == "Y"
        assert opt.label == "Yes, deploy"


class TestQueueInterviewer:
    """Tests for QueueInterviewer with Question/Answer types."""

    async def test_ask_returns_queued_answer(self) -> None:
        qi = QueueInterviewer()
        answer = Answer(value=AnswerValue.YES)
        await qi.responses.put(answer)

        q = Question(text="Proceed?", type=QuestionType.YES_NO)
        result = await qi.ask(q)
        assert result.value == AnswerValue.YES

    async def test_ask_multiple_returns_multiple_answers(self) -> None:
        qi = QueueInterviewer()
        a1 = Answer(value=AnswerValue.YES)
        a2 = Answer(value="option_b", selected_option=Option(key="B", label="B"))
        await qi.responses.put(a1)
        await qi.responses.put(a2)

        questions = [
            Question(text="Q1?", type=QuestionType.YES_NO),
            Question(text="Q2?", type=QuestionType.MULTIPLE_CHOICE),
        ]
        results = await qi.ask_multiple(questions)
        assert len(results) == 2
        assert results[0].value == AnswerValue.YES
        assert results[1].value == "option_b"

    async def test_inform_records_message(self) -> None:
        qi = QueueInterviewer()
        await qi.inform("Status update", stage="build")
        assert qi.messages == ["Status update"]

    async def test_inform_with_stage_still_records(self) -> None:
        qi = QueueInterviewer()
        await qi.inform("msg1", stage="stage1")
        await qi.inform("msg2", stage="stage2")
        assert len(qi.messages) == 2

    async def test_protocol_compliance(self) -> None:
        qi = QueueInterviewer()
        assert isinstance(qi, Interviewer)


class TestAutoApproveInterviewer:
    """Tests for AutoApproveInterviewer."""

    async def test_yes_no_returns_yes(self) -> None:
        aai = AutoApproveInterviewer()
        q = Question(text="Deploy?", type=QuestionType.YES_NO)
        result = await aai.ask(q)
        assert result.value == AnswerValue.YES

    async def test_confirmation_returns_yes(self) -> None:
        aai = AutoApproveInterviewer()
        q = Question(text="Confirm?", type=QuestionType.CONFIRMATION)
        result = await aai.ask(q)
        assert result.value == AnswerValue.YES

    async def test_multiple_choice_selects_first(self) -> None:
        aai = AutoApproveInterviewer()
        opts = [
            Option(key="A", label="Alpha"),
            Option(key="B", label="Beta"),
        ]
        q = Question(text="Pick one", type=QuestionType.MULTIPLE_CHOICE, options=opts)
        result = await aai.ask(q)
        assert result.value == "A"
        assert result.selected_option is opts[0]

    async def test_freeform_returns_default(self) -> None:
        aai = AutoApproveInterviewer()
        default = Answer(value="default answer")
        q = Question(
            text="Enter text",
            type=QuestionType.FREEFORM,
            default=default,
        )
        result = await aai.ask(q)
        assert result.value == "default answer"

    async def test_freeform_no_default_returns_skipped(self) -> None:
        aai = AutoApproveInterviewer()
        q = Question(text="Enter text", type=QuestionType.FREEFORM)
        result = await aai.ask(q)
        assert result.value == AnswerValue.SKIPPED

    async def test_ask_multiple(self) -> None:
        aai = AutoApproveInterviewer()
        questions = [
            Question(text="Q1?", type=QuestionType.YES_NO),
            Question(text="Q2?", type=QuestionType.CONFIRMATION),
        ]
        results = await aai.ask_multiple(questions)
        assert len(results) == 2
        assert all(r.value == AnswerValue.YES for r in results)

    async def test_inform_is_noop(self) -> None:
        aai = AutoApproveInterviewer()
        result = await aai.inform("message", stage="test")
        assert result is None

    async def test_protocol_compliance(self) -> None:
        aai = AutoApproveInterviewer()
        assert isinstance(aai, Interviewer)


class TestCallbackInterviewer:
    """Tests for CallbackInterviewer."""

    async def test_delegates_to_callback(self) -> None:
        async def my_callback(q: Question) -> Answer:
            return Answer(value=f"answered: {q.text}")

        ci = CallbackInterviewer(my_callback)
        q = Question(text="Hello?", type=QuestionType.FREEFORM)
        result = await ci.ask(q)
        assert result.value == "answered: Hello?"

    async def test_ask_multiple_delegates_each(self) -> None:
        calls: list[str] = []

        async def my_callback(q: Question) -> Answer:
            calls.append(q.text)
            return Answer(value=AnswerValue.YES)

        ci = CallbackInterviewer(my_callback)
        questions = [
            Question(text="Q1", type=QuestionType.YES_NO),
            Question(text="Q2", type=QuestionType.YES_NO),
        ]
        results = await ci.ask_multiple(questions)
        assert len(results) == 2
        assert calls == ["Q1", "Q2"]

    async def test_inform_is_noop(self) -> None:
        async def noop(q: Question) -> Answer:
            return Answer(value=AnswerValue.YES)

        ci = CallbackInterviewer(noop)
        result = await ci.inform("message", stage="test")
        assert result is None

    async def test_protocol_compliance(self) -> None:
        async def noop(q: Question) -> Answer:
            return Answer(value=AnswerValue.YES)

        ci = CallbackInterviewer(noop)
        assert isinstance(ci, Interviewer)


class TestRecordingInterviewer:
    """Tests for RecordingInterviewer."""

    async def test_records_interactions(self) -> None:
        qi = QueueInterviewer()
        answer = Answer(value=AnswerValue.YES)
        await qi.responses.put(answer)

        ri = RecordingInterviewer(qi)
        q = Question(text="Deploy?", type=QuestionType.YES_NO, stage="deploy")
        result = await ri.ask(q)

        assert result.value == AnswerValue.YES
        assert len(ri.records) == 1
        assert ri.records[0].question is q
        assert ri.records[0].answer.value == AnswerValue.YES
        assert isinstance(ri.records[0].timestamp, float)

    async def test_records_multiple_interactions(self) -> None:
        qi = QueueInterviewer()
        a1 = Answer(value=AnswerValue.YES)
        a2 = Answer(value=AnswerValue.NO)
        await qi.responses.put(a1)
        await qi.responses.put(a2)

        ri = RecordingInterviewer(qi)
        questions = [
            Question(text="Q1?", type=QuestionType.YES_NO),
            Question(text="Q2?", type=QuestionType.YES_NO),
        ]
        results = await ri.ask_multiple(questions)
        assert len(results) == 2
        assert len(ri.records) == 2
        assert ri.records[0].question.text == "Q1?"
        assert ri.records[1].question.text == "Q2?"

    async def test_delegates_inform(self) -> None:
        qi = QueueInterviewer()
        ri = RecordingInterviewer(qi)
        await ri.inform("Status", stage="build")
        assert qi.messages == ["Status"]

    async def test_protocol_compliance(self) -> None:
        qi = QueueInterviewer()
        ri = RecordingInterviewer(qi)
        assert isinstance(ri, Interviewer)
