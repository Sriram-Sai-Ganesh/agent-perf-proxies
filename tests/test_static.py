from core import store
from core.llm import Reply, Sampling
from core.run import expand, item_id_of, items_file, items_of
from core.runner import run_static
from core.schema import StaticItem, StepRow, from_record, to_record
from core.static.build import bbh_items, gsm8k_final_number, gsm8k_items, mmlu_pro_items, sample_evenly
from core.static.score import build_prompt, extract_answer, is_correct

S = Sampling()


def item(answer_type: str, answer: str, choices: list[str] | None = None, item_id: str = "x/1") -> StaticItem:
    return StaticItem(item_id, "x", "s", "What?", choices or [], answer, answer_type)


class ReplyClient:
    """Answers every question with the same text, as a model would, and records what it was sent."""

    def __init__(self, text: str):
        self.text = text
        self.sent: list = []

    def chat(self, messages, seed, sampling, extra=None):
        self.sent.append((messages, extra))
        return Reply(
            self.text,
            prompt_tokens=30,
            completion_tokens=12,
            cached_tokens=0,
            latency_s=0.5,
            raw={"finish_reason": "stop"},
        )


def test_build_prompt_lists_choices_and_says_how_to_answer():
    (msg,) = build_prompt(item("letter", "B", ["red", "blue"]))
    assert msg["role"] == "user"
    assert "What?\n\nA. red\nB. blue\n\n" in msg["content"] and msg["content"].endswith("`Answer: <letter>`.")
    assert build_prompt(item("number", "4"))[0]["content"].endswith("`Answer: <number>`.")
    text_prompt = build_prompt(item("text", "True"))[0]["content"]
    assert "A. " not in text_prompt and text_prompt.endswith("`Answer: <answer>`.")


def test_extract_answer_from_the_last_answer_line():
    assert extract_answer("Let me think.\nAnswer: B\nAnswer: C", "letter") == "C"
    assert extract_answer("so\nanswer: 42", "number") == "42"
    assert extract_answer("**Answer:** C", "letter") == "C"
    assert extract_answer("**Answer: C**", "letter") == "C"
    assert extract_answer("## Answer: **(D)**", "letter") == "(D)"
    assert extract_answer("Answer: $1,234.00\nthanks", "number") == "$1,234.00"
    assert extract_answer("Answer: (A)", "text") == "(A)"


def test_extract_answer_falls_back_to_the_last_answer_like_thing():
    assert extract_answer("It is probably (C)", "letter") == "C"
    assert extract_answer("I think B. Then again, it is D.", "letter") == "D"
    assert extract_answer("The answer is C)", "letter") == "C"
    assert extract_answer("I considered it for a while", "letter") is None
    assert extract_answer("First 12 then 3,400 and finally 7.5.", "number") == "7.5"
    assert extract_answer("From 2020-2021 it grew by 18", "number") == "18"
    assert extract_answer("no digits here", "number") is None
    assert extract_answer("first line\n\nlast line\n\n", "text") == "last line"
    assert extract_answer("   \n", "text") is None
    assert extract_answer("", "letter") is None


def test_an_empty_answer_line_falls_back():
    assert extract_answer("Answer:\nIt is (B)", "letter") == "B"


def test_is_correct_letter():
    gold = item("letter", "C")
    assert all(is_correct(gold, got) for got in ["C", "(C)", "C.", "C) foo", " c "])
    assert not is_correct(gold, "B") and not is_correct(gold, "Cat") and not is_correct(gold, None)


def test_is_correct_number():
    gold = item("number", "1234")
    assert all(is_correct(gold, got) for got in ["1234", "$1,234.00", "1,234", "1234.0", "about 1234 dollars"])
    assert not is_correct(gold, "1235") and not is_correct(gold, "twelve") and not is_correct(gold, None)
    assert is_correct(item("number", "-3"), "-3.0") and is_correct(item("number", "0.5"), "0.50")
    assert not is_correct(item("number", "12"), "120")


def test_is_correct_text():
    assert is_correct(item("text", "(A)"), "A") and is_correct(item("text", "A"), "(A)")
    assert is_correct(item("text", "True"), "true.") and is_correct(item("text", "valid"), "  Valid ")
    assert is_correct(item("text", "no"), "No.")
    assert not is_correct(item("text", "True"), "False") and not is_correct(item("text", "True"), None)
    assert not is_correct(item("text", "(A)"), "(B)")
    assert is_correct(item("text", "] ]"), "]  ]")


def test_is_correct_bbh_option_compares_the_leading_option_letter():
    assert is_correct(item("text", "(C)"), "(C) the robin") and is_correct(item("text", "(C)"), "C) the robin")
    assert is_correct(item("text", "(K)"), "K.")
    assert not is_correct(item("text", "(C)"), "(B) the robin")
    assert not is_correct(item("text", "(A)"), "A robin")  # the article "A", not option A


MMLU_ROWS = [
    {"question_id": 7, "question": "Q7", "options": ["a", "b", "c"], "answer": "B", "category": "law", "src": "x"},
    {"question_id": 9, "question": "Q9", "options": ["d", "e"], "answer": "A", "category": "math", "src": "y"},
]


def test_mmlu_pro_items():
    law, math = mmlu_pro_items(MMLU_ROWS)
    assert (law.item_id, law.source, law.subset) == ("mmlu_pro/7", "mmlu_pro", "law")
    assert (law.question, law.choices, law.answer, law.answer_type) == ("Q7", ["a", "b", "c"], "B", "letter")
    assert math.item_id == "mmlu_pro/9" and math.subset == "math"


def test_gsm8k_items():
    assert gsm8k_final_number("Work.\n#### 1,234") == "1234"
    assert gsm8k_final_number("Cost is 5 #### 7\n#### -3.5") == "-3.5"
    rows = [{"question": "Q0", "answer": "add\n#### 72"}, {"question": "Q1", "answer": "mul\n#### 1,000"}]
    first, second = gsm8k_items(rows)
    assert (first.item_id, first.source, first.subset, first.choices) == ("gsm8k/0", "gsm8k", "main", [])
    assert (first.answer, first.answer_type) == ("72", "number") and second.item_id == "gsm8k/1"
    assert second.answer == "1000"


def test_bbh_items_number_rows_within_each_task():
    rows = {
        "web_of_lies": [{"input": "i0", "target": "Yes"}],
        "navigate": [{"input": "n0", "target": "No"}, {"input": "n1", "target": "Yes"}],
    }
    items = bbh_items(rows)
    assert [it.item_id for it in items] == ["bbh/navigate/0", "bbh/navigate/1", "bbh/web_of_lies/0"]
    assert (items[0].source, items[0].subset, items[0].question) == ("bbh", "navigate", "n0")
    assert (items[0].answer, items[0].answer_type, items[0].choices) == ("No", "text", [])


def made_up_pool() -> list[StaticItem]:
    sizes = {"a": 10, "b": 10, "c": 2, "d": 10}
    return [StaticItem(f"s/{s}/{i}", "s", s, "q", [], "x", "text") for s, n in sizes.items() for i in range(n)]


def test_sample_evenly_spreads_over_subsets():
    pool = made_up_pool()
    got = sample_evenly(pool, 9, seed=0)  # 4 subsets: 2 each, one subset gets the remaining 1
    counts = {s: sum(it.subset == s for it in got) for s in "abcd"}
    assert sum(counts.values()) == 9 and sorted(counts.values()) == [2, 2, 2, 3]
    assert len({it.item_id for it in got}) == 9


def test_sample_evenly_is_deterministic_and_ignores_input_order():
    pool = made_up_pool()
    first = sample_evenly(pool, 9, seed=0)
    assert sample_evenly(pool, 9, seed=0) == first
    assert sample_evenly(list(reversed(pool)), 9, seed=0) == first
    assert sample_evenly(pool[5:] + pool[:5], 9, seed=0) == first
    assert [it.item_id for it in first] == sorted(it.item_id for it in first)
    assert any(sample_evenly(pool, 9, seed=s) != first for s in range(1, 6))


def test_sample_evenly_gives_all_of_a_small_subset():
    got = sample_evenly(made_up_pool(), 16, seed=3)  # 4 each, but subset c has only 2
    assert sum(it.subset == "c" for it in got) == 2 and len(got) == 14


def test_run_static_correct_and_wrong():
    gold = item("letter", "B", ["x", "y"], item_id="mmlu_pro/5")
    client = ReplyClient("It is the second one.\nAnswer: B")
    (row,) = run_static(client, "a", False, gold, 0, S, "r")
    assert isinstance(row, StepRow) and row.kind == "static" and row.item_id == "mmlu_pro/5" and row.step == 0
    assert (row.action, row.valid, row.reward, row.score, row.done) == ("B", True, 1.0, 1.0, True)
    assert (row.prompt_tokens, row.completion_tokens, row.latency_s) == (30, 12, 0.5)
    assert row.observation == "What?" and row.prompt == build_prompt(gold) and row.completion.endswith("Answer: B")
    assert row.meta == {
        "source": "x",
        "subset": "s",
        "answer_type": "letter",
        "gold": "B",
        "extracted": "B",
        "correct": True,
        "finish_reason": "stop",
    }
    assert client.sent[0][1] == {"chat_template_kwargs": {"enable_thinking": False}}
    (wrong,) = run_static(ReplyClient("Answer: A"), "a", True, gold, 0, S, "r")
    assert (wrong.reward, wrong.score, wrong.meta["correct"], wrong.valid) == (0.0, 0.0, False, True)


def test_run_static_without_an_answer_is_invalid_and_wrong():
    (row,) = run_static(ReplyClient("I cannot say."), "a", False, item("letter", "B"), 0, S, "r")
    assert (row.action, row.valid, row.score, row.meta["extracted"], row.meta["correct"]) == (
        "",
        False,
        0.0,
        None,
        False,
    )


def test_run_static_row_survives_the_store(tmp_path):
    (row,) = run_static(ReplyClient("Answer: 4"), "a", False, item("number", "4", item_id="gsm8k/3"), 0, S, "r")
    path = str(tmp_path / "steps.jsonl")
    store.append(path, [row])
    assert [from_record(StepRow, r) for r in store.read(path)] == [row]
    assert store.done_keys(path) == {("a", "gsm8k/3", 0)}


def test_items_of_reads_a_static_pool(tmp_path):
    pool = [item("letter", "A", ["u", "v"], item_id="p/1"), item("number", "3", item_id="p/2")]
    path = tmp_path / "pool.jsonl"
    store.append(str(path), pool)
    cfg = {"run_id": "r", "static": str(path)}
    assert items_of(cfg) == pool
    assert items_file(cfg) == str(path)
    assert [item_id_of(it) for it in items_of(cfg)] == ["p/1", "p/2"]
    assert from_record(StaticItem, to_record(pool[0])) == pool[0]


def test_items_file_expands_data_root(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    assert items_file({"static": "$DATA_ROOT/static/pool.jsonl"}) == expand("$DATA_ROOT/static/pool.jsonl")
    assert items_file({"probes": "$DATA_ROOT/p.jsonl", "tasks": "t"}) == expand("$DATA_ROOT/p.jsonl")
    assert items_file({"tasks": "$DATA_ROOT/t.parquet"}) == expand("$DATA_ROOT/t.parquet")
