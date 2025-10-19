from services.pamphlet_planner import plan_answer


def test_fact_question_prefers_short_direct():
    plan = plan_answer("開園はいつですか？", {"city": "goto"})
    assert plan.intent == "fact"
    assert plan.style == "short_direct"


def test_overview_question_prefers_medium():
    plan = plan_answer("福江城について概要を教えて", {"city": "goto"})
    assert plan.intent == "overview"
    assert plan.style == "medium_structured"


def test_detail_request_prefers_long():
    plan = plan_answer("初めて行くのでモデルコースを詳しく知りたい", {"city": "goto"})
    assert plan.style == "long_explanatory"
