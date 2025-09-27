from services.pamphlet_planner import plan_answer


def test_sender_prefix_removed():
    plan = plan_answer("User: 開催はいつ？", {"city": "goto"})
    assert plan.query == "開催はいつ？"
    assert plan.style == "short_direct"
