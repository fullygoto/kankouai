from services.route_selector import select_route


def test_priority_order_respected():
    calls = []

    def special_probe(text):
        calls.append("special")
        return False

    def tourism_probe(text):
        calls.append("tourism")
        return False

    def pamphlet_probe(text):
        calls.append("pamphlet")
        return True, {"city": "goto"}

    decision = select_route(
        "五島市の歴史を教えて",
        special_probe=special_probe,
        tourism_probe=tourism_probe,
        pamphlet_probe=pamphlet_probe,
    )

    assert decision.route == "pamphlet"
    assert calls == ["special", "tourism", "pamphlet"]


def test_special_short_circuits_following_probes():
    calls = []

    def special_probe(text):
        calls.append("special")
        return True

    def tourism_probe(text):
        calls.append("tourism")
        return True

    decision = select_route(
        "天気を教えて",
        special_probe=special_probe,
        tourism_probe=tourism_probe,
        pamphlet_probe=lambda _: True,
    )

    assert decision.route == "special"
    assert calls == ["special"]
