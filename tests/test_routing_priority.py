from services.route_selector import RouteDecision, select_route


class Probe:
    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls = []

    def __call__(self, text: str):
        self.calls.append(text)
        if not self._responses:
            return False
        return self._responses.pop(0)


def test_special_route_has_highest_priority():
    special = Probe((True, "weather"))
    tourism = Probe(True)
    pamphlet = Probe(True)

    decision = select_route(
        "今日の天気は？",
        special_probe=special,
        tourism_probe=tourism,
        pamphlet_probe=pamphlet,
    )

    assert isinstance(decision, RouteDecision)
    assert decision.route == "special"
    assert tourism.calls == []
    assert pamphlet.calls == []


def test_multiple_special_intents_share_priority():
    special = Probe((True, "運行状況"))
    tourism = Probe()
    pamphlet = Probe()

    decision = select_route(
        "運行状況を教えて",
        special_probe=special,
        tourism_probe=tourism,
        pamphlet_probe=pamphlet,
    )

    assert decision.route == "special"
    assert tourism.calls == []
    assert pamphlet.calls == []

    special = Probe((True, "展望所マップ"))
    decision = select_route(
        "展望所マップ",
        special_probe=special,
        tourism_probe=tourism,
        pamphlet_probe=pamphlet,
    )
    assert decision.route == "special"


def test_tourism_when_special_fails():
    special = Probe(False)
    tourism_payload = {"title": "観光スポット"}
    tourism = Probe((True, tourism_payload))
    pamphlet = Probe()

    decision = select_route(
        "福江島の観光地", special_probe=special, tourism_probe=tourism, pamphlet_probe=pamphlet
    )

    assert decision.route == "tourism"
    assert decision.payload == tourism_payload
    assert pamphlet.calls == []


def test_pamphlet_when_tourism_missing():
    special = Probe(False)
    tourism = Probe(False)
    pamphlet = Probe((True, {"city": "goto"}))

    decision = select_route(
        "五島市の歴史", special_probe=special, tourism_probe=tourism, pamphlet_probe=pamphlet
    )

    assert decision.route == "pamphlet"
    assert decision.payload == {"city": "goto"}


def test_no_answer_when_all_handlers_decline():
    decision = select_route(
        "未知の質問です", special_probe=lambda _: False, tourism_probe=lambda _: False, pamphlet_probe=lambda _: False
    )

    assert decision.route == "no_answer"
    assert decision.payload is None
