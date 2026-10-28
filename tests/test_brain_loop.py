from agent.brain_loop import Spot, cluster, judge_revisit


def f(metre, status, conf=0.8, purpose="inspect", aisle="tunnel-09/aisle-02"):
    return {"aisle": aisle, "metre": metre, "status": status, "purpose": purpose,
            "issues": [{"label": "late blight", "confidence": conf}] if status != "healthy" else []}


def test_disease_triggers_one_revisit_and_merges_nearby():
    spots = []
    due = cluster([f(15.0, "disease"), f(15.5, "disease"), f(3.0, "healthy")], spots)
    assert len(due) == 1 and abs(due[0].metre - 15.0) < 1e-9


def test_single_low_suspect_waits_for_second():
    spots = []
    assert cluster([f(8.0, "suspect", 0.3)], spots) == []
    assert len(cluster([f(8.4, "suspect", 0.3)], spots)) == 1


def test_revisit_confirms_or_clears():
    s = Spot("tunnel-09/aisle-02", 15.0, state="revisiting")
    judge_revisit([f(15.2, "disease", purpose="revisit")], s); assert s.state == "confirmed"
    s2 = Spot("tunnel-09/aisle-02", 8.0, state="revisiting")
    judge_revisit([f(8.1, "healthy", purpose="revisit")], s2); assert s2.state == "cleared"
