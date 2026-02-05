import pytest
from aegis.state import Action

def test_action():
    action_1 = Action(name="test_action_1", agent_id="test_agent_id", preconditions=frozenset("A"), positive_effects=frozenset("C"),
                    negative_effects=frozenset("A"))

    assert action_1.name =="test_action_1"
    assert action_1._is_applicable(set("A")) is True
    result = action_1.apply(set("A"))
    assert result == set("C")