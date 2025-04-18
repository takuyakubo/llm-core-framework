import pytest
from pydantic import ValidationError

from core.graphs.states import NodeState

class DummyState(NodeState):
    x: int = 0
    y: str = 'test'

def test_emit_error():
    state = DummyState(x=1, y='hello')
    err = state.emit_error('error occurred')
    assert isinstance(err, DummyState)
    assert err.error == 'error occurred'
    assert err.x == 1
    assert err.y == 'hello'
    assert state.error == ''
    assert state.x == 1

def test_validation_error():
    with pytest.raises(ValidationError):
        DummyState(x='not an int', y='hello')