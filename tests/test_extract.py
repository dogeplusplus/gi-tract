import pytest
import numpy as np

from utils.extract import running_length


@pytest.mark.parametrize("sequence,expected,msg", [
    (np.array([1, 1, 0, 0, 1, 0, 0, 1]), "0 2 4 1 7 1", "Segments before, mid and end"),
    (np.array([]), "", "Empty sequence"),
    (np.array([0, 0, 0]), "", "No foreground"),
])
def test_running_length(sequence, expected, msg):
    rle = running_length(sequence)
    assert rle == expected, msg
