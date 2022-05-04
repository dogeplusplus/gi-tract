import pytest
import numpy as np

from utils.extract import running_length, convert_to_rle


@pytest.mark.parametrize("sequence,expected,msg", [
    (np.array([1, 1, 0, 0, 1, 0, 0, 1]), "0 2 4 1 7 1", "Segments before, mid and end"),
    (np.array([]), "", "Empty sequence"),
    (np.array([0, 0, 0]), "", "No foreground"),
])
def test_running_length(sequence, expected, msg):
    rle = running_length(sequence)
    assert rle == expected, msg


def test_convert_to_rle():
    mask = np.array([
        [3, 0, 0, 1, 1],
        [3, 0, 2, 1, 1],
        [0, 0, 2, 0, 0],
        [0, 0, 2, 0, 0],
    ])

    index = 1
    expected = [
        {
            "id": index,
            "class": "large_bowel",
            "predicted": "12 2 16 2",
        },
        {
            "id": index,
            "class": "small_bowel",
            "predicted": "9 3",
        },
        {
            "id": index,
            "class": "stomach",
            "predicted": "0 2",
        }
    ]

    df = convert_to_rle(mask, index)
    rle = df.to_dict("records")
    for x, y in zip(expected, rle):
        assert x == y
