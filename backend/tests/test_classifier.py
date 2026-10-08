import pytest

from jarvis.core.classifier import classify
from jarvis.core.types import TaskType


@pytest.mark.parametrize(
    ("text", "task"),
    [
        ("Escreva um código Python para ler um CSV", TaskType.CODE),
        ("Tem um bug nessa função JavaScript", TaskType.CODE),
        ("Quanto é 37 * 12?", TaskType.MATH),
        ("Resolva a equação 2x + 3 = 7", TaskType.MATH),
        ("Pesquise e compare os notebooks mais leves", TaskType.RESEARCH),
        ("JARVIS, explique Docker.", TaskType.CHAT),
        ("Bom dia! Como você está?", TaskType.CHAT),
    ],
)
def test_classify(text, task):
    assert classify(text) is task


def test_accents_do_not_matter():
    assert classify("CÓDIGO") is classify("codigo") is TaskType.CODE
