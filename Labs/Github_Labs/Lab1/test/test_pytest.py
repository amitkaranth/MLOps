import pytest
from src import calculator


@pytest.mark.parametrize("x, y, expected", [(2, 3, 5), (5, 0, 5), (-1, 1, 0), (-1, -1, -2)])
def test_fun1(x, y, expected):
    assert calculator.fun1(x, y) == expected


def test_fun1_invalid_input():
    with pytest.raises(ValueError):
        calculator.fun1("2", 3)


def test_fun2():
    assert calculator.fun2(2, 3) == -1
    assert calculator.fun2(5,0) == 5
    assert calculator.fun2 (-1, 1) == -2
    assert calculator.fun2 (-1, -1) == 0

@pytest.mark.parametrize("x, y, expected", [(2, 3, 6), (5, 0, 0), (-1, 1, -1), (-1, -1, 1)])
def test_fun3(x, y, expected):
    assert calculator.fun3(x, y) == expected


def test_fun4():
    assert calculator.fun4(2, 3, 5) == 10
    assert calculator.fun4(5,0, -1) == 4
    assert calculator.fun4 (-1, -1, -1) == -3
    
    assert calculator.fun4 (-1, -1, 100) == 98


def test_fun5():
    assert calculator.fun5(2, 3) == 8
    assert calculator.fun5(5, 0) == 1
    assert calculator.fun5(-2, 2) == 4
    assert calculator.fun5(10, 2) == 100


def test_fun6():
    assert calculator.fun6(7, 2) == 1
    assert calculator.fun6(10, 3) == 1
    assert calculator.fun6(-5, 3) == 1
    assert calculator.fun6(8, 4) == 0

