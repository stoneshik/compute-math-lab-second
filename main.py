import math

import pandas as pd
import matplotlib.pyplot as plt
from sympy import (
    init_printing,
    diff,
    latex,
    symbols,
    Symbol,
)


class Equation:
    """
    Класс для уравнений
    """
    def __init__(self, equation_func) -> None:
        self.equation_func = equation_func

    def print(self) -> None:
        print(latex(self.equation_func))

    def get_diff(self):
        return diff(self.equation_func)


class ChordMethod:
    """
    Класс для реализации метода Хорд
    """
    def __init__(self, a: float, b: float) -> None:
        self.a = a
        self.b = b


class NewtonMethod:
    """
    Класс для реализации метода Ньютона
    """
    def __init__(self, a: float, b: float) -> None:
        self.a = a
        self.b = b


class SimpleIterationMethod:
    """
    Класс для реализации метода простой итерации
    """
    def __init__(self, a: float, b: float) -> None:
        self.a = a
        self.b = b


def main() -> None:
    x = Symbol('x')
    equations = (
        Equation(x ** 3 - 2.92 * x ** 2 + 1.435 * x + 0.791),
    )
    while True:
        print("Выберите уравнение:")
        [print(f'{i + 1}. {equation.print()}') for i, equation in enumerate(equations)]
        equation_num = int(input("Введите номер выбранного уравнения...\n"))
        if equation_num < 0 or equation_num > len(equations):
            print("Номер уравнения не найден, повторите ввод")
            continue
        equation: Equation = equations[equation_num]
        print("Выберите границы интервала:")
        a, b = (float(i) for i in input("Введите значения a и b через пробел...\n").split())


if __name__ == '__main__':
    init_printing()
    main()
