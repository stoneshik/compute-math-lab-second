import math
from abc import ABC, abstractmethod

import numpy.ma
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

    def get_string(self) -> str:
        return latex(self.equation_func)

    def get_diff(self):
        return diff(self.equation_func)


class SolutionMethod(ABC):
    def __init__(self, equation: Equation, a: float, b: float) -> None:
        assert a != b, "Значения a и b должны быть различны"
        assert a < b, "Значение a должно быть меньше b"
        self.equation = equation
        self.a = a
        self.b = b

    def check(self) -> bool:
        func = self.equation.equation_func
        x = Symbol('x')
        value_a: float = func.subs(x, self.a).evalf()
        value_b: float = func.subs(x, self.b).evalf()
        # проверка на монотонность для производной на интервале
        number_intervals: int = 100
        func_diff = self.equation.get_diff()
        first_value_diff: float = func_diff.subs(x, self.a).evalf()
        for i in numpy.arange(self.a, self.b, abs(self.b - self.a) / number_intervals):
            if first_value_diff * func_diff.subs(x, i).evalf() < 0:
                print(f"На отрезке [{self.a}; {self.b}] более одного корня")
                return False
        # проверка на разность знаков на концах интервала
        if value_a * value_b > 0:
            print(f"На отрезке [{self.a}; {self.b}] отсутсвуют корни")
            return False
        return True

    @abstractmethod
    def calc(self):
        pass


class ChordMethod(SolutionMethod):
    """
    Класс для реализации метода Хорд
    """
    name: str = 'метод хорд'

    def __init__(self, equation: Equation, a: float, b: float) -> None:
        super().__init__(equation, a, b)

    def calc(self):
        pass


class NewtonMethod(SolutionMethod):
    """
    Класс для реализации метода Ньютона
    """
    name: str = 'метод Ньютона'

    def __init__(self, equation: Equation, a: float, b: float) -> None:
        super().__init__(equation, a, b)

    def calc(self):
        pass


class SimpleIterationMethod(SolutionMethod):
    """
    Класс для реализации метода простой итерации
    """
    name: str = 'метод простой итерации'

    def __init__(self, equation: Equation, a: float, b: float) -> None:
        super().__init__(equation, a, b)

    def calc(self):
        pass


def main() -> None:
    x = Symbol('x')
    equations = (
        Equation(x ** 3 - 2.92 * x ** 2 + 1.435 * x + 0.791),
    )
    solution_methods = (
        ChordMethod,
        NewtonMethod,
        SimpleIterationMethod
    )
    equation = None
    while True:
        print("Выберите уравнение:")
        [print(f"{i + 1}. {equation_iter.get_string()}") for i, equation_iter in enumerate(equations)]
        equation_num = int(input("Введите номер выбранного уравнения...\n"))
        if equation_num < 1 or equation_num > len(equations):
            print("Номер уравнения не найден, повторите ввод")
            continue
        equation = equations[equation_num - 1]
        break
    a: float = 0
    b: float = 0
    while True:
        print("Выберите границы интервала:")
        a, b = (float(i) for i in input("Введите значения a и b через пробел...\n").split())
        if a == b:
            print("Значения должны быть различны")
            continue
        elif a > b:
            print("Значение a должно быть меньше b")
            continue
        break
    solution_method = None
    while solution_method is None:
        print("Выберите метод решения")
        [print(f"{i + 1}. {solution_method_iter.name}") for i, solution_method_iter in enumerate(solution_methods)]
        solution_num = int(input("Введите номер выбранного метода решения...\n"))
        if solution_num < 1 or solution_num > len(equations):
            print("Номер метода не найден, повторите ввод")
            continue
        solution_method = solution_methods[solution_num - 1]
    solution_method = solution_method(equation, a, b)
    if not solution_method.check():
        return
    solution_method.calc()


if __name__ == '__main__':
    init_printing()
    main()
