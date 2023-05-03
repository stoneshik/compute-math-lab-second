import math
from abc import ABC, abstractmethod

import numpy
import matplotlib
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sympy import (
    init_printing,
    diff,
    latex,
    Symbol,
)


class Equation:
    """
    Класс обертка для уравнений
    """
    def __init__(self, equation_func) -> None:
        self.equation_func = equation_func

    def get_string(self) -> str:
        return latex(self.equation_func)

    def get_diff(self):
        return diff(self.equation_func)


class SolutionMethod(ABC):
    """
    Базовый абстрактный класс для классов реализаций методов решения ур-й
    """
    def __init__(self, equation: Equation, a: float, b: float, epsilon: float, field_names_table: list) -> None:
        assert a != b, "Значения a и b должны быть различны"
        assert a < b, "Значение a должно быть меньше b"
        assert epsilon > 0, "Значение эпсилон должно быть больше нуля"
        self._equation = equation
        self._a = a
        self._b = b
        self._epsilon = epsilon
        self._field_names_table = field_names_table

    def check(self) -> bool:
        func = self._equation.equation_func
        x = Symbol('x')
        value_a: float = func.subs(x, self._a).evalf()
        value_b: float = func.subs(x, self._b).evalf()
        # проверка на монотонность для производной на интервале
        number_intervals: int = 100
        func_diff = self._equation.get_diff()
        first_value_diff: float = func_diff.subs(x, self._a).evalf()
        for i in numpy.arange(self._a, self._b, abs(self._b - self._a) / number_intervals):
            if first_value_diff * func_diff.subs(x, i).evalf() < 0:
                print(f"На отрезке [{self._a}; {self._b}] более одного корня")
                return False
        # проверка на разность знаков на концах интервала
        if value_a * value_b > 0:
            print(f"На отрезке [{self._a}; {self._b}] отсутсвуют корни")
            return False
        return True

    def draw(self) -> None:
        plt.figure()
        plt.xlabel(r'$x$', fontsize=14)
        plt.ylabel(r'$F(x)$', fontsize=14)
        plt.title(r"График функции $F(x)$")
        x = Symbol('x')
        x_values = numpy.arange(self._a - 1, self._b + 1, 0.01)
        y_values = [self._equation.equation_func.subs(x, x_iter) for x_iter in x_values]
        plt.plot(x_values, y_values)
        plt.show()

    @abstractmethod
    def calc(self) -> (PrettyTable, None):
        pass


class ChordMethod(SolutionMethod):
    """
    Класс для реализации метода Хорд
    """
    name: str = 'метод хорд'

    def __init__(self, equation: Equation, a: float, b: float, epsilon: float = 0.001) -> None:
        super().__init__(
            equation, a, b, epsilon,
            ['№ итерации', 'a', 'b', 'x', 'F(a)', 'F(b)', 'F(x)', '|Xn+1 - Xn|']
        )

    def calc(self) -> (PrettyTable, None):
        table = PrettyTable()
        table.field_names = self._field_names_table
        func = self._equation.equation_func
        x = Symbol('x')
        a_i: float = self._a
        b_i: float = self._b
        x_i: float = a_i - (
                (b_i - a_i) / (func.subs(x, b_i).evalf() - func.subs(x, a_i).evalf())) * func.subs(x, a_i).evalf()
        f_xi: float = func.subs(x, x_i).evalf()
        f_ai: float = func.subs(x, a_i).evalf()
        f_bi: float = func.subs(x, b_i).evalf()
        table.add_row(['0', a_i, b_i, x_i, f_ai, f_bi, f_xi, abs(a_i - x_i)])
        num_iter: int = 1
        while abs(f_xi) > self._epsilon:
            if f_ai * f_xi < 0:
                b_i = x_i
                f_bi = func.subs(x, b_i).evalf()
            else:
                a_i = x_i
                f_ai = func.subs(x, a_i).evalf()
            x_i = (a_i * f_bi - b_i * f_ai) / (f_bi - f_ai)
            f_xi = func.subs(x, x_i).evalf()
            table.add_row([num_iter, a_i, b_i, x_i, f_ai, f_bi, f_xi, abs(a_i - x_i)])
            num_iter += 1
        return table


class NewtonMethod(SolutionMethod):
    """
    Класс для реализации метода Ньютона
    """
    name: str = 'метод Ньютона'

    def __init__(self, equation: Equation, a: float, b: float, epsilon: float = 0.001) -> None:
        super().__init__(
            equation, a, b, epsilon,
            ['№ итерации', 'Xn', 'f(Xn)', "f'(Xn)", 'Xn+1', '|Xn+1 - Xn|']
        )

    def check(self) -> bool:
        result = super().check()
        if not result:
            return False
        x = Symbol('x')
        # проверка на то равна ли производная нулю на интервале
        func_diff = self._equation.get_diff()
        measurement: float = 0.00001
        if abs(func_diff.subs(x, self._a)) <= measurement or abs(func_diff.subs(x, self._b)) <= measurement:
            print(f"На отрезке [{self._a}; {self._b}] значение прозводной близко к нулю, что не позволяет решить уравнение методом Ньютона")
            return False
        return True

    def calc(self) -> (PrettyTable, None):
        table = PrettyTable()
        table.field_names = self._field_names_table
        func = self._equation.equation_func
        func_diff = self._equation.get_diff()
        func_diff_second = diff(func_diff)
        x = Symbol('x')
        a_i: float = self._a
        b_i: float = self._b
        if func.subs(x, a_i).evalf() * func_diff_second.subs(x, a_i).evalf() > 0:
            x_n: float = a_i
        else:
            x_n: float = b_i
        f_x_n: float = func.subs(x, x_n).evalf()
        f_x_n_diff: float = func_diff.subs(x, x_n).evalf()
        x_n_plus_1: float = x_n - (f_x_n / f_x_n_diff)
        table.add_row(['0', x_n, f_x_n, f_x_n_diff, x_n_plus_1, abs(x_n_plus_1 - x_n)])
        num_iter: int = 1
        while abs(f_x_n) > self._epsilon:
            x_n = x_n_plus_1
            f_x_n = func.subs(x, x_n).evalf()
            f_x_n_diff = func_diff.subs(x, x_n).evalf()
            x_n_plus_1 = x_n - (f_x_n / f_x_n_diff)
            table.add_row([num_iter, x_n, f_x_n, f_x_n_diff, x_n_plus_1, abs(x_n_plus_1 - x_n)])
            num_iter += 1
        return table


class SimpleIterationMethod(SolutionMethod):
    """
    Класс для реализации метода простой итерации
    """
    name: str = 'метод простой итерации'

    def __init__(self, equation: Equation, a: float, b: float, epsilon: float = 0.001) -> None:
        super().__init__(
            equation, a, b, epsilon,
            ['№ итерации', 'Xi', 'Xi+1', 'φ(Xi+1)', 'f(Xi+1)', '|Xi+1 - Xi|']
        )

    def calc(self) -> (PrettyTable, None):
        table = PrettyTable()
        table.field_names = self._field_names_table
        func = self._equation.equation_func
        func_diff = self._equation.get_diff()
        x = Symbol('x')
        a_i: float = self._a
        b_i: float = self._b
        max_diff_value: float = 0.0
        number_intervals: int = 100
        for i in numpy.arange(self._a, self._b, abs(self._b - self._a) / number_intervals):
            if abs(func_diff.subs(x, i).evalf()) > max_diff_value:
                max_diff_value = abs(i)
        lambda_coefficient = -max_diff_value if abs(max_diff_value) < 1 else -(1 / max_diff_value)
        phi_function = x + lambda_coefficient * func
        phi_function_diff = 1 + lambda_coefficient * func_diff
        # проверка сходимости
        if abs(phi_function_diff.subs(x, a_i)) >= 1 or abs(phi_function_diff.subs(x, b_i)) >= 1:
            print("Условие сходимости для выбранного интервала не выполняется")
            return None
        x_i: float = a_i
        x_i_plus_1: float = phi_function.subs(x, x_i)
        phi_x_i_plus_1: float = phi_function.subs(x, x_i_plus_1)
        f_x_i_plus_1: float = func.subs(x, x_i_plus_1)
        table.add_row(['0', x_i, x_i_plus_1, phi_x_i_plus_1, f_x_i_plus_1, abs(x_i_plus_1 - x_i)])
        num_iter: int = 1
        while abs(x_i_plus_1 - x_i) > self._epsilon:
            x_i = x_i_plus_1
            x_i_plus_1 = phi_function.subs(x, x_i)
            phi_x_i_plus_1 = phi_function.subs(x, x_i_plus_1)
            if abs(phi_x_i_plus_1) >= 1:
                print("Условие сходимости для выбранного интервала не выполняется")
                return None
            f_x_i_plus_1 = func.subs(x, x_i_plus_1)
            table.add_row([num_iter, x_i, x_i_plus_1, phi_x_i_plus_1, f_x_i_plus_1, abs(x_i_plus_1 - x_i)])
            num_iter += 1
        return table


def input_from_console(equations, solution_methods) -> SolutionMethod:
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
    while True:
        print("Выберите метод решения")
        [print(f"{i + 1}. {solution_method_iter.name}") for i, solution_method_iter in enumerate(solution_methods)]
        solution_num = int(input("Введите номер выбранного метода решения...\n"))
        if solution_num < 1 or solution_num > len(solution_methods):
            print("Номер метода не найден, повторите ввод")
            continue
        solution_method = solution_methods[solution_num - 1]
        break
    while True:
        epsilon = input(
            "Введите погрешность вычислений (чтобы оставить значение по умолчанию - 0,001 нажмите Enter...\n")
        if epsilon == '':
            solution_method = solution_method(equation, a, b)
            break
        epsilon = float(epsilon)
        if epsilon <= 0:
            print("Значение погрешности должно быть больше нуля")
            continue
        solution_method = solution_method(equation, a, b, epsilon)
        break
    return solution_method


def input_from_file(equations, solution_methods) -> SolutionMethod:
    pass


def input_data(equations, solution_methods) -> SolutionMethod:
    while True:
        print("Выберите способ ввода данных")
        print("1. Через консоль\n2. Через файл")
        num_variant = int(input("Введите выбранное номер выбранного варианта..."))
        if 1 > num_variant > 2:
            print("Введен не правильной номер, повторите ввод")
            continue
        break
    if num_variant == 1:
        return input_from_console(equations, solution_methods)
    return input_from_file(equations, solution_methods)


def output(table: PrettyTable) -> None:
    while True:
        print("Выберите способ вывода данных")
        print("1. Через консоль\n2. Через файл")
        num_variant = int(input("Введите выбранное номер выбранного варианта...\n"))
        if 1 > num_variant > 2:
            print("Введен не правильной номер, повторите ввод")
            continue
        break
    if num_variant == 1:
        print(table)
        return
    file_name: str = input("Введите название файла\n")
    with open(file_name, 'w', encoding='utf-8') as file:
        file.write(str(table))


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
    solution_method = input_from_console(equations, solution_methods)
    if not solution_method.check():
        return
    table: PrettyTable = solution_method.calc()
    if table is None:
        return
    output(table)
    solution_method.draw()


if __name__ == '__main__':
    matplotlib.use('TkAgg')
    init_printing()
    main()
