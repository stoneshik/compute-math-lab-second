import numpy
from prettytable import PrettyTable
from sympy import latex, diff, Symbol, Abs
from sympy.plotting import plot3d


class SystemEquation:
    """
    Класс для системы уравнений
    """

    def __init__(self, equations: tuple, variables: tuple = (Symbol('x1'), Symbol('x2'))) -> None:
        self.equations = equations
        self.variables = variables

    def get_string(self) -> str:
        return f"f1(x1, x2)={latex(self.equations[0])}=0\n   f2(x1, x2)={latex(self.equations[1])}=0"


class SimpleIterationMethodForSystem:
    """
    Класс для метода простой итерации для системы уравнений
    """

    def __init__(self,
                 system_equations: SystemEquation,
                 first_approach: tuple,
                 interval_for_x1: tuple,
                 interval_for_x2: tuple,
                 epsilon: float = 0.001) -> None:
        self._system_equations = system_equations
        self._first_approach = first_approach
        self._interval_for_x1 = interval_for_x1
        self._interval_for_x2 = interval_for_x2
        self._epsilon = epsilon
        self._system_phi_func = (
                self._system_equations.equations[0] + self._system_equations.variables[0],
                self._system_equations.equations[1] + self._system_equations.variables[1]
        )
        self._field_names_table = ['№ итерации',
                                   'X1(k)', 'X2(k)',
                                   'X1(k+1)', 'X2(k+1)',
                                   '|X1(k+1) - X1(k)|', '|X2(k+1) - X2(k)|']
        self._found_roots = (0.0, 0.0)
        self._vector_errors = (0.0, 0.0)
        self._num_iteration_for_found: int = 0

    def check_convergence(self) -> bool:
        """
        Проверка сходимости
        :return: Сходится или нет
        """
        x_1 = self._system_equations.variables[0]
        x_2 = self._system_equations.variables[1]
        all_variants_system_phi_func = (
            (
                self._system_equations.equations[0] + self._system_equations.variables[0],
                self._system_equations.equations[1] + self._system_equations.variables[1]
            ),
            (
                -(self._system_equations.equations[0]) + self._system_equations.variables[0],
                self._system_equations.equations[1] + self._system_equations.variables[1]
            ),
            (
                self._system_equations.equations[0] + self._system_equations.variables[0],
                -(self._system_equations.equations[1]) + self._system_equations.variables[1]
            ),
            (
                -(self._system_equations.equations[0]) + self._system_equations.variables[0],
                -(self._system_equations.equations[1]) + self._system_equations.variables[1]
            ),
        )
        number_intervals: int = 2
        for system_phi_func in all_variants_system_phi_func:
            self._system_phi_func = system_phi_func
            first_equation = Abs(diff(system_phi_func[0], x_1)) + Abs(diff(system_phi_func[0], x_2))
            second_equation = Abs(diff(system_phi_func[1], x_1)) + Abs(diff(system_phi_func[1], x_2))
            for x1_value in numpy.linspace(self._interval_for_x1[0], self._interval_for_x1[1], number_intervals):
                for x2_value in numpy.linspace(self._interval_for_x2[0], self._interval_for_x2[1], number_intervals):
                    a = first_equation.subs({x_1: x1_value, x_2: x2_value}).evalf()
                    b = second_equation.subs({x_1: x1_value, x_2: x2_value}).evalf()
                    if first_equation.subs({x_1: x1_value, x_2: x2_value}).evalf() >= 1 or \
                            second_equation.subs({x_1: x1_value, x_2: x2_value}).evalf() >= 1:
                        break
                else:
                    continue
                break
            else:
                return True
        return False

    def calc(self) -> PrettyTable:
        table = PrettyTable()
        table.field_names = self._field_names_table
        x_1 = self._system_equations.variables[0]
        x_2 = self._system_equations.variables[1]
        approach_k = self._first_approach
        approach_k_plus_1 = (
            self._system_phi_func[0].subs({x_1: approach_k[0], x_2: approach_k[1]}).evalf(),
            self._system_phi_func[1].subs({x_1: approach_k[0], x_2: approach_k[1]}).evalf()
        )
        table.add_row(['0', approach_k[0], approach_k[1],
                       approach_k_plus_1[0], approach_k_plus_1[1],
                       abs(approach_k_plus_1[0] - approach_k[0]),
                       abs(approach_k_plus_1[1] - approach_k[1])])
        num_iter: int = 1
        while abs(approach_k_plus_1[0] - approach_k[0]) > self._epsilon and \
                abs(approach_k_plus_1[1] - approach_k[1]) > self._epsilon:
            approach_k = approach_k_plus_1
            approach_k_plus_1 = (
                self._system_phi_func[0].subs({x_1: approach_k[0], x_2: approach_k[1]}).evalf(),
                self._system_phi_func[1].subs({x_1: approach_k[0], x_2: approach_k[1]}).evalf()
            )
            table.add_row([num_iter, approach_k[0], approach_k[1],
                           approach_k_plus_1[0], approach_k_plus_1[1],
                           abs(approach_k_plus_1[0] - approach_k[0]),
                           abs(approach_k_plus_1[1] - approach_k[1])])
            num_iter += 1
        self._found_roots = approach_k_plus_1
        self._vector_errors = (
            abs(approach_k_plus_1[0] - approach_k[0]),
            abs(approach_k_plus_1[1] - approach_k[1])
        )
        self._num_iteration_for_found = num_iter
        return table

    def check_calc(self):
        print("Для оценки полученного ответа рассчитан вектор невязки")
        x_1 = self._system_equations.variables[0]
        x_2 = self._system_equations.variables[1]
        r1 = self._system_equations.equations[0].subs(
            {x_1: self._found_roots[0], x_2: self._found_roots[1]}).evalf()
        r2 = self._system_equations.equations[1].subs(
            {x_1: self._found_roots[0], x_2: self._found_roots[1]}).evalf()
        print(f"r[1] = {r1}\nr[2] = {r2}")

    def draw(self) -> None:
        plot3d(
            self._system_equations.equations[0],
            self._system_equations.equations[1],
            (self._system_equations.variables[0], self._interval_for_x1[0], self._interval_for_x1[1]),
            (self._system_equations.variables[1], self._interval_for_x2[0], self._interval_for_x2[1])
        )

    def output_result(self) -> str:
        return f"Найденные корни: x[1]={self._found_roots[0]} x[2]={self._found_roots[1]}\n" + \
               f"Вектор погрешностей: ({self._vector_errors[0]}, {self._vector_errors[1]})" + \
               f"\nЧисло итераций: {self._num_iteration_for_found}"


def input_intervals(approach_value: float) -> tuple:
    while True:
        min_value, max_value = (
            float(i) for i in input("Введите значения минимума и максимума через пробел...\n").split())
        if min_value == max_value:
            print("Введенные значения должны отличаться")
            continue
        if min_value > max_value:
            print("Сначала должен быть введен минимум, а потом максимум")
            continue
        if approach_value < min_value or approach_value > max_value:
            print("Введенный интервал не покрывает приближение")
            continue
        return min_value, max_value


def input_data(systems_equation) -> SimpleIterationMethodForSystem:
    system_equation = None
    while True:
        print("Выберите уравнение:")
        [print(f"{i + 1}. {system_equation_iter.get_string()}") for i, system_equation_iter in
         enumerate(systems_equation)]
        equation_num = int(input("Введите номер выбранной системы уравнения...\n"))
        if equation_num < 1 or equation_num > len(systems_equation):
            print("Номер системы уравнений не найден, повторите ввод")
            continue
        system_equation = systems_equation[equation_num - 1]
        break
    while True:
        print("Введите начальное приближение:")
        first_approach: tuple = tuple([float(i) for i in input("Введите значения x1 и x2 через пробел...\n").split()])
        if len(first_approach) != 2:
            print("Должно быть введено 2 значения")
            continue
        break
    print("Определение области сходимости G")
    print("Введите значения интервала для x1")
    interval_for_x1: tuple = input_intervals(first_approach[0])
    print("Введите значения интервала для x2")
    interval_for_x2: tuple = input_intervals(first_approach[1])
    while True:
        epsilon = input(
            "Введите погрешность вычислений (чтобы оставить значение по умолчанию - 0,001 нажмите Enter)...\n")
        if epsilon == '':
            return SimpleIterationMethodForSystem(system_equation, first_approach, interval_for_x1, interval_for_x2)
        epsilon = float(epsilon)
        if epsilon <= 0:
            print("Значение погрешности должно быть больше нуля")
            continue
        return SimpleIterationMethodForSystem(
            system_equation, first_approach, interval_for_x1, interval_for_x2, epsilon)


def main_for_system_equations():
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    systems_equation = (
        SystemEquation((
            0.1 * x1 ** 2 + x1 + 0.2 * x2 ** 2 - 0.3,
            0.2 * x1 ** 2 + x2 + 0.1 * x1 * x2 - 0.7
        )),
        SystemEquation((  # (-1.39, 4.17), (1.331, -3.673)
            3 * x1 ** 3 + 0.2 * x1 + 2 * x2,
            x1 - 3 * x1 * x2 - 16
        )),
        SystemEquation((  # (-0.905, 0.381), (1, 0)
            3 * (x1 - x2) ** 2 + 2 * (x1 + 2 * x2) ** 2 - 5,
            2 * (x1 + 2 * x2) - x1 + x2 - 1
        ))
    )
    solution_method = input_data(systems_equation)
    if solution_method is None:
        return
    if not solution_method.check_convergence():
        print("Уравнение не сходится в выбранной области")
        solution_method.draw()
        return
    table: PrettyTable = solution_method.calc()
    if table is None:
        solution_method.draw()
        return
    print(table)
    print(solution_method.output_result())
    solution_method.check_calc()
    solution_method.draw()
