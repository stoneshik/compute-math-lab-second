import numpy.ma
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

    def __init__(self, system_equations: SystemEquation, first_approach: tuple, epsilon: float = 0.001) -> None:
        self._system_equations = system_equations
        self._first_approach = first_approach
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

    def _check_convergence(self, first_equation, second_equation, approach) -> bool:
        x_1 = self._system_equations.variables[0]
        x_2 = self._system_equations.variables[1]
        if first_equation.subs({x_1: approach[0], x_2: approach[1]}).evalf() >= 1 or \
                second_equation.subs({x_1: approach[0], x_2: approach[1]}).evalf() >= 1:
            return False
        return True

    def create_phi(self) -> bool:
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
        for system_phi_func in all_variants_system_phi_func:
            self._system_phi_func = system_phi_func
            first_equation = Abs(diff(system_phi_func[0], x_1)) + Abs(diff(system_phi_func[0], x_2))
            second_equation = Abs(diff(system_phi_func[1], x_1)) + Abs(diff(system_phi_func[1], x_2))
            if self._check_convergence(first_equation, second_equation, self._first_approach):
                return True
        return False

    def calc(self) -> (PrettyTable, None):
        table = PrettyTable()
        table.field_names = self._field_names_table
        x_1 = self._system_equations.variables[0]
        x_2 = self._system_equations.variables[1]
        approach_k = self._first_approach
        approach_k_plus_1 = (
            self._system_phi_func[0].subs({x_1: approach_k[0], x_2: approach_k[1]}).evalf(),
            self._system_phi_func[1].subs({x_1: approach_k[0], x_2: approach_k[1]}).evalf()
        )
        first_equation = Abs(diff(self._system_phi_func[0], x_1)) + Abs(diff(self._system_phi_func[0], x_2))
        second_equation = Abs(diff(self._system_phi_func[1], x_1)) + Abs(diff(self._system_phi_func[1], x_2))
        if not self._check_convergence(first_equation, second_equation, approach_k_plus_1):
            return None
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
            if not self._check_convergence(first_equation, second_equation, approach_k_plus_1):
                return None
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
        x1_value: float = self._first_approach[0] if self._first_approach[0] != 0 else 0.01
        x2_value: float = self._first_approach[1] if self._first_approach[1] != 0 else 0.01
        plot3d(
            self._system_equations.equations[0],
            self._system_equations.equations[1],
            (self._system_equations.variables[0], x1_value - abs(x1_value) * 3, x1_value + abs(x1_value) * 3),
            (self._system_equations.variables[1], x2_value - abs(x2_value) * 3, x2_value + abs(x2_value) * 3)
        )

    def output_result(self) -> str:
        return f"Найденные корни: x[1]={self._found_roots[0]} x[2]={self._found_roots[1]}\n" + \
               f"Вектор погрешностей: ({self._vector_errors[0]}, {self._vector_errors[1]})" + \
               f"\nЧисло итераций: {self._num_iteration_for_found}"


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
    while True:
        epsilon = input(
            "Введите погрешность вычислений (чтобы оставить значение по умолчанию - 0,001 нажмите Enter)...\n")
        if epsilon == '':
            return SimpleIterationMethodForSystem(system_equation, first_approach)
        epsilon = float(epsilon)
        if epsilon <= 0:
            print("Значение погрешности должно быть больше нуля")
            continue
        return SimpleIterationMethodForSystem(system_equation, first_approach, epsilon)


def find_roots(systems_equation):
    for i in numpy.ma.arange(-5, 5, 0.05):
        for j in numpy.ma.arange(-5, 5, 0.05):
            solution_method = SimpleIterationMethodForSystem(systems_equation[1], (i, j))
            if not solution_method.create_phi():
                continue
                # print(f"не сходится в x1 {i}, x2 {j}")
            table: PrettyTable = solution_method.calc()
            if table is None:
                continue
            print(f"!!Сходится в x1 {i}, x2 {j}!!")
            print(table)
            print(solution_method.output_result())
            solution_method.check_calc()
    solution_method = SimpleIterationMethodForSystem(systems_equation[1], (1, 1))
    solution_method.draw()


def main_for_system_equations():
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    systems_equation = (
        SystemEquation((
            0.1 * x1 ** 2 + x1 + 0.2 * x2 ** 2 - 0.7,
            0.2 * x1 ** 2 + x2 + 0.1 * x1 * x2 + 0.2
        )),
        SystemEquation((  # (-0.542, -0.443), (-0.507, 0.431)
            -1.28 * x1 ** 2 + x1 - 1.18,
            x2 + 2.15 * x2 ** 2 + 0.2975
        )),
        #2 * x1 ** 2 + x1 + 3 * x2 ** 2 - 3.2,
        #3 * x1 ** 2 + x2 + 2 * x1 * x2 - 5.1
        SystemEquation((  # (0.215662019762792, -0.283442856271809); (0.222396612040603, -0.286924516156778)
            1.14 * x1 ** 2 - x1 + 0.48 * x2 ** 2 + 0.125,
            -2 * x1 ** 2 - x2 - 0.8 * x1 * x2 - 0.24
        ))
    )
    """for i in range(-20, 20):
        for j in range(-20, 20):
            solution_method = SimpleIterationMethodForSystem(systems_equation[1], (i, j))
            if not solution_method.create_phi():
                pass
                # print(f"не сходится в x1 {i}, x2 {j}")
            else:
                print(f"!!Сходится в x1 {i}, x2 {j}!!")
    return"""
    find_roots(systems_equation)
    return
    solution_method = input_data(systems_equation)
    if solution_method is None:
        return
    if not solution_method.create_phi():
        print("Уравнение не сходится для выбранного приближения")
        solution_method.draw()
        return
    table: PrettyTable = solution_method.calc()
    if table is None:
        print("Уравнение не сходится для выбранного приближения")
        solution_method.draw()
        return
    print(table)
    print(solution_method.output_result())
    solution_method.check_calc()
    solution_method.draw()
