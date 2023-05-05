from prettytable import PrettyTable
from sympy import latex, diff, Symbol, Abs


class SystemEquation:
    """
    Класс для системы уравнений
    """
    def __init__(self, equations: tuple) -> None:
        self.equations = equations
        self.variables = (Symbol('x1'), Symbol('x2'))

    def get_string(self) -> str:
        return f"f1(x1, x2)={latex(self.equations[0])}\n  f2(x1, x2)={latex(self.equations[1])}"


class SimpleIterationMethodForSystem:
    """
    Класс для метода простой итерации для системы уравнений
    """
    def __init__(self,
                 system_equations: SystemEquation,
                 first_approach: tuple,
                 interval_for_x1: tuple,
                 interval_for_x2: tuple,
                 epsilon: float) -> None:
        self._system_equations = system_equations
        self._first_approach = first_approach
        self._interval_for_x1 = interval_for_x1
        self._interval_for_x2 = interval_for_x2
        self._epsilon = epsilon
        self._system_phi_func = self._create_system_phi_func()
        self._field_names_table = ['№ итерации',
                                   'X1(k)', 'X2(k)',
                                   'X1(k+1)', 'X2(k+1)',
                                   '|X1(k+1) - X1(k)|', '|X2(k+1) - X2(k)|']
        self._found_roots = (0.0, 0.0)
        self._vector_errors = (0.0, 0.0)
        self._num_iteration_for_found: int = 0

    def _create_system_phi_func(self) -> tuple:
        return (
            self._system_equations.equations[0] + self._system_equations.variables[0],
            self._system_equations.equations[1] + self._system_equations.variables[1]
        )

    def check_convergence(self) -> bool:
        x_1 = self._system_equations.variables[0]
        x_2 = self._system_equations.variables[1]
        first_equation = Abs(diff(self._system_phi_func[0], x_1)) + Abs(diff(self._system_phi_func[0], x_2))
        second_equation = Abs(diff(self._system_phi_func[1], x_1)) + Abs(diff(self._system_phi_func[1], x_2))
        # проверка всех пограничных значений
        return (
                first_equation.subs({x_1: self._interval_for_x1[0], x_2: self._interval_for_x2[0]}).evalf() < 1 and
                first_equation.subs({x_1: self._interval_for_x1[1], x_2: self._interval_for_x2[0]}).evalf() < 1 and
                first_equation.subs({x_1: self._interval_for_x1[0], x_2: self._interval_for_x2[1]}).evalf() < 1 and
                first_equation.subs({x_1: self._interval_for_x1[1], x_2: self._interval_for_x2[1]}).evalf() < 1 and
                second_equation.subs({x_1: self._interval_for_x1[0], x_2: self._interval_for_x2[0]}).evalf() < 1 and
                second_equation.subs({x_1: self._interval_for_x1[1], x_2: self._interval_for_x2[0]}).evalf() < 1 and
                second_equation.subs({x_1: self._interval_for_x1[0], x_2: self._interval_for_x2[1]}).evalf() < 1 and
                second_equation.subs({x_1: self._interval_for_x1[1], x_2: self._interval_for_x2[1]}).evalf() < 1
        )

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
        pass

    def draw(self) -> None:
        pass

    def output_result(self) -> str:
        return f"Найденные корни: X1={self._found_roots[0]} X2={self._found_roots[1]}\n" + \
               f"Вектор погрешностей: ({self._vector_errors[0]}, {self._vector_errors[1]})" + \
               f"\nЧисло итераций: {self._num_iteration_for_found}"


def main_for_system_equations():
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    # ссылка на Desmos с графиками https://www.desmos.com/calculator/b8cfcxolgp
    system_equation = (
        SystemEquation((
            0.1 * x1 ** 2 + x1 + 0.2 * x2 ** 2 - 0.3,
            0.2 * x1 ** 2 + x2 + 0.1 * x1 * x2 - 0.7
        )),
    )
    if solution_method is None:
        return
    if not solution_method.check():
        return
    table: PrettyTable = solution_method.calc()
    if table is None:
        return
    output(table, solution_method)
    solution_method.draw()

