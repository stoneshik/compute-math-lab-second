import matplotlib
from sympy import init_printing

from equation import main_for_one_equation
from system_equation import main_for_system_equations


def main() -> None:
    while True:
        print("Выберите что решать:")
        print("1. Нелинейное уравнение\n2. Система нелинейных уравнений")
        num_variant = int(input("Введите номер выбранного варианта...\n"))
        if num_variant < 1 or num_variant > 2:
            print("Введен неправильной номер, повторите ввод")
            continue
        break
    if num_variant == 1:
        main_for_one_equation()
    else:
        main_for_system_equations()


if __name__ == '__main__':
    matplotlib.use('TkAgg')
    init_printing()
    main()
