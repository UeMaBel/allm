"""A tiny calculator: ordinary Python math, ready for allmengine to scramble."""
import argparse


def number(text):
    try:
        return int(text)
    except ValueError:
        return float(text)


def calculate(left, operator, right):
    match operator:
        case "+":
            return left + right
        case "-":
            return left - right
        case "*":
            return left * right
        case "/":
            return left / right
        case _:
            raise ValueError(f"Unknown operator: {operator}")


def main():
    parser = argparse.ArgumentParser(
        description="Tiny calculator. With allmengine, even addition is an adventure.",
        epilog='Example: python calculator.py 2 + 3',
    )
    parser.add_argument("left", type=number)
    parser.add_argument("operator", choices=["+", "-", "*", "/"])
    parser.add_argument("right", type=number)
    args = parser.parse_args()
    try:
        result = calculate(args.left, args.operator, args.right)
    except (ZeroDivisionError, OverflowError, ValueError) as error:
        parser.exit(1, f"Cannot calculate: {error}\n")
    print(f"{args.left} {args.operator} {args.right} = {result}")


if __name__ == "__main__":
    main()
