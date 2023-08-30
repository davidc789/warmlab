""" Central workflow management. """

from argparse import ArgumentParser
from typing import Optional


def a(args):
    pass


def b(args):
    pass


def main(argv: Optional[list[str]]):
    parser = ArgumentParser("main")

    subparsers = parser.add_subparsers(help='sub-command help', required=True)

    # create the parser for the "a" command
    parser_a = subparsers.add_parser('a', help='a help')
    parser_a.add_argument('bar', type=int, help='bar help')
    parser_a.set_defaults(func=a)

    # create the parser for the "b" command
    parser_b = subparsers.add_parser('b', help='b help')
    parser_b.add_argument('--baz', choices='XYZ', help='baz help')
    parser_b.set_defaults(func=b)

    # Read the argument.
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    pass
