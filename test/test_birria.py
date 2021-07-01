import itertools
from typing import List, Tuple, TypeVar, Type, Callable, Any, Union

import pytest
from pytest import CaptureFixture

from birria import (
    cook,
    ingredient,
    ingredients,
    is_cooked,
    is_cooked_instance,
    is_cooked_class,
    serve,
    _match_opt_strings,
)


S = TypeVar("S", int, str, float)
AllowedList = List[S]
IngredientType = Union[
    int, float, str, bool, list, List[str], List[int], List[float], List
]

CAST_LOOKUP = {
    list: str,
    List: str,
    List[str]: str,
    List[float]: float,
    List[int]: int,
}


def arg_strs_to_list(a1: str, *args: str) -> List[str]:
    raw_list = [a1] if a1 else []
    raw_list += [a for a in args if a]
    all_args = " ".join(raw_list)
    return all_args.split(" ")


def assert_exit(
    exit_code: int, exit_func: Callable[..., Any], *args: Any, **kwargs: Any
):
    with pytest.raises(SystemExit) as wrapped_err:
        exit_func(*args, **kwargs)
    assert wrapped_err.type == SystemExit
    assert wrapped_err.value.code == exit_code


def test_cooking_functions():
    @cook
    class CookedClass:
        a: int
        b: int
        c: str
        d: List[float]
        e: bool = False

    assert is_cooked_class(CookedClass)
    assert is_cooked(CookedClass)
    assert not is_cooked_instance(CookedClass)

    cooked = CookedClass(1, 2, "some string", [0.4, 1.2])
    assert is_cooked_instance(cooked)
    assert is_cooked(cooked)
    assert not is_cooked_class(cooked)

    class_ingredients = ingredients(CookedClass)
    instance_ingredients = ingredients(cooked)
    assert class_ingredients == instance_ingredients


def test_cook_class_empty():
    @cook
    class BaseRecipe:
        a: int
        b: float
        d: bool

    with pytest.raises(TypeError):

        @cook
        class EmptyRecipe:
            pass

    @cook
    class DerivedRecipe(BaseRecipe):
        pass

    assert ingredients(DerivedRecipe) == ingredients(BaseRecipe)


@pytest.mark.parametrize(
    "int_val, float_val, str_val, bool_val, int_list, float_list, str_list",
    [
        (
            1,
            2.0,
            "hello",
            False,
            [1, 2, 3],
            [1.0, 2.0, 3.0, 4.5],
            ["some", "lame", "string"],
        ),
        (
            -199,
            -9.1231,
            "",
            True,
            [i for i in range(100)],
            [-0.5 * i for i in range(20, 50, 3)],
            ["a b c d e f".split()],
        ),
        (
            2 ** 64,
            3.142,
            ",".join((str(i) for i in range(100))),
            True,
            [1, 2, 3],
            None,
            [],
        ),
    ],
)
def test_cook_class_init_no_default(
    int_val: int,
    float_val: float,
    str_val: str,
    bool_val: bool,
    int_list: List[int],
    float_list: List[float],
    str_list: List[str],
):
    @cook
    class CookedClass:
        a: int
        b: float
        c: str
        d: bool
        e: List[int]
        f: List[float]
        g: List[str]

    cooked = CookedClass(
        int_val,
        float_val,
        str_val,
        bool_val,
        int_list,
        float_list,
        str_list,
    )
    assert cooked.a == int_val
    assert cooked.b == float_val
    assert cooked.c == str_val
    assert cooked.d == bool_val
    assert cooked.e == int_list
    assert cooked.f == float_list
    assert cooked.g == str_list


@pytest.mark.parametrize(
    "int_val, float_val, str_val, bool_val, int_list, float_list, str_list",
    [
        (
            1,
            2.0,
            "hello",
            False,
            [1, 2, 3],
            [1.0, 2.0, 3.0, 4.5],
            ["some", "lame", "string"],
        ),
        (
            -199,
            -9.1231,
            "",
            True,
            [i for i in range(100)],
            [-0.5 * i for i in range(20, 50, 3)],
            ["a b c d e f".split()],
        ),
        (
            2 ** 64,
            3.142,
            ",".join((str(i) for i in range(100))),
            True,
            [1, 2, 3],
            None,
            [],
        ),
    ],
)
def test_cook_class_all_default(
    int_val: int,
    float_val: float,
    str_val: str,
    bool_val: bool,
    int_list: List[int],
    float_list: List[float],
    str_list: List[str],
):
    @cook
    class CookedClass:
        a: int = int_val
        b: float = float_val
        c: str = str_val
        d: bool = bool_val
        e: List[int] = ingredient(default_factory=lambda: int_list)
        f: List[float] = ingredient(default_factory=lambda: float_list)
        g: List[str] = ingredient(default_factory=lambda: str_list)

    cooked = CookedClass()
    assert cooked.a == int_val
    assert cooked.b == float_val
    assert cooked.c == str_val
    assert cooked.d == bool_val
    assert cooked.e == int_list
    assert cooked.f == float_list
    assert cooked.g == str_list


@pytest.mark.parametrize(
    "int_val, float_val, str_val, bool_val, int_list, float_list, str_list",
    [
        (
            1,
            2.0,
            "hello",
            False,
            [1, 2, 3],
            [1.0, 2.0, 3.0, 4.5],
            ["some", "lame", "string"],
        ),
        (
            -199,
            -9.1231,
            "",
            True,
            [i for i in range(100)],
            [-0.5 * i for i in range(20, 50, 3)],
            ["a b c d e f".split()],
        ),
        (
            2 ** 64,
            3.142,
            ",".join((str(i) for i in range(100))),
            True,
            [1, 2, 3],
            None,
            [],
        ),
    ],
)
def test_cook_class_init_mixed_ordered(
    int_val: int,
    float_val: float,
    str_val: str,
    bool_val: bool,
    int_list: List[int],
    float_list: List[float],
    str_list: List[str],
):
    @cook
    class CookedClass:
        a: int
        b: float
        c: str
        d: bool = bool_val
        e: List[int] = ingredient(default_factory=lambda: int_list)
        f: List[float] = ingredient(default_factory=lambda: float_list)
        g: List[str] = ingredient(default_factory=lambda: str_list)

    cooked = CookedClass(int_val, float_val, str_val)
    assert cooked.a == int_val
    assert cooked.b == float_val
    assert cooked.c == str_val
    assert cooked.d == bool_val
    assert cooked.e == int_list
    assert cooked.f == float_list
    assert cooked.g == str_list


@pytest.mark.parametrize(
    "int_val, float_val, str_val, bool_val, int_list, float_list, str_list",
    [
        (
            1,
            2.0,
            "hello",
            False,
            [1, 2, 3],
            [1.0, 2.0, 3.0, 4.5],
            ["some", "lame", "string"],
        ),
        (
            -199,
            -9.1231,
            "",
            True,
            [i for i in range(100)],
            [-0.5 * i for i in range(20, 50, 3)],
            ["a b c d e f".split()],
        ),
        (
            2 ** 64,
            3.142,
            ",".join((str(i) for i in range(100))),
            True,
            [1, 2, 3],
            None,
            [],
        ),
    ],
)
def test_cook_class_init_mixed_unordered(
    int_val: int,
    float_val: float,
    str_val: str,
    bool_val: bool,
    int_list: List[int],
    float_list: List[float],
    str_list: List[str],
):
    @cook
    class CookedClass:
        a: int
        b: float = float_val
        c: str
        d: bool
        e: List[int] = ingredient(default_factory=lambda: int_list)
        f: List[float]
        g: List[str]

    # test for proper ordering of positional argument
    # regardless of the order of the ingredient/field
    # declaration, the generated __init__ should
    # have proper argument ordering: positional (non-default)
    # followed by named arguments (default)
    # in this case, the proper order is
    # a, c, d, f, g (positional) then
    # b, e, f (named)

    # test for __init__ with just the postional argument
    cooked = CookedClass(int_val, str_val, bool_val, float_list, str_list)
    assert cooked.a == int_val
    assert cooked.b == float_val
    assert cooked.c == str_val
    assert cooked.d == bool_val
    assert cooked.e == int_list
    assert cooked.f == float_list
    assert cooked.g == str_list


@pytest.mark.parametrize(
    "int_val_tup, float_val_tup, str_val_tup, bool_val_tup, int_list_tup, float_list_tup, str_list_tup",
    [
        (
            (1, -1),
            (2.0, 0.0),
            ("hello", "goodbye"),
            (True, False),
            ([1, 2, 3], [3, 2, 1]),
            ([1.0, 2.0, 3.0, 4.5], [0.0]),
            (["some", "lame", "string"], ["some", "cool", "string"]),
        ),
        (
            (-199, None),
            (-9.1231, None),
            ("", None),
            (True, None),
            ([i for i in range(100)], None),
            ([-0.5 * i for i in range(20, 50, 3)], None),
            (["a b c d e f".split()], None),
        ),
        (
            (2 ** 64, 2 ** 64),
            (3.142, 3.142),
            (
                ",".join((str(i) for i in range(100))),
                ",".join((str(i) for i in range(100))),
            ),
            (True, True),
            ([1, 2, 3], [1, 2, 3]),
            (None, None),
            ([], []),
        ),
    ],
)
def test_cook_class_init_default_override(
    int_val_tup: Tuple[int, int],
    float_val_tup: Tuple[float, float],
    str_val_tup: Tuple[str, str],
    bool_val_tup: Tuple[bool, bool],
    int_list_tup: Tuple[List[int], List[int]],
    float_list_tup: Tuple[List[float], List[float]],
    str_list_tup: Tuple[List[str], List[str]],
):

    def_int, override_int = int_val_tup
    def_float, override_float = float_val_tup
    def_str, override_str = str_val_tup
    def_bool, override_bool = bool_val_tup
    def_int_list, override_int_list = int_list_tup
    def_float_list, override_float_list = float_list_tup
    def_str_list, override_str_list = str_list_tup

    @cook
    class CookedClass:
        a: int = def_int
        b: float = def_float
        c: str = def_str
        d: bool = def_bool
        e: List[int] = ingredient(default_factory=lambda: def_int_list)
        f: List[float] = ingredient(default_factory=lambda: def_float_list)
        g: List[str] = ingredient(default_factory=lambda: def_str_list)

    cooked_boring = CookedClass()
    assert cooked_boring.a == def_int
    assert cooked_boring.b == def_float
    assert cooked_boring.c == def_str
    assert cooked_boring.d == def_bool
    assert cooked_boring.e == def_int_list
    assert cooked_boring.f == def_float_list
    assert cooked_boring.g == def_str_list

    cooked_fun = CookedClass(
        a=override_int,
        b=override_float,
        c=override_str,
        d=override_bool,
        e=override_int_list,
        f=override_float_list,
        g=override_str_list,
    )
    assert cooked_fun.a == override_int
    assert cooked_fun.b == override_float
    assert cooked_fun.c == override_str
    assert cooked_fun.d == override_bool
    assert cooked_fun.e == override_int_list
    assert cooked_fun.f == override_float_list
    assert cooked_fun.g == override_str_list


def test_cook_inheritence_no_override():
    @cook
    class Grandma:
        gran_int: int
        gran_float: float
        gran_str: str

    @cook
    class Ma(Grandma):
        ma_list_int: List[int]
        ma_bool: bool

    @cook
    class Me(Ma):
        me_list_float: List[float]

    grandma = Grandma(-1, 0.0, "grandma")
    assert grandma.gran_int == -1
    assert grandma.gran_float == 0.0
    assert grandma.gran_str == "grandma"

    ma = Ma(1, 0.5, "ma", [1, 2, 3], True)
    assert ma.gran_int == 1
    assert ma.gran_float == 0.5
    assert ma.gran_str == "ma"
    assert ma.ma_list_int == [1, 2, 3]
    assert ma.ma_bool

    me = Me(2, 3.142, "me", [3, 2, 1], False, [0.5, 0.25, 0.125])
    assert me.gran_int == 2
    assert me.gran_float == 3.142
    assert me.gran_str == "me"
    assert me.ma_list_int == [3, 2, 1]
    assert not me.ma_bool
    assert me.me_list_float == [0.5, 0.25, 0.125]

    # test for ingredients of derived class
    # includes the ingredients declared
    # in base classes

    grandma_ingredients = ingredients(grandma)
    ma_ingredients = ingredients(ma)
    me_ingredients = ingredients(me)
    for i in grandma_ingredients:
        assert i in ma_ingredients
    for i in ma_ingredients:
        assert i in me_ingredients


def test_cook_inheritence_override():
    @cook
    class Grandma:
        gran_int: int
        gran_float: float
        gran_str: str = "grandma"

    @cook
    class Ma(Grandma):
        ma_list_int: List[int]
        ma_bool: bool = True
        gran_str: str = "ma"

    @cook
    class Me(Ma):
        ma_bool: bool = False
        me_list_float: List[float]
        gran_str: str = "me"

    # test override parent classes' attributes
    assert Grandma.gran_str == "grandma"
    assert Ma.ma_bool
    assert Ma.gran_str == "ma"
    assert Me.gran_str == "me"
    assert not Me.ma_bool

    # test override on instance's attributes
    grandma = Grandma(-1, 0.0)
    assert grandma.gran_int == -1
    assert grandma.gran_float == 0.0
    assert grandma.gran_str == "grandma"

    ma = Ma(1, 0.5, [1, 2, 3])
    assert ma.gran_int == 1
    assert ma.gran_float == 0.5
    assert ma.gran_str == "ma"
    assert ma.ma_list_int == [1, 2, 3]
    assert ma.ma_bool

    me = Me(2, 3.142, [3, 2, 1], [0.5, 0.25, 0.125])
    assert me.gran_int == 2
    assert me.gran_float == 3.142
    assert me.gran_str == "me"
    assert me.ma_list_int == [3, 2, 1]
    assert not me.ma_bool
    assert me.me_list_float == [0.5, 0.25, 0.125]

    # test if the overrides show up in ingredients()
    grandma_ingredients = {i.name: i for i in ingredients(grandma)}
    ma_ingredients = {i.name: i for i in ingredients(ma)}
    me_ingredients = {i.name: i for i in ingredients(me)}

    assert grandma_ingredients["gran_str"].default != ma_ingredients["gran_str"].default
    assert grandma_ingredients["gran_str"].default != me_ingredients["gran_str"].default
    assert ma_ingredients["gran_str"].default != me_ingredients["gran_str"].default
    assert ma_ingredients["ma_bool"].default != me_ingredients["ma_bool"].default

    for i in grandma_ingredients:
        assert i in ma_ingredients
        assert i in me_ingredients


@pytest.mark.parametrize(
    "raw_ingredients",
    [
        ["1", "2.0", "hello"],
        ["0", "-1231.123", "some lame string"],
        ["0", "-1231.123", "some lame string"],
    ],
)
def test_serve_positional(
    raw_ingredients: List[str],
):
    @cook
    class CookedClass:
        a: int
        b: float
        c: str

    served_birria = serve(CookedClass, raw_ingredients)
    exp_int, exp_float, exp_str = raw_ingredients

    assert served_birria.a == int(exp_int)
    assert served_birria.b == float(exp_float)
    assert served_birria.c == exp_str

    # does parsing still work with extra items
    # probably should just fail for now
    assert_exit(1, serve, CookedClass, raw_ingredients + [1, 2, 3])

    # cases where we fail with exit code 1
    assert_exit(1, serve, CookedClass, [])
    num_args = len(raw_ingredients)
    for i in range(num_args - 1):
        # trim the list one by one forward and backward
        assert_exit(1, serve, CookedClass, raw_ingredients[i + 1 :])
        assert_exit(1, serve, CookedClass, raw_ingredients[: -i - 1])


@pytest.mark.parametrize(
    "scalar_int, scalar_float, scalar_str, bool_flag, list_int, list_float, list_str",
    [
        (
            "-a 1",
            "-b 3.142",
            "-c some",
            "-d",
            "-e 1 2 3",
            "-f 1.0 2.0 3.0",
            "-g some lame string",
        ),
        (
            "-a -100",
            "-b 0.5",
            "-c None",
            "",
            "-e -1 -2 -3 -4 -5",
            "-f 0.0",
            "-g some nice string",
        ),
        (
            "-a -100",
            "-b 0.5",
            "-c -c",
            "",
            "-e -1 -2 -3 -4 -5",
            "-f 0.0",
            "-g some nice string",
        ),
    ],
)
def test_serve_named(
    scalar_int: str,
    scalar_float: str,
    scalar_str: str,
    bool_flag: str,
    list_int: str,
    list_float: str,
    list_str: str,
):
    @cook
    class CookedClass:
        a: int = None
        b: float = None
        c: str = None
        d: bool = None
        e: List[int] = None
        f: List[float] = None
        g: List[str] = None

    cooked_w_no_seasoning = CookedClass()
    assert cooked_w_no_seasoning.a is None
    assert cooked_w_no_seasoning.b is None
    assert cooked_w_no_seasoning.c is None
    assert cooked_w_no_seasoning.d is None
    assert cooked_w_no_seasoning.e is None
    assert cooked_w_no_seasoning.f is None
    assert cooked_w_no_seasoning.g is None

    served_w_no_seasoning = serve(CookedClass, raw_ingredients=[])
    assert served_w_no_seasoning.a is None
    assert served_w_no_seasoning.b is None
    assert served_w_no_seasoning.c is None
    # default behaviour for boolean flags
    # if no default boolean value is provided
    # is to set it to False
    assert served_w_no_seasoning.d is False
    assert served_w_no_seasoning.e is None
    assert served_w_no_seasoning.f is None
    assert served_w_no_seasoning.g is None

    all_args = [
        scalar_int,
        scalar_float,
        scalar_str,
        bool_flag,
        list_int,
        list_float,
        list_str,
    ]

    # test with regular order
    arg_list = arg_strs_to_list(*all_args)

    exp_int = int(scalar_int.split(" ")[1])
    exp_float = float(scalar_float.split(" ")[1])
    exp_str = scalar_str.split(" ")[1]
    exp_bool = bool_flag == "-d"
    exp_int_list = [int(i) for i in list_int.split(" ")[1:]]
    exp_float_list = [float(i) for i in list_float.split(" ")[1:]]
    exp_str_list = [i for i in list_str.split(" ")[1:]]

    served_seasoned = serve(CookedClass, raw_ingredients=arg_list)
    assert served_seasoned.a == exp_int
    assert served_seasoned.b == exp_float
    assert served_seasoned.c == exp_str
    assert served_seasoned.d == exp_bool
    assert served_seasoned.e == exp_int_list
    assert served_seasoned.f == exp_float_list
    assert served_seasoned.g == exp_str_list

    # test with permutations
    for arg_perm in itertools.permutations(all_args):
        arg_list = arg_strs_to_list(*arg_perm)
        served_seasoned_scrambled = serve(CookedClass, raw_ingredients=arg_list)
        assert served_seasoned_scrambled.a == exp_int
        assert served_seasoned_scrambled.b == exp_float
        assert served_seasoned_scrambled.c == exp_str
        assert served_seasoned_scrambled.d == exp_bool
        assert served_seasoned_scrambled.e == exp_int_list
        assert served_seasoned_scrambled.f == exp_float_list
        assert served_seasoned_scrambled.g == exp_str_list


@pytest.mark.parametrize(
    "scalar_int, scalar_float, scalar_str, bool_flag, list_int, list_float, list_str",
    [
        (
            "1",
            "3.142",
            "some",
            "-d",
            "-e 1 2 3",
            "-f 1.0 2.0 3.0",
            "-g some lame string",
        ),
        (
            "-100",
            "0.5",
            "None",
            "",
            "-e -1 -2 -3 -4 -5",
            "-f 0.0",
            "-g some nice string",
        ),
    ],
)
def test_serve_mixed(
    scalar_int: str,
    scalar_float: str,
    scalar_str: str,
    bool_flag: str,
    list_int: str,
    list_float: str,
    list_str: str,
):
    @cook
    class CookedClass:
        a: int
        b: float
        c: str
        d: bool = True
        e: List[int] = None
        f: List[float] = None
        g: List[str] = None

    exp_int = int(scalar_int)
    exp_float = float(scalar_float)
    exp_str = scalar_str
    exp_bool = not (bool_flag == "-d")
    exp_int_list = [int(i) for i in list_int.split(" ")[1:]]
    exp_float_list = [float(i) for i in list_float.split(" ")[1:]]
    exp_str_list = [i for i in list_str.split(" ")[1:]]

    all_pos_args = [scalar_int, scalar_float, scalar_str]
    all_named_args = [bool_flag, list_int, list_float, list_str]

    cooked_w_no_seasoning = CookedClass(exp_int, exp_float, exp_str)
    assert cooked_w_no_seasoning.a == exp_int
    assert cooked_w_no_seasoning.b == exp_float
    assert cooked_w_no_seasoning.c == exp_str
    assert cooked_w_no_seasoning.d
    assert cooked_w_no_seasoning.e is None
    assert cooked_w_no_seasoning.f is None
    assert cooked_w_no_seasoning.g is None

    pos_arg_list = [scalar_int, scalar_float, scalar_str]
    served_w_no_seasoning = serve(CookedClass, raw_ingredients=all_pos_args)
    assert served_w_no_seasoning.a == exp_int
    assert served_w_no_seasoning.b == exp_float
    assert served_w_no_seasoning.c == exp_str
    assert served_w_no_seasoning.d
    assert served_w_no_seasoning.e is None
    assert served_w_no_seasoning.f is None
    assert served_w_no_seasoning.g is None

    # test all permutations for named arguments
    for perm_narg in itertools.permutations(all_named_args):
        narg_list = arg_strs_to_list(*perm_narg)

        # pos followed by narg
        args = pos_arg_list + narg_list
        served_seasoned_scrambled = serve(CookedClass, raw_ingredients=args)
        assert served_seasoned_scrambled.a == exp_int
        assert served_seasoned_scrambled.b == exp_float
        assert served_seasoned_scrambled.c == exp_str
        assert served_seasoned_scrambled.d == exp_bool
        assert served_seasoned_scrambled.e == exp_int_list
        assert served_seasoned_scrambled.f == exp_float_list
        assert served_seasoned_scrambled.g == exp_str_list

        # narg followed by pos
        args = narg_list + pos_arg_list
        served_seasoned_scrambled = serve(CookedClass, raw_ingredients=args)
        assert served_seasoned_scrambled.a == exp_int
        assert served_seasoned_scrambled.b == exp_float
        assert served_seasoned_scrambled.c == exp_str
        assert served_seasoned_scrambled.d == exp_bool
        assert served_seasoned_scrambled.e == exp_int_list
        assert served_seasoned_scrambled.f == exp_float_list
        assert served_seasoned_scrambled.g == exp_str_list


@pytest.mark.parametrize(
    "default, more, itype",
    [
        ([], ["some", "lame", "string"], List[str]),
        ([1, 2, 3], [str(i) for i in range(10)], List[int]),
        (None, ["0.5", "0.25", "0.125"], List[float]),
        (None, ["asdfas", "fdasf", "fcaed"], List[int]),
        (None, ["asdfas", "fdasf", "fcaed"], List[float]),
    ],
)
def test_serve_list_default(default: Any, more: AllowedList, itype: Type[AllowedList]):
    @cook
    class CookedClass:
        shopping_list: itype = ingredient(default_factory=lambda: default)

    served_w_no_seasoning = serve(CookedClass, raw_ingredients=[])
    assert served_w_no_seasoning.shopping_list == default

    assert_exit(1, serve, CookedClass, ["-shopping_list"])

    args = ["-shopping_list"] + more
    cast_inner = CAST_LOOKUP[itype]
    wrong_type = False
    try:
        casted_more = [cast_inner(i) for i in more]
    except ValueError:
        wrong_type = True

    if wrong_type:
        # if a value is of the wrong type (cast failed)
        # we should exit
        assert_exit(1, serve, CookedClass, args)
    else:
        served_seasoned = serve(CookedClass, raw_ingredients=args)

        if isinstance(default, list):
            # if default is provided and it's a list,
            # the parsed items should be added to default
            assert served_seasoned.shopping_list == default + casted_more
        else:
            # otherwise just return the parsed items packed in a list
            assert served_seasoned.shopping_list == casted_more


@pytest.mark.parametrize(
    "args, itype",
    [
        (["1", "2", "3"], List[int]),
        (["some", "lame", "string"], List[str]),
        (["0.5", "0.25", "0.125"], List[float]),
        (["what", "the", "f", "ever"], list),
        (["one"], List),
        (["asdfas", "fdasf", "fcaed"], List[int]),
        (["asdfas", "fdasf", "fcaed"], List[float]),
    ],
)
def test_serve_list_no_default(args: AllowedList, itype: Type[AllowedList]):
    @cook
    class CookedClass:
        shopping_list: itype

    # if a list field is non-default,
    # and no corresponding values are
    # found, exit with code 1
    assert_exit(1, serve, CookedClass, [])

    cast = CAST_LOOKUP[itype]
    wrong_type = False

    try:
        expected = [cast(i) for i in args]
    except ValueError:
        wrong_type = True

    if not wrong_type:
        served = serve(CookedClass, raw_ingredients=args)
        assert served.shopping_list == expected
    else:
        assert_exit(1, serve, CookedClass, args)


def test_serve_list_mixed():
    @cook
    class CookedClass:
        a: List[str]
        b: List[int] = None
        c: List[float] = ingredient(default_factory=list)
        d: List[float] = ingredient(default_factory=lambda: [1.0, 2.0, 3.0])

    exp_a = ["some", "lame", "string"]

    served_w_no_seasoning = serve(CookedClass, raw_ingredients=exp_a)

    assert served_w_no_seasoning.a == exp_a
    assert served_w_no_seasoning.b is None
    assert served_w_no_seasoning.c == []
    assert served_w_no_seasoning.d == [1.0, 2.0, 3.0]

    assert_exit(1, serve, CookedClass, exp_a + ["-b"])
    assert_exit(1, serve, CookedClass, exp_a + ["-c"])
    assert_exit(1, serve, CookedClass, exp_a + ["-d"])

    b_vals = ["-b", "1", "2", "3"]
    exp_b = [int(i) for i in b_vals[1:]]
    c_vals = ["-c", "0.5", "0.25", "0.125"]
    exp_c = [float(i) for i in c_vals[1:]]
    d_vals = ["-d", "-1.0", "-2.0", "-3.0"]
    exp_d = [1.0, 2.0, 3.0] + [float(i) for i in d_vals[1:]]

    all_named = [b_vals, c_vals, d_vals]

    # test all the permutations
    for perm in itertools.permutations(all_named):
        flatten = [i for i in itertools.chain.from_iterable(perm)]
        args = exp_a + flatten
        served_seasoned = serve(CookedClass, raw_ingredients=args)
        assert served_seasoned.a == exp_a
        assert served_seasoned.b == exp_b
        assert served_seasoned.c == exp_c
        assert served_seasoned.d == exp_d

        rev_args = flatten + exp_a
        assert_exit(1, serve, CookedClass, rev_args)


def test_serve_bool():
    @cook
    class CookedClass:
        b: bool

    cooked_w_no_seasoning = serve(CookedClass, raw_ingredients=[])
    assert cooked_w_no_seasoning.b is False

    cooked_seasoned = serve(CookedClass, raw_ingredients=["-b"])
    assert cooked_seasoned.b is True

    @cook
    class TrueCookedClass:
        b: bool = True

    cooked_w_no_seasoning = serve(TrueCookedClass, raw_ingredients=[])
    assert cooked_w_no_seasoning.b is True
    cooked_seasoned = serve(TrueCookedClass, raw_ingredients=["-b"])
    assert cooked_seasoned.b is False

    @cook
    class FalseCookedClass:
        b: bool = False

    cooked_w_no_seasoning = serve(FalseCookedClass, raw_ingredients=[])
    assert cooked_w_no_seasoning.b is False
    cooked_seasoned = serve(FalseCookedClass, raw_ingredients=["-b"])
    assert cooked_seasoned.b is True

    @cook
    class WeirdCookedClass:
        b: bool = None

    cooked_w_no_seasoning = serve(WeirdCookedClass, raw_ingredients=[])
    assert cooked_w_no_seasoning.b is False
    cooked_seasoned = serve(WeirdCookedClass, raw_ingredients=["-b"])
    assert cooked_seasoned.b is True


@pytest.mark.parametrize(
    "prefixes, extra_prefixes", [(["-", "--"], None), (None, ["/"]), (["-"], ["+"])]
)
def test_serve_prefix(prefixes: List[str], extra_prefixes: List[str]):
    @cook
    class CookedClass:
        a: int = None
        b: float = None
        c: str = None
        d: List[int] = None

    # allowed prefixes are '-', '/, '+'
    # the parser should be to able
    # to match option strings with the provided prefixes
    # in any arbitrary combination.
    # Something like "-a 1 /b 0.5 +c string -d .....)"
    # should just work

    ingredient_names = ["a", "b", "c", "d"]
    prefixes = prefixes or ["-"]
    if extra_prefixes:
        prefixes += extra_prefixes
    illegal_prefix = not all(p in ("-", "+", "/") for p in prefixes)

    exp_a = 1
    exp_b = 0.5
    exp_c = "hello"
    exp_d = [1, 2, 3]
    exp_lookup = {"a": exp_a, "b": exp_b, "c": exp_c, "d": exp_d}

    # iterate through:
    # -a -b -c -d
    # -a -b -c --d
    # -a -b --c -d
    # -a --b -c -d
    # -a --b -c -d
    # -a --b -c --d
    # -a --b --c -d
    # -a --b --c --d
    # repeats for --a (and then for b, c, d)
    for prefix_combination in itertools.product(prefixes, repeat=len(ingredient_names)):
        arg_strs = []
        for i, p in enumerate(prefix_combination):
            name = ingredient_names[i]
            opt_str = f"{p}{name}"
            val = exp_lookup[name]
            if isinstance(val, list):
                arg_str = f"{opt_str} {' '.join([str(v) for v in val])}"
            else:
                arg_str = f"{opt_str} {val}"

            arg_strs.append(arg_str)

        for perm in itertools.permutations(arg_strs):
            args = arg_strs_to_list(*perm)
            if illegal_prefix:
                # should raise ValueError here
                with pytest.raises(ValueError):
                    served = serve(
                        CookedClass,
                        raw_ingredients=args,
                        prefixes=prefixes,
                        extra_prefixes=extra_prefixes,
                    )
            else:
                served = serve(
                    CookedClass,
                    raw_ingredients=args,
                    prefixes=prefixes,
                    extra_prefixes=extra_prefixes,
                )
                assert served.a == exp_a
                assert served.b == exp_b
                assert served.c == exp_c
                assert served.d == exp_d


def test_serve_snake_case_names():
    @cook
    class Recipe:
        long_name: str = "hello"
        even_longer_name: float = 0.0
        ridiculously_long_name: List[str] = None

    # for snake case names (underscore between the words)
    # such as "long_name", we should accept both
    # "{prefix}long-name" and "{prefix}long_name" as
    # valid option strings
    exp_a = "lame"
    exp_b = 0.5
    exp_c = ["some", "lame", "string"]
    exp_lookup = {
        "long name": exp_a,
        "even longer name": exp_b,
        "ridiculously long name": exp_c,
    }

    deliminators = ["-", "_"]
    ingredient_names_no_delim = [
        "long name",
        "even longer name",
        "ridiculously long name",
    ]

    # cases like -long-name ... long_name
    # should be treated like duplicate
    # instances of the same argument ->
    # print out error and exit
    assert_exit(1, serve, Recipe, ["-long-name", "long", "-long_name", "name"])

    for delim_combo in itertools.product(
        deliminators, repeat=len(ingredient_names_no_delim)
    ):
        arg_strs = []
        for i, delim in enumerate(delim_combo):
            name = ingredient_names_no_delim[i]
            opt_str = f"-{delim.join(name.split())}"
            val = exp_lookup[name]
            if isinstance(val, list):
                arg_str = f"{opt_str} {' '.join([str(v) for v in val])}"
            else:
                arg_str = f"{opt_str} {val}"

            arg_strs.append(arg_str)

        for perm in itertools.permutations(arg_strs):
            args = arg_strs_to_list(*perm)
            served = serve(
                Recipe,
                raw_ingredients=args,
            )
            assert served.long_name == exp_a
            assert served.even_longer_name == exp_b
            assert served.ridiculously_long_name == exp_c


@pytest.mark.parametrize("prefixes", [["-", "+", "/"]])
def test_help(prefixes: List[str], capsys: CaptureFixture[str]):

    # test that "help" and "h" are reserved
    @cook
    class Recipe:
        help: str

    with pytest.raises(TypeError):
        _ = serve(Recipe)

    @cook
    class Recipe:
        h: str

    with pytest.raises(TypeError):
        _ = serve(Recipe)

    @cook
    class Recipe:
        help: str = "help"

    with pytest.raises(TypeError):
        _ = serve(Recipe)

    @cook
    class Recipe:
        h: str = None

    with pytest.raises(TypeError):
        _ = serve(Recipe)

    # test that we print out something to stderr and exit 0
    # when the help option string is the only argument on the cli,
    # don't really need to test for the exact help string
    # as that's subject to change
    @cook
    class Recipe:
        a: int
        b: float
        c: str
        d: bool
        f: List[str] = None

    help_opt_strs = ["".join(s) for s in itertools.product(prefixes, ("h", "help"))]

    for s in help_opt_strs:
        assert_exit(0, serve, Recipe, [s], prefixes)
        captured = capsys.readouterr().err
        assert captured
        assert_exit(0, serve, Recipe, [s], prefixes, None, "test description")
        captured_w_description = capsys.readouterr().err
        assert captured_w_description
        assert captured_w_description != captured

    # test that parsing behaves normally if an item is the same
    # as the help option string. The parser should just treat
    # that as any other item in the list.

    exp_a = 1
    exp_b = 2.0
    exp_c = "hello"

    for s in help_opt_strs:
        args = [str(exp_a), str(exp_b), s]
        served = serve(Recipe, raw_ingredients=args)
        assert served.a == exp_a
        assert served.b == exp_b
        assert served.c == s
        args = [str(exp_a), str(exp_b), exp_c, "-f", s, "more", "strings", "-d"]
        served = serve(Recipe, raw_ingredients=args)
        assert served.a == exp_a
        assert served.b == exp_b
        assert served.c == exp_c
        assert served.d
        assert served.f == [s, "more", "strings"]


@pytest.mark.parametrize(
    "prefixes, names, expected, raw_names",
    [
        (["-", "+", "/"], ["b"], [(1, "b")], ["blah", "b", "+bloh", "bbb", "+-b"]),
        (
            ["-"],
            ["blah", "b"],
            [(0, "blah"), (3, "b")],
            ["blah", "+b", "-b", "b", "+-b", "bob"],
        ),
    ],
)
def test_match_opt_string_all_permumations(
    prefixes: List[str],
    names: List[str],
    expected: List[Tuple[int, str]],
    raw_names: List[str],
):
    # test for permutations of option strings (prefixes * opt names)
    # prefixes -> list of possible prefix
    # names -> names to match against
    # expected -> list of expected index, name tuples that the function yields
    # raw_names -> list of strings, including the names to match against,
    #               iterate through all the permuations of prefixes ordering
    #               and combine them with the strings to generate a list
    #               of argument to be passed to _match_opt_strings().
    #               assert that at each iteration, the yielded tuple
    #               is the expected one.

    for prefix_combo in itertools.product(prefixes, repeat=len(raw_names)):
        arg_strs = []
        for i, p in enumerate(prefix_combo):
            name = raw_names[i]
            opt_str = f"{p}{name}"
            arg_strs.append(opt_str)

        print(arg_strs)
        idx_name_tuples = [
            (i, name) for i, name in _match_opt_strings(names, prefixes, arg_strs)
        ]
        assert idx_name_tuples == expected


@pytest.mark.parametrize(
    "prefixes, names, expected, args_in",
    [
        (
            ["-"],
            ["b", "bleh"],
            [(1, "b"), (4, "bleh")],
            ["b", "-b", "+b", "/b", "-bleh", "bleh", "+bleh", "/bleh"],
        ),
        (
            ["+"],
            ["b", "bleh"],
            [(2, "b"), (6, "bleh")],
            ["b", "-b", "+b", "/b", "-bleh", "bleh", "+bleh", "/bleh"],
        ),
        (
            ["/"],
            ["b", "bleh"],
            [(3, "b"), (7, "bleh")],
            ["b", "-b", "+b", "/b", "-bleh", "bleh", "+bleh", "/bleh"],
        ),
        (
            ["-", "+", "/"],
            ["b", "bleh"],
            [(1, "b"), (2, "b"), (3, "b"), (7, "bleh"), (9, "bleh"), (10, "bleh")],
            [
                "b",
                "-b",
                "+b",
                "/b",
                "-bbb",
                "+bbb",
                "/bbb",
                "-bleh",
                "bleh",
                "+bleh",
                "/bleh",
                "-blehelo",
                "+blehelo",
                "/blehelo",
            ],
        ),
    ],
)
def test_match_opt_strings(
    prefixes: List[str],
    names: List[str],
    expected: List[Tuple[int, str]],
    args_in: List[str],
):
    idx_name_tuples = [t for t in _match_opt_strings(names, prefixes, args_in)]
    assert idx_name_tuples == expected


@pytest.mark.parametrize(
    "prefix, alias_a, alias_b, alias_c, alias_d, exp_a, exp_b, exp_c, exp_d",
    [
        ("-", "all", "bleh", "cringe", "dude", 10, 3.142, "crack", True),
        ("+", "all", "bleh", "cringe", "dude", 10, 3.142, "crack", True),
        ("/", "all", "bleh", "cringe", "dude", 10, 3.142, "crack", True),
    ],
)
def test_alias(
    prefix: str,
    alias_a: str,
    alias_b: str,
    alias_c: str,
    alias_d: str,
    exp_a: int,
    exp_b: float,
    exp_c: str,
    exp_d: bool,
):
    @cook
    class Recipe:
        a: int = ingredient(alias=alias_a, default=None)
        b: float = ingredient(alias=alias_b, default=None)
        c: str = ingredient(alias=alias_c, default=None)
        d: bool = ingredient(alias=alias_d)

    normal_args = [
        f"{prefix}a {exp_a}",
        f"{prefix}b {exp_b}",
        f"{prefix}c {exp_c}",
        f"{prefix}d",
    ]
    alias_args = [
        f"{prefix}{alias_a} {exp_a}",
        f"{prefix}{alias_b} {exp_b}",
        f"{prefix}{alias_c} {exp_c}",
        f"{prefix}{alias_d}",
    ]

    served_norm = serve(
        Recipe, raw_ingredients=" ".join(normal_args).split(), prefixes=[prefix]
    )
    assert served_norm.a == exp_a
    assert served_norm.b == exp_b
    assert served_norm.c == exp_c
    assert served_norm.d == exp_d

    for perm in itertools.permutations(alias_args):
        served_alias = serve(
            Recipe, raw_ingredients=" ".join(perm).split(), prefixes=[prefix]
        )
        assert served_alias.a == exp_a
        assert served_alias.b == exp_b
        assert served_alias.c == exp_c
        assert served_alias.d == exp_d
        assert served_norm == served_alias

    # if alias is the same name as the ingredient name
    # should rase ValueError on serve()
    @cook
    class WrongRecipe:
        hello: str = ingredient(alias="hello", default=None)

    with pytest.raises(ValueError):
        serve(WrongRecipe, raw_ingredients=["world"])
