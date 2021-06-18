import builtins
from itertools import chain
import re
import sys
from types import MappingProxyType, MemberDescriptorType
from typing import (
    Any,
    Callable,
    List,
    Iterable,
    Tuple,
    TypeVar,
    Type,
    IO,
    Dict,
    Mapping,
    Union,
    Optional,
)


class _HAS_DEFAULT_FACTORY_CLASS:
    def __repr__(self) -> str:
        return "<factory>"


class _MISSING_TYPE:
    pass


MISSING = _MISSING_TYPE()
_HAS_DEFAULT_FACTORY = _HAS_DEFAULT_FACTORY_CLASS()
_EMPTY_METADATA: Mapping = MappingProxyType({})

# list will be treated by the parser as List[str] (items are parsed as strings)
_SUPPORTED_TYPES = (
    str,
    int,
    float,
    bool,
    List[str],
    List[int],
    List[float],
    list,
    List,
)
_SUPPORTED_LIST_TYPES = (list, List[str], List[int], List[float], List)
_LIST_CASTS: Dict[Any, type] = {
    list: str,
    List: str,
    List[str]: str,
    List[int]: int,
    List[float]: float,
}

# default prefixes used by the parser to match
# against option strings
_DEFAULT_PREFIXES = ["-"]

# raw strings for building regex
# to match against option strings
_ALLOWED_PREFIXES = {
    "-": r"\-",
    "+": r"\+",
    "/": r"\/",
}

CookedBirria = TypeVar("CookedBirria")
UserClass = TypeVar("UserClass")
OptionalType = Union[_MISSING_TYPE, Type, None]
CookedIngredient = Union[str, int, float, bool, List[int], List[float], List[str]]


# The name of an attribute on the class where we store the Field
# objects.  Also used to check if a class is a cooked class.
_FIELDS = "__arg_fields__"

# The name of an attribute on the class that stores the parameters to
# @dataclass.
_PARAMS = "__arg_params__"


class Ingredient:
    __slots__ = (
        "name",
        "type",
        "short",
        "default",
        "default_factory",
        "help",
    )

    def __init__(
        self,
        default: Any,
        default_factory: Callable[[], Any],
        help: Optional[str],
    ):
        self.name: str = None  # type: ignore[assignment]
        self.type: type = None  # type: ignore[assignment]
        self.default = default
        self.default_factory = default_factory
        self.help = help

    def __repr__(self) -> str:
        return (
            "Ingredient("
            f"name={self.name!r},"
            f"type={self.type!r},"
            f"default={self.default!r},"
            f"default_factory={self.default_factory!r},"
            ")"
        )

    # This is used to support the PEP 487 __set_name__ protocol in the
    # case where we're using a field that contains a descriptor as a
    # default value.  For details on __set_name__, see
    # https://www.python.org/dev/peps/pep-0487/#implementation-details.
    #
    # Note that in _process_class, this Field object is overwritten
    # with the default value, so the end result is a descriptor that
    # had __set_name__ called on it at the right time.
    def __set_name__(self, owner, name: str):
        func = getattr(type(self.default), "__set_name__", None)
        if func:
            # __set_name__ present, call it
            func(self.default, owner, name)


def ingredient(
    *, default=MISSING, default_factory=MISSING, help: str = None
) -> Ingredient:
    if default is not MISSING and default_factory is not MISSING:
        raise ValueError("can't specify both default and default factory")
    return Ingredient(default, default_factory, help)


def _create_fn(
    name: str,
    args: Iterable[str],
    body: Iterable[str],
    *,
    globals: dict = None,
    locals: dict = None,
    return_type: OptionalType = MISSING,
):
    if locals is None:
        locals = {}
    if "BUILTINS" not in locals:
        locals["BUILTINS"] = builtins

    return_annotation = ""
    if return_type is not MISSING:
        locals["_return_type"] = return_type
        return_annotation = "->_return_type"
    args_str = ",".join(args)
    body_str = "\n".join(f"  {b}" for b in body)

    # compute text for the entire function
    txt = f" def {name}({args_str}){return_annotation}:\n{body_str}"

    local_vars = ", ".join(locals.keys())
    txt = f"def __create_fn__({local_vars}):\n{txt}\n return {name}"
    ns: dict = {}
    exec(txt, globals, ns)
    return ns["__create_fn__"](**locals)


def _field_init(f: Ingredient, globals: dict, self_name: str):
    default_name = f"_dflt_{f.name}"
    if f.default_factory is not MISSING:
        globals[default_name] = f.default_factory
        value: Optional[str] = (
            f"{default_name}() "
            f"if {f.name} is _HAS_DEFAULT_FACTORY "
            f"else {f.name}"
        )
    else:
        if f.default is MISSING:
            # no default, simple assignment
            value = f.name
        elif f.default is not MISSING:
            globals[default_name] = f.default
            value = f.name

    return _field_assign(f.name, value, self_name)


def _field_assign(name: Optional[str], value: Optional[str], self_name: str) -> str:
    return f"{self_name}.{name}={value}"


def _init_param(f: Ingredient) -> str:
    # Return __init__ parameter string for this field
    # Ex: equivalent of "x:int=3"

    if f.default is MISSING and f.default_factory is MISSING:
        default = ""
    elif f.default is not MISSING:
        # There's a default, this will be the name that's used to look
        # it up.
        default = f"=_dflt_{f.name}"
    elif f.default_factory is not MISSING:
        # There's a factory function.  Set a marker.
        default = "=_HAS_DEFAULT_FACTORY"
    return f"{f.name}:_type_{f.name}{default}"


def _init_fn(fields: Iterable[Ingredient], self_name: str, globals: dict):
    locals: Dict[str, Any] = {f"_type_{f.name}": f.type for f in fields}
    locals.update({"MISSING": MISSING, "_HAS_DEFAULT_FACTORY": _HAS_DEFAULT_FACTORY})

    body_lines = [_field_init(f, locals, self_name) for f in fields]
    if not body_lines:
        body_lines = ["pass"]

    return _create_fn(
        "__init__",
        [self_name] + [_init_param(f) for f in fields],
        body_lines,
        locals=locals,
        globals=globals,
        return_type=None,
    )


def _repr_fn(fields: Iterable[Ingredient], globals: dict):
    return _create_fn(
        "__repr__",
        ("self",),
        [
            'return self.__class__.__qualname__ + f"('
            + ", ".join([f"{f.name}={{self.{f.name}!r}}" for f in fields])
            + ')"'
        ],
        globals=globals,
    )


def _eq_fn(self_tuple: str, other_tuple: str, globals: dict):
    return _create_fn(
        "__eq__",
        ("self", "other"),
        [
            "if other.__class__ is self.__class__:",
            f" return {self_tuple}=={other_tuple}",
            "return NotImplemented",
        ],
        globals=globals,
    )


def _set_new_attributes(cls, a_name: str, value: Any) -> bool:
    # Don't overwrite existing attribute
    # Return True if attribute already exists
    if a_name in cls.__dict__:
        return True
    setattr(cls, a_name, value)
    return False


def _get_ingredient(cls, a_name: str, a_type: type) -> Ingredient:
    default = getattr(cls, a_name, MISSING)
    f: Ingredient
    if isinstance(default, Ingredient):
        f = default
    else:
        # field in __slots__, so no default value
        if isinstance(default, MemberDescriptorType):
            default = MISSING
        f = ingredient(default=default)

    f.name = a_name
    f.type = a_type

    # disallow mutable defaults for known container types
    if isinstance(f.default, list):
        raise ValueError(
            f"mutable default {type(f.default)} for field "
            f"{f.name} is not allowed: use default_factory() for list"
        )
    if isinstance(f.default, (set, dict)):
        raise ValueError(
            f"mutable default {type(f.default)} for field "
            "dict and set are not supported as default values: use default_factory() for list"
        )
    return f


def _tuple_str(name: str, fields: List[Ingredient]) -> str:
    if not fields:
        return "()"
    # trailing commas for tuple with 1 item
    return f"({','.join([f'{name}.{f.name}' for f in fields])},)"


def _proc_class(cls: Type[UserClass]) -> Type[UserClass]:
    # print(f"Processing class {cls.__name__}")
    fields = {}

    if cls.__module__ in sys.modules:
        globals = sys.modules[cls.__module__].__dict__
    else:
        # Theoretically this can happen if someone writes
        # a custom string to cls.__module__.  In which case
        # such dataclass won't be fully introspectable
        # (w.r.t. typing.get_type_hints) but will still function
        # correctly.
        globals = {}

    base_non_default_fields = []
    base_default_fields = []
    for b in cls.__mro__[-1:0:-1]:
        # only include classes that use our decorator
        base_fields = getattr(b, _FIELDS, None)
        if base_fields:
            # print(f"Base field for {b.__name__}")
            # pprint(base_fields)
            for f in base_fields.values():
                if f.default is MISSING and f.default_factory is MISSING:
                    base_non_default_fields.append(f)
                else:
                    base_default_fields.append(f)

    # Annotations defined in this class and not the base classses.
    # if __annotations__ is not present, then this class adds no
    # new annotations. This is for fields added by this class.

    # Fields are from __annotations__, which is ordered. Defaults
    # are from class attributes.

    cls_annotations = cls.__dict__.get("__annotations__", {})
    cls_fields = []
    for fname, ftype in cls_annotations.items():
        if ftype not in _SUPPORTED_TYPES:
            raise TypeError(f"type {ftype} not supported")
        cls_fields.append(_get_ingredient(cls, fname, ftype))
    # pprint(cls_fields)

    # sort the fields into default and non-default
    non_default_fields = []
    default_fields = []

    for f in cls_fields:
        # if the class attribute exists and is a "Field"
        # replace it with the real default.
        attr = getattr(cls, f.name, MISSING)
        if isinstance(attr, Ingredient):
            if f.default is MISSING:
                # if no default set, delete the class attribute
                delattr(cls, f.name)
                if f.default_factory is MISSING:
                    non_default_fields.append(f)
                else:
                    default_fields.append(f)
            else:
                setattr(cls, f.name, f.default)
                default_fields.append(f)
        elif attr is MISSING:
            non_default_fields.append(f)
        else:
            default_fields.append(f)

    # Dict insertion is ordered, so default fields follow
    # non-default fields
    for f in chain(
        base_non_default_fields, non_default_fields, base_default_fields, default_fields
    ):
        # print(f"Inserting {f.name} in {cls.__name__}")
        fields[f.name] = f

    # raise TypeError for Field members with no annotations
    for name, val in cls.__dict__.items():
        if isinstance(val, Ingredient) and name not in cls_annotations:
            raise TypeError(f"{name!r} is a field but has no type annotations")

    # set all the fields to __arg_fields__
    # and mark this class as a cooked class
    # pprint(fields)
    setattr(cls, _FIELDS, fields)
    flds = list(fields.values())

    # attach __init__
    _set_new_attributes(
        cls,
        "__init__",
        _init_fn(flds, "__cooked_self__" if "self" in fields else "self", globals),
    )

    # attach __repr__
    _set_new_attributes(cls, "__repr__", _repr_fn(flds, globals))

    # attach __eq___
    # we want this for easy debugging and testing
    self_tuple = _tuple_str("self", flds)
    other_tuple = _tuple_str("other", flds)
    _set_new_attributes(cls, "__eq__", _eq_fn(self_tuple, other_tuple, globals=globals))

    return cls


# _cls should not be specified by keyword, hence the underscore,
# it's only used to detect if the decorator is called with params
# or not
def cook(_cls=None):
    def wrap(cls):
        return _proc_class(cls)

    # called as @dataclass()
    if _cls is None:
        return wrap

    # called as @dataclass
    return wrap(_cls)


def ingredients(class_or_instance: object) -> Tuple[Ingredient, ...]:
    try:
        fields = getattr(class_or_instance, _FIELDS)
    except AttributeError:
        raise TypeError("must be called with a birria class type or instance")
    return tuple(fields.values())


def is_cooked_instance(obj: object) -> bool:
    return hasattr(type(obj), _FIELDS)


def is_cooked_class(cls: object) -> bool:
    if not isinstance(cls, type):
        return False

    return hasattr(cls, _FIELDS)


def is_cooked(obj: object) -> bool:
    cls = obj if isinstance(obj, type) else type(obj)
    return hasattr(cls, _FIELDS)


def _print_err_and_exit(msg: str, out: IO = sys.stderr, exit_code=1, cr: str = "\n"):
    print(f"{msg}{cr}", file=out)
    sys.exit(exit_code)


def serve(
    recipe: Type[CookedBirria],
    raw_ingredients: List[str] = None,
    prefixes: List[str] = None,
    extra_prefixes: List[str] = None,
) -> CookedBirria:
    if not is_cooked(recipe):
        raise ValueError(
            "Argument arg_cls must be either a PyArg class or an instance of an PyArg class"
        )

    # use slice to avoid modifying the original list
    if raw_ingredients is None:
        # exclude the first argument (the program invocation)
        preprepped_ingredients = sys.argv[1:]
    else:
        preprepped_ingredients = raw_ingredients[:]

    opt_ingredients: Dict[str, Ingredient] = {}
    # map aliasses to real ingredient names
    # when applicable
    opt_ingredients_aliases: Dict[str, str] = {}
    req_ingredients: List[Ingredient] = []
    cooking: Dict[str, CookedIngredient] = {}

    # check for type annotation as well as real values (best effort)
    # if class attr exists or field type is boolean (flag),
    # it's a named argument

    # boolean is always a named argument (aka a flag)
    # if the default for a boolean is set to a boolean,
    # the behaviour is to reverse that value if the
    # corresponding flag is present in the argument list.
    # In all other cases (no default set, default not set to
    # a boolean), set the value to True if the flag is present.

    # list types are a bit more tricky. First of all
    # multiple list fields are basically impossible to
    # handle if they are not optional (and therefore preceded
    # by an option string). So if there are list arguments,
    # the restrictions are:
    # 1. If a non-default field specifies a list type, only one positional
    # argument is allowed (all items parsed not preceded by an option string
    # will be gathered into a list)
    # in addition to other default fields
    # 2. Following 1), the ordering of argument is now "strict", meaning
    # that positional arguments come first in the argument list, followed
    # by named arguments
    # 3. If a default field specifies a list type, and provides a default
    # list through default_factory, any corresponding parsed items will
    # be added to the default list.
    # 4. Following 3), if the field specifies a default value that's not
    # a list, the default value will be overridden by a list of the parsed
    # items.

    seen_req_list = False
    for f in ingredients(recipe):
        default = getattr(recipe, f.name, MISSING)
        if default is MISSING:
            if f.type == bool:
                # if no default, set the "parsed" value
                # to False here so we can reverse it later
                opt_ingredients[f.name] = f
                cooking[f.name] = False
            elif f.type in _SUPPORTED_LIST_TYPES:
                # list type and default_factory exists -> default field
                if f.default_factory is not MISSING:
                    opt_ingredients[f.name] = f
                else:
                    if seen_req_list:
                        raise TypeError(
                            f"{f.name} cannot be a list,"
                            " only one non-default ingredient can be list"
                        )
                    seen_req_list = True
                    req_ingredients.append(f)
            else:
                if seen_req_list:
                    raise TypeError(
                        "Only one non-default field allowed if using list type,"
                        " use a default field if you want to specify more list fields"
                    )
                req_ingredients.append(f)
        else:
            if f.type == bool:
                # if default is a bool, add it to the parsed
                # dictionary, else set it to False
                if type(default) == bool:
                    cooking[f.name] = default
                else:
                    cooking[f.name] = False
            opt_ingredients[f.name] = f

    num_req = len(req_ingredients)
    # just some sanity checks

    if num_req and not preprepped_ingredients:
        _print_err_and_exit(f"No arguments! Need at least {num_req}")

    if not seen_req_list and len(preprepped_ingredients) < num_req:
        _print_err_and_exit(f"Too few arguments! Need at least {num_req}")

    if num_req > 1 and seen_req_list:
        raise TypeError(
            "Only one non-default field allowed if using list type,"
            "Use a default field if you want to specify more list fields"
        )

    if not prefixes:
        prefixes = _DEFAULT_PREFIXES

    if extra_prefixes:
        prefixes += extra_prefixes

    for p in prefixes:
        if p not in _ALLOWED_PREFIXES:
            raise ValueError(f"'{p}' not supported as a prefix")

    # if the first argument is a named argument,
    # all the postional arguments will follow the last
    # named argument. Likewise, if the first argument
    # is a positional argument, all the named arguments
    # follow the last positional argument.
    # Force this convention to make our life easier

    prefix_group = r"|".join(_ALLOWED_PREFIXES[p] for p in prefixes)
    name_list = []
    for n in opt_ingredients.keys():
        name_list.append(n)
        if "_" in n:
            dash_name = "-".join(n.split("_"))
            name_list.append(dash_name)
            opt_ingredients_aliases[dash_name] = n
    name_group = "|".join(name_list)
    # print(prefixes)
    # print(prefix_rgx_group)
    opt_rgx = re.compile(rf"({prefix_group})({name_group})")
    # loop through the list once first
    # to detect duplicate instances of the same field
    opt_idx = {}

    if opt_ingredients:
        for i, arg in enumerate(preprepped_ingredients):
            match = opt_rgx.match(arg)
            if match is not None:
                name = match.groups()[1]
                if name not in opt_ingredients:
                    name = opt_ingredients_aliases[name]
                if name in opt_idx:
                    _print_err_and_exit(f"Duplicate instances of {name}")
                opt_idx[name] = i

    if not opt_ingredients or not opt_idx:
        # no named arguments (either from the class definition or from the list),
        # just parse the list for positional argument
        if not seen_req_list:
            for i, arg in enumerate(preprepped_ingredients):
                try:
                    ingredient = req_ingredients[i]
                    cooking[ingredient.name] = ingredient.type(arg)
                except IndexError:
                    _print_err_and_exit(f"Too many arguments, only need {num_req}")
                except ValueError:
                    _print_err_and_exit(
                        f"Value {arg} of the wrong type, needs {ingredient.type}"
                    )
            return recipe(**cooking)  # type: ignore[call-arg]
        else:
            vals = []
            ingredient = req_ingredients[0]
            cast = _LIST_CASTS[ingredient.type]
            try:
                for arg in preprepped_ingredients:
                    vals.append(cast(arg))
                if not vals:
                    _print_err_and_exit(
                        f"Need at least one value for {ingredient.name}"
                    )
            except ValueError:
                _print_err_and_exit(
                    f"Value {arg} for {ingredient.name} of the wrong type, needs {cast}"
                )

            return recipe(vals)  # type: ignore[call-arg]

    # if the first named arg appears first in the argument list,
    # all the positional arguments are at the end of the list,
    # otherwise they are at the start of the list. In either cases,
    # we parse the positional arguments first, then the remaining
    # list contains all the named arguments. This avoids cases with
    # list named arguments like:
    # prog -list n1 n2 n3 | p1 p2 p3
    #      named list     | pos args
    # In this case, if we parse the named arguments first, we can't
    # the parser thinks that p1 p2 p3 items in "list". This is not
    # a problem if we parse the positional arguments first.

    opt_first = min(opt_idx.values()) == 0

    last_opt_idx: int = len(preprepped_ingredients)
    if opt_first:
        # non-default field is a list, all named arguments
        # must be at the end of the list
        if seen_req_list:
            _print_err_and_exit(
                "When using list type for non-default field"
                "All default arguments must be specified after the non-default argument"
            )

        # if the first argument is named, named arguments
        # go on until the first positional argument,
        # which is the number of positional arguments from
        # the end of the list
        if num_req:
            last_opt_idx = -num_req

    if num_req:
        pos_args_slice: List[str]
        if not opt_first:
            # if there are named arguments and the first one
            # doesn't start at at the beginning, start from the
            # beginning.
            pos_args_slice = preprepped_ingredients[:num_req]
            # named arguments start from the last positional argument
        else:
            # named arguments start at the beginning, go to
            # back of the list
            pos_args_slice = preprepped_ingredients[-num_req:]
            # named arguments go until the first positional argument

        if not seen_req_list:
            for i, arg in enumerate(pos_args_slice):
                if opt_rgx.match(arg):
                    _print_err_and_exit("Named and positional argument cannot mix")
                try:
                    ingredient = req_ingredients[i]
                    cooking[ingredient.name] = ingredient.type(arg)
                except IndexError:
                    _print_err_and_exit(f"Need positinal argument: {ingredient.name}")
                except ValueError:
                    _print_err_and_exit(
                        f"Value {arg} for {ingredient.name} of the wrong type, needs {ingredient.type}"
                    )
        else:
            vals = []
            ingredient = req_ingredients[0]
            cast = _LIST_CASTS[ingredient.type]
            try:
                for arg in preprepped_ingredients:
                    # gather items into a list until we find an option string
                    if opt_rgx.match(arg):
                        break
                    vals.append(cast(arg))
                if not vals:
                    _print_err_and_exit(
                        f"Need at least one value for {ingredient.name}"
                    )
            except ValueError:
                _print_err_and_exit(
                    f"Value {arg} for {ingredient.name} of the wrong type, needs {cast}"
                )
            cooking[ingredient.name] = vals

    for name, idx in opt_idx.items():
        ingredient = opt_ingredients[name]
        arg_type = ingredient.type
        if arg_type not in _SUPPORTED_LIST_TYPES:
            if arg_type == bool:
                # reverse the previously set value
                cooking[name] = not cooking[name]
            else:
                try:
                    cooking[name] = arg_type(preprepped_ingredients[idx + 1])
                except ValueError:
                    _print_err_and_exit(f"Wrong type for {name}, needs {arg_type}")
        else:
            vals = []
            cast = _LIST_CASTS[arg_type]
            for arg in preprepped_ingredients[idx + 1 : last_opt_idx]:
                if opt_rgx.match(arg):
                    break
                try:
                    vals.append(cast(arg))
                except ValueError:
                    _print_err_and_exit(
                        f"value {arg} for {name} has the wrong type, needs {cast}"
                    )
                if vals:
                    if ingredient.default_factory is not MISSING:
                        def_list = ingredient.default_factory()
                        if isinstance(def_list, list):
                            cooking[name] = def_list + vals
                        else:
                            cooking[name] = vals
                    else:
                        cooking[name] = vals

    return recipe(**cooking)  # type: ignore[call-arg]
