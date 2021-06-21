# Birria: simple cli argument parsing

Declare your "Recipe" with type annotations like `dataclass`, then "serve"
the arguments. For more information about type annotations:

- [PEP484](https://www.python.org/dev/peps/pep-0484/)
- [PEP544](https://www.python.org/dev/peps/pep-0544/)

## Quickstart

Install using pip:
```bash
pip install birria
```

In your source code.

```python
from birria import cook, serve

@cook
class Recipe:
    first: int
    second: float
    third: str
    fourth: str = None
    fifth: List[int] = None
    sixth: bool

dish = serve(Recipe)

print(f"First: {dish.first}")
print(f"Second: {dish.second}")
print(f"Third: {dish.third}")
print(f"Fourth: {dish.fourth}")
print(f"Fifth: {dish.fifth}")
print(f"Sixth: {dish.sixth})
```

Then, from the shell:

```bash
> python myprog.py 1 2.0 three -fourth four -fifth 1 2 3 4 5 -sixth
> First: 1
> Second: 2.0
> Third: three
> Fourth: four
> Fifth: [1, 2, 3, 4, 5]
> Sixth: True
```

## Why?

Because i was making some birria simultaneously, and I'm a hunger-driven creature. 

## Why for real?
Take the example from quickstart, to do that with `argparse`,
```python
import argparse

parser = ArgumentParser()
parser.add_argument("first", type=int)
parser.add_argument("second", type=float)
parser.add_argument("third", type=str)
parser.add_argument("-fourth", type=str, default=None)
parser.add_argument("-fifth", type=int, nargs="*", default=None)
parser.add_argument("-sixth", action="store_true")

args = parser.parse_args()

# print out the parsed values
...
```
Why is this worse? Well it's not really worse, but it's  more verbose,
and one of the main motivations for `birria` was to simplify things. Admittedly,
that means less power and complicated features than those supported by `argparse`,
(all those fancy actions like "store_true", "append_const", etc.), but IMO, most
of the time, those things are overkill, and simplicity is generally better.

In addition, another example: say you want to reuse a group of arguments across multiple
programs, with `argparse`, this is what you generally do:

```python
parent_parser = argparse.ArgumentParser(add_help=False)
# add your common arguments down here
parent_parser.add_argument("-base", action="store_true")

...

app_parser = argparse.ArgumentParser(parents=[parent_parser])
# again, add your arguments
...

another_parser = argparse.ArgumentParser(parents=[parent_parser])
# again, add your arguments
...
```


With `birria`, you can just write a base recipe and extend it

```python
@cook
class BaseRecipe:
    # common arguments here
    base: bool
    ...

@cook
class FancyRecipe(BaseRecipe):
    # specific arguments here
    # you can also override
    # the arguments in the base
    # class

    # this overrides the default
    # value of base (False)
    # base: bool = True
    extra: List[int]
```


## More use cases

Here are some more common use cases (more will probably will be added).

### Declare required and optional arguments in arbitrary order

```python
@cook
class Recipe:
    first_opt: int = 0
    first_req: str
    second_req: float
    second_opt: str = "some lame string"
    third_opt: List[float] = None
    fourth_opt: bool
```
Note that a bool field/ingredient is always optional (like a flag).


### Reverse bool flag
```python
@cook
class Recipe:
    b1: bool
    b2: bool = True

# print out parsed arguments
```
```bash
> python myprog.py -b1 -b2 
> b1: True, b2: False
```

This is useful to mimic the "store_true" and "store_false" behaviours from `argparse`. The
parser essentially reverses the default boolean value of the field it finds a corresponding
flag. If the default value is anything else than a boolean,
when the corresponding flag is found, the parsed value is `True`.

### Lists
```python
from birria import cook, ingredient, serve

@cook
class Recipe:
    l1: List[float] = None
    l2: List[int] = ingredient(default_factory: lambda: [1, 2, 3])

# print out parsed arguments
```
```bash
> python myprog.py -l1 0.5 0.121 3.142 -l2 4 5 6
> l1: [0.5, 0.121, 3.142]
> l2: [1, 2, 3, 4, 5, 6]
```
Note that you have to call `ingredient()` with the `default_factory` parameter to specify
a list as the default value. This is because Python stores default member attributes
as class attributes, so multiple instances of the same class will share the same variable,
like in this example:

```python
class A:
    def_list = []
    def append(self, elem):
        self.def_list.append(elem)

a1 = A()
a2 = A()
a2.append(1)
a2.append(2)

assert a1.def_list == [1, 2]        # true
assert a2.def_list == [1, 2]        # also true
assert a1.def_list is a2.def_list   # big true true
```

This is not desirable for our use case as we want each instances to have its own list
attributes. Using a factory to initialize a default list solves this problem. This is
basically what `dataclass` does.

Another thing to note is this appending behaviour is only applied if the field/ingredient
is annotated as a list type (see limitations for more details), AND the default
value is a list. If the default value is not a list, the resulting value will be a
new list containing the parsed corrsponding items.

You can have as many list type ingredients as you want if you declare them as optional.
If a non-optional ingredient is annotated as a list, then only one non optional ingredient
is allowed, and the ordering of the arguments on the cli becomes "strict": items for the
non-optional ingredient must come first, then followed by any optional items. Otherwise,
the parser can't tell which item belongs to which ingredient.

```python
@cook
class Recipe:
    req_list: List[int]
    flag: bool
    opt_list: List[str]
```
```bash
> python myprog.py 1 2 3 -flag -opt-list some more strings         # good
> python myprog.py -flag -opt-list some more strings 1 2 3         # bad
```

### Long names
For longer ingredient names, snake-casing is recommended, as the parser accepts
both the snake-cased names and dash-separated names

```bash
python myprog.py -some-very-long-name-for-a-flag    # good
python myprog.py -some_very_long_name_for_a_flag    # also good  
```

## Limitations
Here are current limitations of `birria`. Some of them are by design and therefore
will likely never be "fixed".

### Optional and non-optional argument mixing
By design, `birria` doesn't allow mixed ordering of optional and non-optional
arguments on the cli. However, ordering of the optional argument can be arbitrary.


```bash
> python myprog.py 1 2 3 -p 1 -b -c some lame string    # good
> python -b -p 1 -c some lame string 1 2 3              # good
> python myprog.py -b 1 2 3 -p 1 -c some lame string    # bad
```

### Supported types
`birria` currently only supports these types:

`int, float, str, bool, list, List, List[str], List[int], List[float]`

Note that `list` and `List` will be treated as `List[str]`.

More scalar types like `byte` will likely be added

### Prefixes

By default, `birria` only accepts `-` as the prefix for option strings, other supported
characters are: `['+', '/']` . More will likely be added in the future.

You can specify what option string to accept through the parameters `prefixes` and `extra_prefixes`
of `serve`

```python
dish = serve(Recipe, prefixes=['+'], extra_prefixes=['/'])
```

The resulting prefix list will be all the characters in `prefixes` and `extra_prefixes`.

## Details

### cook decorator

The `cook` decorator is heavily inspired by `dataclass` (I basically stole the code), but it
is a very simplified and limited version of a dataclass, and not meant to be used in the same way.
It's also written to be easy way to declare required and optional arguments in arbitrary
order, even with inheritence, so its behaviour in that aspect is not at all like a dataclass.
Consider this example.

```python
@cook
class BaseRecipe:
    req_int: int
    opt_int: int = None


@cook
class FancyRecipe:
    req_float: float
    opt_list: list = None

    # this is the generated __init__
    def __init__(self, req_int: int, req_float: float, opt_int: int = None, opt_list: list = None):
        ...

    # __repr__ and __eq__ are also generated for
    # easy printing and debugging

# FancyRecipe will have
# req_int:      int =>      no default value
# req_float:    float =>    no default value
# opt_int:      int =>      defaults to None
# opt_list:     list =>     defaults to None 
```

Using `dataclass`, this would fail, for very good reasons that I won't get into here. However,
because `cook` is only meant to turn a class into a list of declarations, and the resulting
class instances are only meant to be used as dumb containers, this behaviour is fine. Behind
the scenes, just like `dataclass`, `cook` looks through all the class's base classes in
reverse MRO, and remembers the fields of those decorated by `cook`. Finally, it looks
through the fields of the current class, and sorts all the fields into "optional" and
"required" buckets. Then `__init__` is generated with "correct" argument ordering. In
addition, fields with the same names from the base classes will still be overridden by
the derived class.

### Parser behaviour

The parser is basically just the function `serve`. The first step is to validate the "recipe".

```python
@cook
class GoodRecipe1:
    a: int
    b: float
    c: str
    d: List[int] = None

@cook
class BadRecipe1:
    a: int
    b: SomeUnsupportType    # will raise TypeError


# if a required ingredient is a list,
# only one required ingredient is allowed
# so this is bad
@cook
class BadRecipe2:
    a: List[int]
    b: str                  

# but this is good
@cook
class GoodRecipe2:
    a: List[int]
    b: List[int] = None
    c: str = None
```

The next phase is the actual parsing. Values from the cli is passed to the parser through
`sys.argv` as a list. Regardless of whether there are optional ingredients in the recipe,
the parser parses the required items first. Because intermixing of optional and required
items are not allowed, in all cases except when the required ingredient is a list, the
parser can always figure out where the required values are. The "algorithm" for parsing is:

1. If the recipe doesn't define any optional ingredient, parse the whole argument list for
required items.
    - Likewise, if there are no required ingredient, parse the whole list for optional items.
2. If there are optional ingredients, build regex to scan where option strings are in the list.
    - If the first option string is the first item, required items are at the end of the list.
    - Otherwise, the required items start at the beginning of the list.
    - If no option string is found in the list, just parse the whole list for required items.
3. Parse the required items.
4. Parse the remaining list for optional items.


```bash
# from GoodRecipe1
# 3 required scalar items
# 1 optional list items

# no option string found, parse whole list for required items
> python myprog.py 1 2.0 three

# option string first item, parse the last 3 items of the list
# for required items, then the remaning list
> python myprog.py -d 0 1 2 3 4 5 1 2.0 three

# option string not first item, parse the first 3 items of the list
# for required items, then the remaining list 
> python myprog.py 1 2.0 three -d 0 1 2 3 4 5

# d = [0, 1, 2, 3, 4, 5]
# a = 1
# b = 2.0
# c = "three"

# from GoodRecipe2
# arbitrary order of optional values
> python 1 2 3 -c hello -b 0 -1 -2
# or
> python 1 2 3 -b 0 -1 -2 -c hello
```

In the case where there's a required ingredient that's a list, the parser assumes that
the first items are items belonging to that required ingredient. Items will be gathered
into that list until an option string is matched. So if you're using a list as a required
ingredient, make sure that values for it come before all the optional items.

```python
@cook
class Recipe:
    a: List[str]
    b: bool
    c: List[int] = None
```

```bash

# gather items into list for a until "-b" or "-c",
# whichever comes first
> python myprog.py some lame string -b -c 1 2 3
# a = ["some", "lame", "string]
# b = True
# c = [1, 2, 3]

# sees -b and stops immediately,
# so empty list for a, but a is required
# so prints out error and exits
> python myprog.py -b -c 1 2 3 some lame string
```