# Development

## Configure development environment

1. Be sure to have a github account with some SSH key configured.
(or you won't be able to push modifications) and that you configured git:
```
git config --global user.name "Your Name Comes Here"
git config --global user.email you@yourdomain.example.com
```

2. Clone the repo and enter its root folder:
```
git clone git@github.com:xdas-dev/xdas.git
cd xdas
```

3. Checkout to the `dev` branch:
```
git checkout dev
```

4. Create a dedicated conda environment
```
conda create -n xdas-dev
conda activate xdas-dev
conda install pip
```

5. Install xdas and optional dependencies in editable mode:
```
pip install -e ".[dev,docs,tests]"
```

6. Check that all tests are passing (one test is not passing but this is ok):
```
pytest
```

7. Check that the documentation builds without errors:
```
cd docs
make html
cd ..
```
You can then inspect the documentation by opening `index.html` in docs/_build/html with
any browser (see later).

## Package structure

Here a quick overview:
```
. (general configs file are in the root directory)
├── xdas (python source code are here. Note that there is an xdas folder
│   inside the root folder which is usually also name xdas)
├── docs (everything related to documentation is here)
└── tests (tests are here. This folder mirrors the xdas tree structure with prepended
   test_ to all files).
```

## Adding a function

Most of the time, you will add functions (or methods) to implement new features.
Small internal functions can be quite undocumented but ideally, function that will bue
used by users requires to:
- write the body of the function
- write a docstring that explains how to use the function
- add an example into the docstring
- add tests
- include the function to the documentation

We will go through all the details


## How to edit the code

The usual workflow is:
- create a branch to have your independent version of the code
- make modifications by doing as many commits as you want
- format your code
- check that the tests are passing (and that the documentation builds).
- publish your branch online
- make a pull request to ask the permission to merge your code.

To do this you can either use git in the [terminal](https://git-scm.com/docs/gittutorial)
or within [VS Code](https://code.visualstudio.com/docs/sourcecontrol/overview).


## How to write a docstring with examples

You need to add multiline string that follows the
[numpydoc](https://numpydoc.readthedocs.io/) style. Here an example:

```python
def your_function(arg1, arg2=None):
   """
   Explain in one line the main objective.

   Add additional information with with as many comments as you wants. You will need
   to wrap you code (ideally max 88 char per line).

   Parameters
   ----------
   arg1: type (e.g., float)
       Some description.
   arg2: type, optional
       Some description.

   Returns
   -------
   type:
       Some description.

   Examples
   --------
   >>> your_function(1, "value")
   "result"

   """
   return do_something_with_args(arg1, arg2)
```

Note that the outputs of the examples will be used as tests. `pytest` will check that
the hard coded output matches what the code actually outputs. This is a first
quick way to add tests to your function.

## How to format your code

Run (in the root xdas folder or where you are working):
```
black .
isort .
```

## How to add tests

[`pytest`](https://docs.pytest.org) is used for testing.

If you are working on a function in a script in `xdas/dir/file.py` then test must be
located in `tests/dir/test_file.py`. In `pytest` each test is one function. Each function
can check several things with the `assert` python statement. Functions
can be grouped into classes. Bellow an example:

```python
import pytest

class TestMyModule:
   def test_my_function(self): # here self is generally unused
       assert mu_function(0) == 42  # it must be True otherwise the test doesn't pass
       with pytest.raises(ValueError):  # check it raise the correct error
           my_function(-1)
```

Note that here we have one test class for the entire module (the `file.py`) but we could
have one test class per function if those require to test a lot of things. When testing
classes, one testing class per developed class is generally the way to go.

The to run check that your test are passing:
```
pytest
```

Note that VS Code as a nice
[set of features](https://code.visualstudio.com/docs/python/testing) to run test.

## Editing the documentation

The documentation is written in [markdown](https://www.markdownguide.org/basic-syntax/).

It uses the [MyST](https://jupyterbook.org/en/stable/content/myst.html) syntax to run
code as if the documentation was a jupyter notebook.

The simpler si probably to take inspiration from some pre-existing documentation.

If you need some stuff to be run outside some documentation page (e.g. to generate some
data) you can place your code at the end of the `docs/conf.py` file.

To build the documentation go in the docs folder an run:
```
make html
```

You can then inspect the documentation by opening `index.html` in docs/_build/html with
any browser.

Note that VS Code as an extension called Live Preview to render html. Once this extension is installed,
right click on the `index.html` file and click on Show preview.