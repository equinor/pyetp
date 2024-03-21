# Contributing

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

You can contribute in many ways:

## Types of Contributions

### Report Bugs

Report bugs at https://github.com/equinor/pyetp/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* pyetp version
* Detailed steps to reproduce the bug.

### Fix Bugs

Look through the Git issues for bugs. Anything tagged with "bug"
and "help wanted" is open to whoever wants to implement it.

### Implement Features

Look through the Git issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

### Write Documentation

We could always use more documentation, whether as part of the
official docs, in docstrings, or even on the web in blog posts,
articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue
at https://github.com/equinor/pyetp/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.

## Get Started!

Ready to contribute? Here's how to set up ``pyetp`` for local development.

1. Fork the ``pyetp`` repo on Github equinor to your personal user
2. Clone your fork locally
3. Start the development container. [Info](https://containers.dev/)
4. Create a branch for local development:
5. Make your changes locally.

6. When you're done making changes, check that your changes pass flake8 and the tests
```
poetry run flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=docs
poetry run pytest --cov-report=term-missing --cov=pyetp tests/ | tee pytest-coverage.txt
```

7. Commit your changes and push your branch
8. Submit a pull request.


## Writing commit messages

Commit messages should be clear and follow a few basic rules. Example:

```
    ENH: add functionality X to numpy.<submodule>.
```
The first line of the commit message starts with a capitalized acronym
(options listed below) indicating what type of commit this is.  Then a blank
line, then more text if needed.  Lines shouldn't be longer than 72
characters.  If the commit is related to a ticket, indicate that with
``"See #3456", "Cf. #3344, "See ticket 3456", "Closes #3456"`` or similar.

Read Chris Beams hints on commit messages <https://chris.beams.io/posts/git-commit/>.

Describing the motivation for a change, the nature of a bug for bug fixes or
some details on what an enhancement does are also good to include in a commit message.
Messages should be understandable without looking at the code changes.
A commit message like FIX: fix another one is an example of what not to do;
the reader has to go look for context elsewhere.

Standard acronyms to start the commit message with are:

```
    API: an (incompatible) API change (will be rare)
    PERF: performance or bench-marking
    BLD: change related to building
    BUG: bug fix
    FIX: fixes wrt to technical issues, e.g. wrong requirements.txt
    DEP: deprecate something, or remove a deprecated object
    DOC: documentation, addition, updates
    ENH: enhancement, new functionality
    CLN: code cleanup, maintenance commit (refactoring, typos, PEP, etc.)
    REV: revert an earlier commit
    TST: addition or modification of tests
    REL: related to release
```

## Type hints

[PEP 484](https://peps.python.org/pep-0484/)

## Docstrings

[Numpy](https://numpydoc.readthedocs.io/en/latest/format.html)

## Style guidelines

[PEP 8](https://peps.python.org/pep-0008/)

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docstrings and/or docs should be updated.


