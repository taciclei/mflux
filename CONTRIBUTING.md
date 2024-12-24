# Contributing to MFLUX

First off, thank you for considering contributing to MFLUX! It's people like you that make MFLUX such a great tool.

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* Use a clear and descriptive title
* Describe the exact steps which reproduce the problem
* Provide specific examples to demonstrate the steps
* Describe the behavior you observed after following the steps
* Explain which behavior you expected to see instead and why
* Include details about your configuration and environment

### Suggesting Enhancements

If you have a suggestion for the project, we'd love to hear about it! Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* A clear and descriptive title
* A detailed description of the proposed feature
* An explanation of why this enhancement would be useful
* Possible implementation details

### Pull Requests

* Fill in the required template
* Do not include issue numbers in the PR title
* Include screenshots and animated GIFs in your pull request whenever possible
* Follow the Python style guide
* Include thoughtfully-worded, well-structured tests
* Document new code based on the Documentation Styleguide
* End all files with a newline

## Development Process

1. Fork the repo and create your branch from `main`
2. If you've added code that should be tested, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes
5. Make sure your code lints
6. Issue that pull request!

### Development Setup

```bash
# Clone your fork
git clone https://github.com/<your-username>/mflux.git

# Add upstream remote
git remote add upstream https://github.com/original/mflux.git

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
make install-dev
```

### Code Style

We use `ruff` for Python code formatting. Please ensure your code follows our style guidelines:

```bash
# Format code
make format

# Run linter
make lint
```

### Running Tests

```bash
# Run all tests
make test

# Run specific test
python -m pytest tests/test_specific.py
```

## Documentation

* Use docstrings for all public modules, functions, classes, and methods
* Follow Google style for docstrings
* Keep documentation up to date with code changes

### Example Docstring

```python
def function_name(param1: type, param2: type) -> return_type:
    """Short description of function.

    Longer description of function if needed.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ErrorType: Description of when this error is raised
    """
    pass
```

## Git Commit Messages

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
* feat: A new feature
* fix: A bug fix
* docs: Documentation only changes
* style: Changes that do not affect the meaning of the code
* refactor: A code change that neither fixes a bug nor adds a feature
* perf: A code change that improves performance
* test: Adding missing tests
* chore: Changes to the build process or auxiliary tools

## Questions?

Feel free to contact the project maintainers if you have any questions or need clarification on anything.

Thank you for contributing to MFLUX! 🚀
