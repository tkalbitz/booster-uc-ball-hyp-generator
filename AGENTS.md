# CLAUDE.md for booster-uc-ball-hyp-generator

## Requirement Clarification Protocol

### MANDATORY: Ask Questions Before Coding

**NEVER start writing code immediately. Always follow this protocol first:**

1. **Understand the core requirement**: What is the main goal or problem to solve?
2. **Identify key constraints**: What are the technical, business, or performance limitations?
3. **Clarify scope and boundaries**: What is included/excluded from this implementation?
4. **Determine input/output specifications**: What data flows in and out? What formats?
5. **Understand integration points**: How does this connect with existing systems/code?
6. **Identify success criteria**: How will we know the implementation is correct and complete?

### Question Guidelines

- **Ask the most important questions first** - prioritize questions that would fundamentally change the approach
- **Be specific and targeted** - avoid generic questions like "any preferences?"
- **Focus on technical decisions** - architecture, data structures, algorithms, interfaces
- **Clarify ambiguous requirements** - don't make assumptions about unclear specifications
- **Confirm understanding** - restate requirements in your own words for validation

### When to Start Coding

**Only begin implementation when you have:**
- [ ] Clear understanding of the problem and requirements
- [ ] Identified the optimal technical approach
- [ ] Confirmed all major architectural decisions
- [ ] Understood the expected inputs, outputs, and data flow
- [ ] Clarified any ambiguous or missing requirements
- [ ] Formulated a complete plan that will satisfy the request

**If you're unsure about any aspect, ask more questions rather than making assumptions.**

## Python Code Generation Guidelines

### Environment Requirements
- **Python Version**: 3.12+ only (no backward compatibility needed)
- **Maximum line width**: 120 characters
- **Formatter**: Use ruff as primary formatter
- **Quality Checks**: All code must pass pylint, mypy, and ruff checks

### Type Hints & Type System
- **Modern Type Hints**: Use Python 3.12+ syntax only
  - `list[str]` NOT `List[str]`
  - `dict[str, int]` NOT `Dict[str, int]`
  - `str | None` NOT `Optional[str]`
  - `int | float` NOT `Union[int, float]`
- **Type Aliases**: Use `type` keyword for type aliases (NOT `TypeAlias` or `TypeAliasType`)
- **No typing imports**: Do NOT import `set`, `list`, `dict`, `tuple` from `typing` - they are built-in keywords
- **Dataclasses over types**: Use dataclasses instead of defining new types via `type ...` except for simple mappings like `type Unit = str`
- **No Any**: It's forbidden to use `Any` to fix type issues - find proper type solutions
- **Complete type coverage**: Include proper type hints throughout all code

### Code Structure & Style

#### Imports
- **Absolute imports only**: No relative imports allowed
- **Module level only**: Import statements must only be at module level
- **Standard library first**: Order imports: standard library, third-party, local imports

#### Functions & Documentation
- **Docstrings required**: All functions/classes must include docstrings
- **Short docstrings**: For functions with ≤3 parameters, use one-line docstrings
- **Imperative mood**: First line of docstring should be in imperative mood
- **Docstring accuracy**: Function docstring (name, returns, parameter names and types) must exactly match the function signature

#### Walrus Operator Preference
- **Prefer walrus operator `:=`** when line limit allows
- **Single value only**: Unpacking assignment (multiple values) not supported with walrus operator

```python
# Preferred
if result := operation(param):
    # do something

# Instead of
result = operation(param)
if result:
    # do something
```

### Comments Policy - STRICT NO-COMMENT RULE

- **WRITE NO COMMENTS AT ALL** - Code should be self-documenting
- **EXCEPTION**: Only add comments for truly complex algorithms, non-obvious business logic, or external API workarounds
- **NEVER add comments that describe what code does**:
  - No setup/initialization comments ("Configure logging", "Initialize converter")
  - No action comments ("Convert PDF", "Write to file")  
  - No obvious state comments ("Ensure directory exists")
- **If you think a comment is needed to explain what code does, refactor the code instead**
- **Allowed comments** (rare exceptions):
  - Why a particular algorithm was chosen
  - Why a workaround is needed for a specific bug/limitation
  - Why performance optimization was necessary
  - Why a non-standard approach was taken

### Coding Patterns & Best Practices

#### Control Flow
- **Favor early exits** to reduce nesting/indentation
- **Single if statements**: Prefer single `if` statement instead of nested `if` statements
- **List comprehensions**: Prefer list comprehension to create transformed lists instead of manual append() loops

#### Data Structures & Operations
- **Strongly prefer dataclasses over dictionaries**: Use dataclasses for structured data to provide:
  - **Type safety**: All fields are explicitly typed and validated
  - **Field discoverability**: IDE auto-completion shows all available fields  
  - **Runtime validation**: Automatic type checking and validation in `__post_init__`
  - **Immutability options**: Use `frozen=True` for immutable data structures
  ```python
  # Preferred - Dataclass with type safety
  @dataclass
  class Config:
      timeout: int = 60
      batch_size: int = 8
      enabled: bool = True
  
  # Avoid - Dictionary without type safety
  config = {
      "timeout": 60,
      "batch_size": 8, 
      "enabled": True
  }
  ```
- **Tuple operations**: When checking multiple things use tuples for string startswith
  ```python
  # Preferred
  if line_stripped.startswith(("//", "/*")):
  
  # Instead of
  if line_stripped.startswith("//") or line_stripped.startswith("/*"):
  ```
- **Dataclasses over tuples**: Use dataclasses instead of tuples when returning multiple values
  ```python
  # Preferred - Dataclass with named fields
  @dataclass
  class ProcessingResult:
      success: bool
      processed_count: int
      error_message: str | None
  
  # Avoid - Tuple without field names
  result = (True, 5, None)  # Hard to understand what each element represents
  ```
- **Named boolean parameters**: Use named parameters for boolean function arguments
  ```python
  # Preferred - Clear intent
  process_images(enable_caching=True, validate_input=False)
  
  # Avoid - Unclear boolean values
  process_images(True, False)
  ```

#### File System Operations
- **pathlib only**: Use `pathlib` for all file operations and path handling
- **Test for access**: When working with filesystem, test for access/permissions
- **Environment access**: Can use `os` for environment variables

#### Object-Oriented Programming
- **Avoid private member access**: Don't access private members from client classes

### Error Handling & Logging

#### Exception Handling
- **Generic Exception handling**: When catching generic Exception, add `# noqa: BLE001`
- **String variables for exceptions**: NEVER use string literals when raising exceptions
  ```python
  # Correct
  msg: str = "LLM model name is required in configuration"
  raise RuntimeError(msg)
  
  # WRONG
  raise RuntimeError("LLM model name is required in configuration")
  ```
- **Avoid catching own exceptions**: Don't raise exceptions within try blocks that would be caught by the same block

#### Logging
- **Logger naming**: Use `_logger` as logger variable name
- **Exception logging**: Use `_logger.exception("Message")` instead of `_logger.error("Message: %s", ex)`
- **NO f-strings in logging**: Do NOT use f-strings for logging
  ```python
  # Correct
  _logger.exception("Value %s", variable)
  
  # WRONG
  _logger.exception(f"Value {variable}")
  ```
- **Store exception messages**: Store exception messages in variables before using in f-strings

### Regular Expressions

#### Documentation Requirements
- **Comment every regex**: For each regular expression, include a comment explaining what it matches with an example
- **Complex regex handling**: For complex regex (3+ groups or contains "?:" or 3+ bracket groups):
  - Add detailed comment explaining exactly what it matches and why
  - Use `re.VERBOSE` flag for multi-line regex with comments
  - Break down complex regex into smaller, manageable parts
  - Use named capture groups for better understanding

### Async Code Requirements
- **Task references**: Always store references to results of `asyncio.create_task()` and `asyncio.ensure_future()`
- **Await completion**: Ensure all task references are eventually awaited or gathered

### Testing Guidelines
- **Use assert**: In unit tests, use `assert` instead of `assertEquals`, `assertIn`, etc.
- **Test coverage**: Write comprehensive tests for all functionality
- **pytest-qt**: Use pytest-qt for PySide6 GUI testing

## Dependency Management

**Use uv and pyproject.toml for all dependency management. Never use requirements.txt.**

### Installation Commands
- **Install dependencies**: `uv sync`
- **Add dependency**: `uv add package-name`
- **Add dev dependency**: `uv add --dev package-name`
- **Run commands**: `uv run command` (e.g., `uv run ruff check`)
- **Type checking**: `uv run mypy .`
- **Testing**: `uv run pytest`

## Quality Assurance

### Before finishing a requested change
- [ ] All code passes `ruff check`
- [ ] All code passes `mypy` type checking
- [ ] All tests pass with `pytest`
- [ ] All code passes `ruff format` (run after successful ruff, mypy and pytest checks)
- [ ] No comments except for complex algorithms/workarounds
- [ ] All functions have proper docstrings
- [ ] All exception messages stored in variables

### Code Review Checklist
- [ ] Uses Python 3.12+ type hints exclusively
- [ ] No relative imports
- [ ] Proper error handling with string variables
- [ ] pathlib used for all file operations
- [ ] Logging uses format strings, not f-strings
- [ ] Walrus operator used where appropriate
- [ ] List comprehensions used instead of manual loops
- [ ] Early exits reduce nesting
- [ ] Dataclasses used instead of dictionaries for structured data
- [ ] No Any types - all code is fully typed

Follow these guidelines strictly to ensure high-quality, maintainable code that meets all requirements.