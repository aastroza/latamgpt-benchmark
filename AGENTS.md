# LATAMGPT BENCHMARKING

## Coding guidelines and philosophy

- You should generate code that is simple and redable, avoid unnecesary abstractions and complexity. This is a research codebase so we want to be mantainable and readable.
- Avoid overly defensive coding, no need for a lot of `try, except` patterns, no need for fallbacks or backups - I want the code to fail if something is wrong so that i can fix it.
- Do not add demo-only flags or placeholder CLI options that gate real functionality (e.g., `--run` just to toggle execution); scripts should run their main logic directly.
- Adhere to python 3.12+ conventions
- Write code and comments always in english.