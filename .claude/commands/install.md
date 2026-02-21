# /project:install — Set up the Attractor development environment

Run the install script to set up (or reinstall) the Attractor project.

## Arguments

`$ARGUMENTS` — Pass `fresh` or `--fresh` to remove the existing `.venv/` and reinstall from scratch. If empty, perform a normal install (skip venv creation if it already exists).

## Instructions

1. Run the install script:
   ```
   bash .claude/scripts/install.sh $ARGUMENTS
   ```

2. Read the script's output carefully. Report results to the user:
   - Which Python version was found
   - Whether `uv` or `pip` was used
   - Whether the install was fresh or incremental
   - Any verification failures (import check, CLI check, test results)
   - Whether a `.env` file was created or already existed

3. If the script exits with a non-zero status, diagnose the failure:
   - **Python not found**: Suggest installing Python 3.11+ via `brew install python@3.13` or pyenv
   - **Install failed**: Check `pyproject.toml` for syntax issues, or suggest `--fresh` to start clean
   - **Tests failed**: Run `pytest -v` to show detailed failures and help debug

4. After a successful install, remind the user:
   - Activate the venv: `source .venv/bin/activate`
   - Run the agent: `attractor run`
   - Run tests: `pytest -v`
