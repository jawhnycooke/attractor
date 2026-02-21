# /project:dev — Manage development background services

Manage dev watchers (test runner, linter) for the Attractor project.

## Arguments

`$ARGUMENTS` — One of:
- `start [tests|lint|all]` — Start watcher(s). Default: `all`
- `stop [tests|lint|all]` — Stop watcher(s). Default: `all`
- `status` — Show environment info, running services, API key status
- *(empty)* — Default to `status`

## Instructions

1. Determine the action from `$ARGUMENTS`. If empty, use `status`.

2. Run the dev script with the appropriate arguments:
   ```
   bash .claude/scripts/dev.sh $ARGUMENTS
   ```
   If `$ARGUMENTS` is empty, run:
   ```
   bash .claude/scripts/dev.sh status
   ```

3. Report the output to the user in a clear summary:

   **For `status`**: Show a table or formatted list of:
   - Python version and venv path
   - Package installer (uv or pip)
   - API key status (set / not set — never print values)
   - Running watchers with PIDs
   - Test count

   **For `start`**: Confirm which services started and their PIDs. If a service is already running, note that.

   **For `stop`**: Confirm which services were stopped. If none were running, note that.

4. If the venv is not activated or doesn't exist, suggest running `/project:install` first.

5. If `start tests` fails because `pytest-watcher` isn't installed, the script will attempt to install it automatically. Report if that install fails.
