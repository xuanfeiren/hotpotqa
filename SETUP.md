# HotpotQA Development Environment Setup

## Virtual Environment

This project uses `uv` for Python package management with a virtual environment located at `.venv/`.

### Activating the Virtual Environment

```bash
source .venv/bin/activate
```

## Installed Editable Repositories

The following repositories are installed as editable packages, meaning changes to the source code will be immediately reflected:

1. **Trace** (`trace-opt==0.1.3.9`)
   - Location: `./Trace`
   - Repository: https://github.com/xuanfeiren/Trace

2. **OpenEvolve** (`openevolve==0.2.23`)
   - Location: `./openevolve`
   - Repository: https://github.com/xuanfeiren/openevolve

3. **GEPA** (`gepa==0.0.22`)
   - Location: `./gepa-repo`
   - Repository: https://github.com/xuanfeiren/gepa-repo

4. **DSPy** (`dspy==3.1.0b1`)
   - Location: `./dspy-repo`
   - Repository: https://github.com/xuanfeiren/dspy-repo

## Usage

To use any of these packages in your Python scripts:

```python
import trace_opt
import openevolve
import gepa
import dspy
```

Since they're installed as editable packages, any changes you make to the source code in the respective directories will be immediately available without reinstalling.

## Updating Dependencies

If you need to install additional packages:

```bash
source .venv/bin/activate
uv pip install <package-name>
```

## Pulling Latest Changes

To update any of the repositories:

```bash
cd <repository-directory>
git pull
```

No reinstallation is needed since they're installed as editable packages.
