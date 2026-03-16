sync target='layerlab':
	./sync {{target}}

test *args:
	uv pip install -e . && uv run pytest {{args}}

typecheck path='src':
	uv run basedpyright {{path}}

alias i := install
install:
	uv sync && uv pip install -e .
