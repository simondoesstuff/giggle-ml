sync target='layerlab':
	./sync {{target}}

typecheck path='src':
	uv run basedpyright {{path}}

test *args:
	just typecheck && echo ''
	uv pip install -e . && uv run pytest {{args}}

alias i := install
install:
	uv sync && uv pip install -e .

pull path target='layerlab':
	scp layerlab:~/projects/giggle-ml/{{path}} .
