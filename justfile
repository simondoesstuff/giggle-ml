sync target='layerlab':
	./sync {{target}}

test:
	uv pip install -e . && uv run pytest
