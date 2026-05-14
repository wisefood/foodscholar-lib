.PHONY: test lint typecheck clean build upload all bump-patch bump-minor bump-major

test:
	@pytest; status=$$?; if [ $$status -eq 5 ]; then echo "No tests collected — skipping."; exit 0; else exit $$status; fi

lint:
	ruff check src tests
	ruff format --check src tests

typecheck:
	mypy src

clean:
	rm -rf dist/ build/ *.egg-info src/*.egg-info .ruff_cache .mypy_cache .pytest_cache

build: clean
	python -m hatch build

upload:
	twine upload dist/*

bump-patch:
	@python -c "import re; f='pyproject.toml'; t=open(f).read(); v=re.search(r'version\s*=\s*\"(\d+)\.(\d+)\.(\d+)\"',t).groups(); nv=f'{v[0]}.{v[1]}.{int(v[2])+1}'; open(f,'w').write(re.sub(r'version\s*=\s*\"[\d.]+\"',f'version = \"{nv}\"',t)); print(f'Bumped to {nv}')"

bump-minor:
	@python -c "import re; f='pyproject.toml'; t=open(f).read(); v=re.search(r'version\s*=\s*\"(\d+)\.(\d+)\.(\d+)\"',t).groups(); nv=f'{v[0]}.{int(v[1])+1}.0'; open(f,'w').write(re.sub(r'version\s*=\s*\"[\d.]+\"',f'version = \"{nv}\"',t)); print(f'Bumped to {nv}')"

bump-major:
	@python -c "import re; f='pyproject.toml'; t=open(f).read(); v=re.search(r'version\s*=\s*\"(\d+)\.(\d+)\.(\d+)\"',t).groups(); nv=f'{int(v[0])+1}.0.0'; open(f,'w').write(re.sub(r'version\s*=\s*\"[\d.]+\"',f'version = \"{nv}\"',t)); print(f'Bumped to {nv}')"

all: test bump-patch build upload
