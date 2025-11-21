PYTHONPATH?=src
UV?=uv

# Workaround for building pydantic-core on Python 3.14 via PyO3 ABI3
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1

.PHONY: sync lint fix fmt fmt-check type test pre-commit ci build publish release

sync:
	$(UV) sync --group dev

.PHONY: pin-3.13 pin-3.14 pyver

# Pin uv to Python 3.13 (recommended until 3.14 wheels are ready)
pin-3.13:
	$(UV) python install 3.13
	$(UV) python pin 3.13

# Pin uv back to Python 3.14
pin-3.14:
	$(UV) python install 3.14
	$(UV) python pin 3.14

pyver:
	$(UV) python list

lint:
	$(UV) run ruff check .

fix:
	$(UV) run ruff check . --fix

fmt:
	$(UV) run black .

fmt-check:
	$(UV) run black --check .

type:
	$(UV) run ty check --exclude legacy/ --exclude backups/ --exclude .specstory/ src

test:
	PYTHONPATH=$(PYTHONPATH) $(UV) run pytest -q $(PYTHONPATH)

.PHONY: cov
cov:
	PYTHONPATH=$(PYTHONPATH) $(UV) run pytest -q $(PYTHONPATH) --cov=src --cov-report=term-missing

.PHONY: extras-all
extras-all:
	$(UV) pip install '.[all]'

.PHONY: env-clean
env-clean:
	rm -rf .venv
	$(UV) sync --group dev

pre-commit:
	$(UV) run pre-commit install
	$(UV) run pre-commit run --all-files

ci: lint fmt-check type test

build:
	$(UV) build

publish:
	# Set UV_PUBLISH_TOKEN or run with `uv publish --token ...`
	$(UV) publish

release: test build
	@echo "Tagging release..."
	@export gittag=v$$(rg -n "^version\s*=\s*\"(.*)\"" -o -r '$$1' pyproject.toml | head -1); \
	git add -A && git commit -m "chore(release): $$gittag" || true; \
	git tag -a "$$gittag" -m "Release $$gittag" || true; \
	echo "Run 'git push --follow-tags' to publish the tag";
