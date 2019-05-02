lint: pylint flake8 mypy
pylint:
	pylint codenames
flake8:
	flake8 codenames
mypy:
	mypy codenames --strict

build-docker:
	docker build -t codenames_ai:latest --network host .

run-api-docker: build-docker
	docker run -p 8080:8080 codenames_ai:latest
