integ:
	python -m pytest ./environment/test/ -s

pytest:
	python -m pytest ./environment/core/**/*_test.py \
	  ./environment/core/*_test.py \
	  ./environment/client/*_test.py  -s

compose:
	docker-compose build
	docker-compose up

build:
	docker-compose build

up:
	docker-compose up

run:
	python agent/experiment.py