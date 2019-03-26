integration:
	python -m pytest test/

pytest:
	python -m pytest ./client/*_test.py ./environment/*_test.py\
	 ./ingress/*_test.py ./server/*_test.py

compose:
	docker-compose build
	docker-compose up

build:
	docker-compose build

up:
	docker-compose up

run:
	python agent/experiment.py