integration:
	python -m pytest test/

pytest:
	python -m pytest ./client/*_test.py ./environment/*_test.py\
	 ./ingress/*_test.py ./server/*_test.py

createdb:
	python utils/main.py -r createdb

flushdb:
	python utils/main.py -r flush

dropdb:
	python utils/main.py -r drop

runs:
	python server/main.py

runi:
	python ingress/features_ingress.py

runp:
	python ingress/prices_ingress.py

runa:
	python agent/experiment.py