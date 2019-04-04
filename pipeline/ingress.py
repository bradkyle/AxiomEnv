import threading
import time
import argparse
from enum import Enum
import argparse
from kafka import KafkaProducer
import json

class Ingress():
    def __init__(
        self,
        maintain_every=360,
        topic=None,
        url="localhost:9092"
    ):
        self.maintain_every = maintain_every
        
        self.producer = self.connect_kafka_producer(url)

        if topic is None:
            raise ValueError("Topic name must be set")
        else:
            self.topic=topic
        

    def setup(self):
        self._setup()

    def start(self):
        thread = threading.Thread(target=self.maintain, args=())
        thread.daemon = True
        if not thread.is_alive():
            thread.start()

        self._start()

    def stop(self):
        print("Stopping...")
        self._stop()

    def run(self):
        self.setup()
        self.start()

    def reset(self):
        self.stop()
        self.run()

    def maintain(self):
        while True:
            time.sleep(self.maintain_every)
            self._maintain()

    def _setup(self):
        raise NotImplementedError

    def _maintain(self):
        raise NotImplementedError

    def _stop(self):
        raise NotImplementedError

    def _start(self):
        raise NotImplementedError

    def publish(self, key, value):
        try:
            key_bytes = bytes(json.dumps(key), encoding='utf-8')
            value_bytes = bytes(json.dumps(value), encoding='utf-8')
            self.producer.send(self.topic, key=key_bytes, value=value_bytes)
            self.producer.flush()
            print('Message published successfully.')
        except Exception as ex:
            print('Exception in publishing message')
            print(ex)

    def connect_kafka_producer(self, url):
        _producer = None
        try:
            # host.docker.internal is how a docker container connects to the local
            # machine.
            # Don't use in production, this only works with Docker for Mac in
            # development
            _producer = KafkaProducer(
                bootstrap_servers=[url],
                api_version=(0, 10))
        except Exception as ex:
            print('Exception while connecting Kafka')
            print(ex)
        finally:
            return _producer

