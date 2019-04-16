import faust

app = faust.App(
    'id',
    broker=['kafka://kafka1.example.com:9092',
            'kafka://kafka2.example.com:9092'],
)