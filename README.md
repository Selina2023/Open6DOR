# Open6DOR

```bash
cd vision/GroundedSAM/GroundingDINO
pip install -e .
cd ../segment_anything
pip install -e .
cd ../../..
```



## Troubleshooting

- requests.exceptions.ConnectionError: HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /bert-base-uncased/resolve/main/tf_model.h5 (Caused by NewConnectionError('<urllib3.connection.HTTPSConnection object at 0x7f4769a3cc40>: Failed to establish a new connection: [Errno 101] Network is unreachable'))
    - Solution: Network error, In China, try global proxy.