import time, json, urllib.request

base = "http://localhost:5000"

tests = [
    ("/", {}),
    ("/api/constellation", {"modulation": "QPSK", "snr": 15}),
    ("/api/ofdm-spectrum", {}),
    ("/api/beam-pattern", {"n_antennas": 8}),
    ("/api/mimo-sumrate", {"n_tx": 8, "n_users": 2}),
    ("/api/numerology", {"mu": 0}),
    ("/api/ldpc-demo", {}),
]

for url, params in tests:
    body = json.dumps(params).encode()
    req = urllib.request.Request(
        f"{base}{url}", data=body,
        headers={"Content-Type": "application/json"}
    )
    try:
        r = urllib.request.urlopen(req, timeout=30)
        data = json.loads(r.read())
        if "image" in data:
            print(f"OK {url}: image={len(data['image'])} chars")
        elif "info" in data:
            print(f"OK {url}: info present")
        elif "results" in data:
            print(f"OK {url}: {len(data['results'])} results")
        else:
            print(f"OK {url}: {list(data.keys())}")
    except Exception as e:
        print(f"FAIL {url}: {e}")

print("\nDONE")
