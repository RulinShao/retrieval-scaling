# Serving a public API on Hyak

#### Step 1. Run the serve script
```bash
python serve2.py
```

#### Step 2. Forward the port to a public address on tricycle
```bash
ssh <UW_ID>@tricycle.cs.washington.edu
ssh -NL 0.0.0.0:<FORWARD_PORT>:<job node>:<SERVE_PORT> klone-login
```

#### Step 3. Send request to the address
```bash
curl -X POST localhost:<FORWARD_PORT>/search -H "Content-Type: application/json" -d '{"query": "example query", "domains": "pes2o"}'
```