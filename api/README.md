# Sending Requests
If the API has been served, you can either send single or bulk query requests to it.

**Bash Examples.**

```bash
# single-query request
curl -X POST <user>@<address>:<port>/search -H "Content-Type: application/json" -d '{"query": "Where was Marie Curie born?", "n_docs": 1, "domains": "MassiveDS"}'

# multi-query request
curl -X POST <user>@<address>:<port>/search -H "Content-Type: application/json" -d '{"query": ["Where was Marie Curie born?", "What is the capital of France?", "Who invented the telephone?"], "n_docs": 2, "domains": "MassiveDS"}'
```

Example output of a multi-query request:
```json
{
  "message": "Search completed for '['Where was Marie Curie born?', 'What is the capital of France?', 'Who invented the telephone?']' from MassiveDS",
  "n_docs": 2,
  "query": [
    "Where was Marie Curie born?",
    "What is the capital of France?",
    "Who invented the telephone?"
  ],
  "results": {
    "n_docs": 2,
    "query": [
      "Where was Marie Curie born?",
      "What is the capital of France?",
      "Who invented the telephone?"
    ],
    "results": {
      "IDs": [
        [
          [3, 3893807],
          [17, 11728753]
        ],
        [
          [14, 12939685],
          [22, 1070951]
        ],
        [
          [28, 18823956],
          [22, 10406782]
        ]
      ],
      "passages": [
        [
          "Marie Skłodowska Curie (November 7, 1867 – July 4, 1934) was a physicist and chemist of Polish upbringing and, subsequently, French citizenship. ...",
          "=> Maria Skłodowska, better known as Marie Curie, was born on 7 November in Warsaw, Poland. ..."
        ],
        [
          "Paris is the capital and most populous city in France, as well as the administrative capital of the region of Île-de-France. ...",
          "[paʁi] ( listen)) is the capital and largest city of France. ..."
        ],
        [
          "Antonio Meucci (Florence, April 13, 1808 – October 18, 1889) was an Italian inventor. ...",
          "The telephone or phone is a telecommunications device that transmits speech by means of electric signals. ..."
        ]
      ],
      "scores": [
        [
          1.8422218561172485,
          1.8394594192504883
        ],
        [
          1.5528039932250977,
          1.5502511262893677
        ],
        [
          1.714379906654358,
          1.706493854522705
        ]
      ]
    }
  }
}
```


# Prepare Environment
```bash
conda env create -f environment.yml
conda activate scaling
```


# Serve the Index
First, search `rulin@` in this repo and replace with `<your_id@>`.

### Serve individual shards
To serve the prebuilt MassiveDS IVF index, first serve the index shards on different gpu nodes:
```bash
sbatch launch_workers.sh
```

### Serve the main node
Once the indices are on, the live endpoints will be logged in `running_ports_massiveds.txt`. 
You can then serve a main node to perform distributed online search over all running endpoints (make sure all the endpoints in the file are running):
```bash
sbatch launch_main_node.sh
```

The endpoint for the search main node will be logged in `running_ports_main_node.txt`. 




# Example: serving a public API on Hyak

#### Step 1. Run the serve script
```bash
sbatch launch_workers.sh
sbatch launch_main_node.sh
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