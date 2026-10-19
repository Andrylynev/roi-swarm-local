"""Text-only WhiteRabbitNeo consultation; never executes generated code or tools."""

import argparse
import json
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", help="Authorized lab/audit question")
    args = parser.parse_args()
    payload = {
        "model": "whiterabbit-7b",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a cybersecurity analyst assisting an authorized audit "
                    "or isolated lab. "
                    "Separate confirmed evidence from hypotheses. Do not claim to have run tools."
                ),
            },
            {"role": "user", "content": args.prompt},
        ],
        "max_tokens": 2048,
        "temperature": 0.2,
    }
    request = urllib.request.Request(
        "http://127.0.0.1:8091/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request) as response:
        result = json.load(response)
    print(result["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
