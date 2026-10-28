"""One bounded local HTTP request; parent process handles cancellation."""
import sys
import urllib.request


def main():
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat", data=sys.stdin.buffer.read(),
        headers={"Content-Type": "application/json"}, method="POST")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=120) as response:
        data = response.read(2_000_001)
        if len(data) > 2_000_000:
            raise ValueError("Response too large")
        sys.stdout.buffer.write(data)


if __name__ == "__main__":
    main()
