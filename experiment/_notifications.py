"""Optional authenticated notifications for explicitly configured private topics."""

import os
import re
import urllib.request
from http.client import HTTPException


def ntfy(title, body, tags="brain"):
    topic = os.environ.get("DODO_NTFY_TOPIC", "")
    token = os.environ.get("DODO_NTFY_TOKEN", "").strip()
    if not token or not re.fullmatch(r"[A-Za-z0-9_-]{32,64}", topic):
        return
    request = urllib.request.Request(
        f"https://ntfy.sh/{topic}",
        data=body.encode(),
        headers={"Title": title, "Tags": tags, "Authorization": f"Bearer {token}"},
    )
    try:
        urllib.request.urlopen(request, timeout=10).close()
    except (HTTPException, OSError, ValueError) as error:
        print(f"[--] ntfy failed: {type(error).__name__}", flush=True)
