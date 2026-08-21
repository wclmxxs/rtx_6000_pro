#!/usr/bin/env python3
from __future__ import annotations

import os
import socket

import uvicorn


def create_listeners(port: int) -> list[socket.socket]:
    listeners: list[socket.socket] = []
    try:
        ipv4 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listeners.append(ipv4)
        ipv4.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        ipv4.bind(("0.0.0.0", port))
        actual_port = int(ipv4.getsockname()[1])
        ipv4.listen(2048)
        ipv4.setblocking(False)

        ipv6 = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        listeners.append(ipv6)
        ipv6.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Keep the IPv6 socket separate from the explicit IPv4 socket. This
        # works regardless of the host's net.ipv6.bindv6only setting.
        ipv6.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        ipv6.bind(("::", actual_port))
        ipv6.listen(2048)
        ipv6.setblocking(False)
        return listeners
    except Exception:
        for listener in listeners:
            listener.close()
        raise


def main() -> None:
    port = int(os.getenv("PORT", "30010"))
    listeners = create_listeners(port)
    print(
        f"API listening on IPv4 0.0.0.0:{port} and IPv6 [::]:{port}",
        flush=True,
    )
    config = uvicorn.Config(
        "app.server:app",
        workers=1,
        timeout_keep_alive=120,
    )
    uvicorn.Server(config).run(sockets=listeners)


if __name__ == "__main__":
    main()
