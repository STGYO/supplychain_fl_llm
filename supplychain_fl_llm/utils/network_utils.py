from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import socket

DEFAULT_FL_PORT = 8080
DEFAULT_BLOCKCHAIN_PORT = 8545


@dataclass(frozen=True)
class Endpoint:
    host: str
    port: int

    def as_address(self) -> str:
        return f"{self.host}:{self.port}"


def parse_host_port(value: str, default_port: int) -> Endpoint:
    """Parse an endpoint string like host:port into an Endpoint."""
    if not value:
        raise ValueError("Endpoint value cannot be empty")

    if ":" in value:
        host, port_text = value.rsplit(":", maxsplit=1)
        if not port_text.isdigit():
            raise ValueError(f"Invalid port in endpoint: {value}")
        port = int(port_text)
    else:
        host = value
        port = default_port

    if not host:
        raise ValueError(f"Invalid host in endpoint: {value}")

    if port < 1 or port > 65535:
        raise ValueError(f"Port out of range in endpoint: {value}")

    return Endpoint(host=host, port=port)


def is_valid_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def resolve_host(host: str) -> str:
    return socket.gethostbyname(host)


def is_tcp_port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        try:
            return sock.connect_ex((host, port)) == 0
        except OSError:
            return False


def ensure_endpoint_reachable(endpoint: Endpoint, timeout: float = 2.0) -> None:
    if not is_tcp_port_open(endpoint.host, endpoint.port, timeout=timeout):
        raise ConnectionError(
            f"Endpoint {endpoint.as_address()} is not reachable over TCP. "
            "Check firewall rules, LAN IP, and service binding."
        )
