"""Separate local ROI API. Never starts production startup.py or changes its ports."""

from pathlib import Path
from urllib.parse import urlsplit

from local.gateway import LocalModelGateway
from local.messages import LocalCompletionClient
from roi_swarm.api import create_app
from roi_swarm.graph import build_graph
from roi_swarm.runner import SwarmRunner

ROOT = Path(__file__).resolve().parents[1]


def validate_local_settings(settings):
    endpoint = urlsplit(settings.provider.base_url)
    if endpoint.scheme != "http" or endpoint.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Local inference requires a loopback HTTP endpoint")
    for field in ("worker_context_tokens", "main_context_tokens"):
        if getattr(settings.context, field) > 65536:
            raise ValueError("Local context is limited to the native 65536-token guard")


class LocalSwarmRunner(SwarmRunner):
    def __init__(self):
        super().__init__(ROOT / "local/settings.yaml")
        validate_local_settings(self.settings)
        self.client = LocalCompletionClient(self.client)
        self.gateway = LocalModelGateway(self.client)
        self.graph = build_graph(self.settings, self.gateway, self.context_pipeline)


def create_local_app():
    return create_app(LocalSwarmRunner())
