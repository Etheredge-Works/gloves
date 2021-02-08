import pytest
import docker


def test_component():
   client = docker.from_env()
   