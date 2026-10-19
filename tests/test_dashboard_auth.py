"""The initial dashboard token stays out of HTTP query/access logs."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_dashboard_accepts_fragment_token_and_clears_url():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is optional")
    html = (Path(__file__).parents[1] / "src/roi_swarm/static/dashboard.html").read_text(
        encoding="utf-8"
    )
    script = html.split("let activeId=null, current=null, pinned=false;", 1)[1].split(
        "const authHeaders=", 1
    )[0]
    harness = r"""
const assert=require('node:assert/strict');
const location={hash:'#token=fixture%20token',search:'',pathname:'/dashboard'};
const stored={};
const localStorage={setItem:(k,v)=>stored[k]=v};
let clean=null;
const history={replaceState:(a,b,path)=>clean=path};
"""
    result = subprocess.run(
        [
            node,
            "-e",
            harness
            + script
            + "\n"
            + "assert.equal(stored.roiToken,'fixture token');"
            + "assert.equal(clean,'/dashboard');",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
