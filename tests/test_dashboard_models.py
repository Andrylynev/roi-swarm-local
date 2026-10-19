"""Offline contracts for the dashboard; no service or provider calls."""

import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest

DASHBOARD = Path(__file__).parents[1] / "src/roi_swarm/static/dashboard.html"
ROLES = ("router", "planner", "worker", "critic", "synthesizer")


class DashboardHTML(HTMLParser):
    def __init__(self, html: str) -> None:
        super().__init__()
        self.elements: list[tuple[str, dict]] = []
        self.feed(html)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.elements.append((tag, dict(attrs)))


def test_model_configuration_has_accessible_editable_role_inputs() -> None:
    html = DASHBOARD.read_text(encoding="utf-8")
    elements = DashboardHTML(html).elements
    by_id = {attrs["id"]: (tag, attrs) for tag, attrs in elements if "id" in attrs}
    assert "model-config" in by_id
    assert html.index('id="model-config"') < html.index('class="hero"')
    for role in ROLES:
        tag, attrs = by_id[f"model-{role}"]
        assert tag == "input"
        assert attrs["list"] == "model-options"
        assert "readonly" not in attrs
        assert any(t == "label" and a.get("for") == f"model-{role}" for t, a in elements)
        assert any(a.get("data-model-role") == role for _, a in elements)
    assert by_id["model-options"][0] == "datalist"
    assert by_id["model-status"][1]["aria-live"] == "polite"
    for name in ("model-refresh", "model-apply", "model-discard"):
        assert by_id[name][0] == "button"


def test_model_settings_offline_javascript_contract() -> None:
    """Run real UI logic against an in-memory DOM and HTTP boundary."""
    html = DASHBOARD.read_text(encoding="utf-8")
    assert "/* ---------- model configuration ---------- */" in html
    script = html.split("/* ---------- model configuration ---------- */", 1)[1].split(
        "/* ---------- end model configuration ---------- */", 1
    )[0]
    assert html.index("const $=") < html.index("const modelSettings=")
    assert "Opus" not in html and "Luna" not in html
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is optional; static HTML contract remains available")
    harness = r"""
const assert = require('node:assert/strict');
class Element {
  constructor() { this.value=''; this.textContent=''; this.dataset={};
    this.disabled=false; this.children=[]; this.handlers={}; }
  addEventListener(name, callback) { this.handlers[name]=callback; }
  setAttribute(name, value) { this[name]=value; }
  replaceChildren(...children) { this.children=children; }
  set innerHTML(value) { throw Error('Unsafe HTML write: '+value); }
}
const elements=new Map();
const $=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id)};
const roles=['router','planner','worker','critic','synthesizer'];
const who=roles.map(role=>{const el=new Element();el.dataset.modelRole=role;return el});
const document={querySelectorAll:()=>who,createElement:()=>new Element(),hidden:false};
const authHeaders=()=>({Authorization:'Bearer offline'});
const calls=[];
let serverRoles=Object.fromEntries(roles.map(role=>[role,role+'-original']));
let active=0, putStatus=200, catalogStatus=200, getStatus=200, delayRead=null;
const malicious='<img src=x onerror=alert(1)>';
const fetch=async (url, options={})=>{
  assert.equal(options.headers.Authorization,'Bearer offline');
  calls.push({url,...options});
  if(url==='/api/model-options')return {ok:catalogStatus===200,status:catalogStatus,
    json:async()=>({models:[malicious,'custom-model'],source:'configured',
      warning:'offline catalog'})};
  if(options.method==='PUT') {
    assert.equal(options.headers['Content-Type'],'application/json');
    const body=JSON.parse(options.body);
    assert.deepEqual(Object.keys(body.roles),roles);
    if(putStatus===200)serverRoles={...body.roles};
    return {ok:putStatus===200,status:putStatus,
      json:async()=>({roles:serverRoles,active_operations:active})};
  }
  if(delayRead)await delayRead;
  return {ok:getStatus===200,status:getStatus,
    json:async()=>({roles:{...serverRoles},active_operations:active})};
};
"""
    scenarios = r"""
(async()=>{
  await Promise.all([modelSettings.refreshSettings(),modelSettings.refreshCatalog()]);
  assert.equal($('model-router').value,'router-original');
  assert.equal($('model-apply').disabled,true);
  assert.equal($('model-options').children[0].textContent,malicious);
  assert.match($('model-catalog').textContent,/offline catalog/);
  const edit=(role,value)=>{const el=$('model-'+role);el.value=value;el.handlers.input()};
  edit('worker',malicious);
  assert.equal($('model-config').dataset.dirty,'true');
  const reads=calls.length;
  await modelSettings.refreshSettings();
  assert.equal(calls.length,reads,'dirty state must skip polling');
  assert.equal($('model-worker').value,malicious);
  for(const status of [409,422,503]) {
    putStatus=status;
    await $('model-form').handlers.submit({preventDefault(){}});
    assert.equal($('model-worker').value,malicious);
    assert.equal($('model-config').dataset.dirty,'true');
    assert.equal($('model-status').dataset.kind,'error');
    assert.match($('model-status').textContent,new RegExp(String(status)));
    assert.equal($('model-worker').disabled,false);
  }
  putStatus=200;
  const save=$('model-form').handlers.submit({preventDefault(){}});
  assert.equal($('model-worker').disabled,true);
  assert.equal($('model-discard').disabled,true);
  assert.equal($('model-refresh').disabled,true);
  await save;
  assert.equal(serverRoles.worker,malicious);
  assert.equal(who[2].textContent,malicious);
  assert.equal($('model-status').dataset.kind,'success');
  assert.equal($('model-config').dataset.dirty,'false');
  edit('router','temporary');
  $('model-discard').handlers.click();
  assert.equal($('model-router').value,'router-original');
  let release;
  delayRead=new Promise(resolve=>release=resolve);
  const poll=modelSettings.refreshSettings();
  edit('router','edit-during-poll');
  release();await poll;delayRead=null;
  assert.equal($('model-router').value,'edit-during-poll');
  $('model-discard').handlers.click();
  serverRoles.router='external-update';active=2;
  await modelSettings.refreshSettings();
  assert.equal($('model-router').value,'external-update');
  assert.match($('model-activity').textContent,/2/);
  getStatus=503;
  await modelSettings.refreshSettings();
  assert.equal($('model-status').dataset.kind,'error');
  assert.equal($('model-router').value,'external-update');
  getStatus=200;
  await modelSettings.refreshSettings();
  assert.notEqual($('model-status').dataset.kind,'error','successful polling clears stale errors');
  assert.equal(calls.filter(c=>c.url==='/api/model-options').length,1);
  catalogStatus=503;
  await modelSettings.refreshCatalog();
  assert.equal($('model-options').children[0].textContent,malicious);
  assert.match($('model-catalog').textContent,/503/);
  catalogStatus=200;
  await modelSettings.refreshCatalog();
  assert.equal(calls.filter(c=>c.url==='/api/model-options').length,3);
  console.log('offline model UI: OK');
})().catch(error=>{console.error(error);process.exitCode=1});
"""
    result = subprocess.run(
        [node, "-e", harness + script + scenarios],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "offline model UI: OK" in result.stdout
    assert "setInterval(modelSettings.refreshSettings,6000)" in html