"""Offline sleep-panel contracts; never start inference or contact a provider."""

import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest

DASHBOARD = Path(__file__).parents[1] / "src/roi_swarm/static/dashboard.html"


class DashboardHTML(HTMLParser):
    def __init__(self, html: str) -> None:
        super().__init__()
        self.elements: list[tuple[str, dict]] = []
        self.feed(html)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.elements.append((tag, dict(attrs)))


def test_sleep_panel_is_compact_accessible_configuration_before_hero() -> None:
    html = DASHBOARD.read_text(encoding="utf-8")
    elements = DashboardHTML(html).elements
    by_id = {attrs["id"]: (tag, attrs) for tag, attrs in elements if "id" in attrs}
    assert "sleep-config" in by_id
    assert html.index('id="model-config"') < html.index('id="sleep-config"')
    assert html.index('id="sleep-config"') < html.index('class="hero"')
    tag, attrs = by_id["sleep-enabled"]
    assert tag == "input" and attrs["type"] == "checkbox"
    assert "disabled" in attrs
    assert any(t == "label" and a.get("for") == "sleep-enabled" for t, a in elements)
    assert by_id["sleep-state"][1]["aria-live"] == "polite"
    assert by_id["sleep-status"][1]["aria-live"] == "polite"
    assert by_id["sleep-config"][1]["aria-labelledby"] == "sleep-title"
    assert by_id["sleep-enabled"][1]["aria-describedby"] == "sleep-help"
    assert by_id["sleep-calls"][0] in {"span", "dd"}
    assert not any(tag == "progress" for tag, _ in elements)
    for name in (
        "sleep-model",
        "sleep-mode",
        "sleep-foreground",
        "sleep-daily-limit",
        "sleep-cycle-limit",
        "sleep-steps",
        "sleep-pending",
        "sleep-next-step",
        "sleep-error",
        "sleep-improvement",
        "sleep-restart",
    ):
        assert name in by_id
    assert by_id["sleep-snapshot-btn"][0] == "button"
    assert by_id["sleep-details"][0] == "details"
    assert by_id["sleep-snapshot"][0] == "pre"
    panel = html.split('id="sleep-config"', 1)[1].split('<section class="hero">', 1)[0]
    for phrase in (
        "Режим сна",
        "Снимок",
        "задачи пользователя",
        "наблюдения",
        "гипотезы",
        "следующие проверки",
        "не внутреннее состояние модели",
        "вызовы внешнего провайдера",
        "могут быть платными",
        "не возобновляется с прерванного токена",
        "новый запрос",
        "приостанавливает",
        "без суточного лимита",
        "без лимита шагов в цикле",
        "выключите",
        "тестов",
        "ревью",
        "чистого Git",
        "перезапуск",
        "отдельно",
    ):
        assert phrase in panel
    assert "3 попыток" not in panel and "24 вызовов" not in panel
    assert "<script src=" not in html and "<link " not in html
    assert "if(document.hidden||t-last<33)return" in html


def run_dashboard_javascript(scenario: str, module: str = "sleep") -> None:
    html = DASHBOARD.read_text(encoding="utf-8")
    start = f"/* ---------- {module} configuration ---------- */"
    end = f"/* ---------- end {module} configuration ---------- */"
    assert start in html
    script = html.split(start, 1)[1].split(end, 1)[0]
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is optional; static HTML contracts remain available")
    harness = r"""
const assert = require('node:assert/strict');
class Element {
  constructor() { this.textContent='';this.disabled=true;this.checked=false;
    this.hidden=false;this.dataset={};this.handlers={}; }
  addEventListener(name,callback) { this.handlers[name]=callback; }
  setAttribute(name,value) { this[name]=value; }
  set innerHTML(value) { throw Error('Unsafe HTML write: '+value); }
}
const elements=new Map();
const $=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id)};
const document={hidden:false,querySelectorAll:()=>[],createElement:()=>new Element()};
const authHeaders=()=>({Authorization:'Bearer offline'});
let confirmed=false,confirmations=[];
const confirm=message=>{confirmations.push(message);return confirmed};
const calls=[];
const malicious='<img src=x onerror=alert(1)>';
let server={enabled:false,state:'idle',model:malicious,foreground_count:2,daily_calls:127,
  daily_call_limit:null,max_steps_per_cycle:null,step_count:54,pending_cycles:8,
  last_error:null,auto_improve:true,auto_improve_state:'blocked',
  auto_improve_reason:'Git dirty',restart_required:true};
let getStatus=200,putStatus=200,snapshotStatus=200,releaseStatus=200,delayRead=null;
let modelContext={worker_context_tokens:800000,main_context_tokens:1000000};
let snapshot={current:{observations:[malicious],hypotheses:['check'],
  next_step:malicious},queue:[{next_step:'next check'}]};
const fetch=async (url,options={})=>{
  assert.equal(options.headers.Authorization,'Bearer offline');
  calls.push({url,...options});
  if(url==='/api/model-settings') {
    assert.notEqual(options.method,'PUT','context display is read only');
    return {ok:true,status:200,json:async()=>({
      roles:Object.fromEntries(['router','planner','worker','critic','synthesizer'].map(role=>[role,role+'-id'])),
      active_operations:0,context_limits:modelContext})};
  }
  if(url==='/api/sleep/snapshot')return {ok:snapshotStatus===200,status:snapshotStatus,
    json:async()=>snapshot};
  if(url==='/api/sleep/release-tools') {
    assert.equal(options.method,'POST');
    if(releaseStatus===200)server={...server,pending_tool_turns:0,tool_lease_recovery_hold:false};
    return {ok:releaseStatus===200,status:releaseStatus,json:async()=>({...server})};
  }
  assert.equal(url,'/api/sleep','sleep controls must never change models or infer');
  if(options.method==='PUT') {
    assert.equal(options.headers['Content-Type'],'application/json');
    const body=JSON.parse(options.body);
    assert.deepEqual(Object.keys(body),['enabled']);
    assert.equal(typeof body.enabled,'boolean');
    if(putStatus===200)server={...server,enabled:body.enabled};
    return {ok:putStatus===200,status:putStatus,json:async()=>({...server})};
  }
  const body={...server};
  if(delayRead)await delayRead;
  return {ok:getStatus===200,status:getStatus,json:async()=>body};
};
"""
    result = subprocess.run(
        [
            node,
            "-e",
            harness + script + "\n(async()=>{\n" + scenario + "\n})()"
            ".catch(error=>{console.error(error);process.exitCode=1});",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_sleep_polling_reports_real_counts_and_toggle_reads_back() -> None:
    run_dashboard_javascript(r"""
assert.equal($('sleep-enabled').disabled,true);
getStatus=503;await sleepSettings.refresh();
assert.equal($('sleep-enabled').disabled,true);
assert.match($('sleep-state').textContent,/Недоступно/);
assert.match($('sleep-status').textContent,/503/);
getStatus=200;await sleepSettings.refresh();
assert.equal($('sleep-enabled').disabled,false);
assert.equal($('sleep-enabled').checked,false);
assert.match($('sleep-mode').textContent,/выключен/);
assert.match($('sleep-state').textContent,/idle/);
assert.equal($('sleep-model').textContent,malicious);
assert.equal($('sleep-foreground').textContent,'2');
assert.equal($('sleep-calls').textContent,'127');
assert.equal($('sleep-daily-limit').textContent,'без лимита');
assert.equal($('sleep-cycle-limit').textContent,'без лимита');
assert.equal($('sleep-steps').textContent,'54');
assert.equal($('sleep-pending').textContent,'8');
assert.match($('sleep-improvement').textContent,/blocked/);
assert.match($('sleep-improvement').textContent,/Git dirty/);
assert.equal($('sleep-restart').hidden,false);
assert.match($('sleep-restart').textContent,/отдельно/);
$('sleep-enabled').checked=true;
const save=$('sleep-enabled').handlers.change();
assert.equal($('sleep-enabled').disabled,true);
await save;
assert.equal(server.enabled,true);
assert.equal($('sleep-enabled').checked,true);
assert.match($('sleep-mode').textContent,/включён/);
assert.equal(calls.at(-2).method,'PUT');
assert.notEqual(calls.at(-1).method,'PUT','successful PUT must be read back');
server={...server,state:'paused',foreground_count:5,daily_calls:128,
  daily_call_limit:500,max_steps_per_cycle:100,last_error:malicious,
  auto_improve_state:'tested',auto_improve_reason:null,restart_required:false,
  next_step:'verify artifacts'};
await sleepSettings.refresh();
assert.match($('sleep-state').textContent,/paused/);
assert.equal($('sleep-foreground').textContent,'5');
assert.equal($('sleep-calls').textContent,'128');
assert.equal($('sleep-daily-limit').textContent,'500');
assert.equal($('sleep-cycle-limit').textContent,'100');
assert.equal($('sleep-next-step').textContent,'verify artifacts');
assert.equal($('sleep-error').textContent,malicious);
assert.equal($('sleep-error').hidden,false);
assert.equal($('sleep-restart').hidden,true);
assert.match($('sleep-improvement').textContent,/tested/);
assert.doesNotMatch($('sleep-improvement').textContent,/Git dirty/);
$('sleep-enabled').checked=false;await $('sleep-enabled').handlers.change();
assert.equal(server.enabled,false,'same toggle stops sleep');
const reads=calls.length;document.hidden=true;await sleepSettings.refresh();
assert.equal(calls.length,reads,'hidden dashboard must not poll sleep');
""")
    html = DASHBOARD.read_text(encoding="utf-8")
    refresh = html.split("async function refresh(){", 1)[1].split("function renderTotals", 1)[0]
    assert "sleepSettings.refresh()" in refresh
    assert "setInterval(refresh,1200)" in html
    sleep_script = html.split("/* ---------- sleep configuration ---------- */", 1)[1].split(
        "/* ---------- end sleep configuration ---------- */", 1
    )[0]
    assert "setInterval" not in sleep_script and "requestAnimationFrame" not in sleep_script


def test_sleep_snapshot_is_on_demand_public_text_with_next_check() -> None:
    run_dashboard_javascript(r"""
await sleepSettings.refresh();
assert.equal(calls.filter(c=>c.url==='/api/sleep/snapshot').length,0);
assert.equal($('sleep-snapshot-btn').disabled,false);
const snapshotRead=$('sleep-snapshot-btn').handlers.click();
assert.equal($('sleep-snapshot-btn').disabled,true);
await snapshotRead;
assert.equal($('sleep-snapshot-btn').disabled,false);
assert.equal($('sleep-snapshot').textContent,JSON.stringify(snapshot,null,2));
assert.equal($('sleep-next-step').textContent,malicious);
await sleepSettings.refresh();
assert.equal($('sleep-next-step').textContent,malicious);
assert.equal(calls.filter(c=>c.url==='/api/sleep/snapshot').length,1);
snapshotStatus=503;await $('sleep-snapshot-btn').handlers.click();
assert.match($('sleep-status').textContent,/503/);
assert.equal($('sleep-status').dataset.kind,'error');
assert.equal($('sleep-snapshot').textContent,JSON.stringify(snapshot,null,2));
snapshotStatus=200;snapshot={current:null,queue:[{next_step:'queued check'}]};
await $('sleep-snapshot-btn').handlers.click();
assert.equal($('sleep-next-step').textContent,'queued check');
snapshot={current:null,queue:[]};await $('sleep-snapshot-btn').handlers.click();
assert.equal($('sleep-next-step').textContent,'—');
assert.deepEqual(JSON.parse($('sleep-snapshot').textContent),snapshot);
assert.equal(calls.filter(c=>c.method==='PUT').length,0,'snapshot must be read only');
""")


def test_sleep_failures_never_forge_confirmation_or_overwrite_saved_toggle() -> None:
    run_dashboard_javascript(r"""
for(const code of [401,503]) {
  getStatus=code;await sleepSettings.refresh();
  assert.equal($('sleep-enabled').disabled,true);
  assert.match($('sleep-status').textContent,new RegExp(String(code)));
}
getStatus=200;await sleepSettings.refresh();
for(const code of [401,409,422,503]) {
  putStatus=code;$('sleep-enabled').checked=true;
  await $('sleep-enabled').handlers.change();
  assert.equal(server.enabled,false);
  assert.equal($('sleep-enabled').checked,false);
  assert.equal($('sleep-enabled').disabled,false);
  assert.equal($('sleep-status').dataset.kind,'error');
  assert.match($('sleep-status').textContent,new RegExp(String(code)));
}
putStatus=200;getStatus=503;$('sleep-enabled').checked=true;
await $('sleep-enabled').handlers.change();
assert.equal(server.enabled,true);
assert.equal($('sleep-enabled').disabled,true,'failed read-back is unconfirmed');
assert.equal($('sleep-status').dataset.kind,'error');
getStatus=200;await sleepSettings.refresh();
assert.equal($('sleep-enabled').checked,true);
assert.equal($('sleep-enabled').disabled,false);
let release;delayRead=new Promise(resolve=>release=resolve);
const poll=sleepSettings.refresh();
const afterPollStart=calls.length;await sleepSettings.refresh();
assert.equal(calls.length,afterPollStart,'only one sleep poll may be in flight');
$('sleep-enabled').checked=false;
const save=$('sleep-enabled').handlers.change();
release();await Promise.all([poll,save]);delayRead=null;
assert.equal(server.enabled,false);
assert.equal($('sleep-enabled').checked,false,'old poll cannot undo a new setting');
assert.equal($('sleep-enabled').disabled,false);
server={enabled:null,state:'invalid'};await sleepSettings.refresh();
assert.equal($('sleep-enabled').disabled,true);
assert.equal($('sleep-status').dataset.kind,'error');
server={enabled:true,state:'idle',daily_call_limit:null,max_steps_per_cycle:null};
await sleepSettings.refresh();
assert.equal($('sleep-enabled').disabled,false);
assert.equal($('sleep-restart').hidden,true);
assert.equal($('sleep-error').hidden,true);
assert.equal($('sleep-calls').textContent,'—','missing counter is not zero');
""")


def test_abandoned_tools_require_explicit_confirmed_release() -> None:
    html = DASHBOARD.read_text(encoding="utf-8")
    by_id = {a["id"]: (t, a) for t, a in DashboardHTML(html).elements if "id" in a}
    assert by_id["sleep-release-tools"][0] == "button"
    assert "disabled" in by_id["sleep-release-tools"][1]
    assert "Снять ожидание прерванных tools" in html
    assert "даже после перезапуска" in html
    run_dashboard_javascript(r"""
assert.equal($('sleep-release-tools').disabled,true);
await sleepSettings.refresh();
assert.equal($('sleep-release-tools').disabled,true);
await $('sleep-release-tools').handlers.click();
assert.equal(confirmations.length,0);
server={...server,pending_tool_turns:2,tool_lease_recovery_hold:true};
await sleepSettings.refresh();
assert.equal($('sleep-release-tools').disabled,false);
assert.match($('sleep-tools').textContent,/2/);
assert.match($('sleep-tools').textContent,/восстановлен/);
const beforeCancel=calls.length;await $('sleep-release-tools').handlers.click();
assert.equal(calls.length,beforeCancel,'cancel must not issue POST');
assert.match(confirmations[0],/внешние tools/);
assert.match(confirmations[0],/остановлены/);
confirmed=true;
for(const code of [409,503]) {
  releaseStatus=code;await $('sleep-release-tools').handlers.click();
  assert.equal(server.pending_tool_turns,2);
  assert.equal($('sleep-status').dataset.kind,'error');
  assert.match($('sleep-status').textContent,new RegExp(String(code)));
  assert.equal($('sleep-release-tools').disabled,false);
}
releaseStatus=200;const release=$('sleep-release-tools').handlers.click();
assert.equal($('sleep-release-tools').disabled,true);await release;
assert.equal(server.pending_tool_turns,0);
assert.equal($('sleep-release-tools').disabled,true);
assert.equal(calls.at(-2).url,'/api/sleep/release-tools');
assert.equal(calls.at(-1).url,'/api/sleep','release must be read back');
server={...server,tool_lease_recovery_hold:true};await sleepSettings.refresh();
assert.equal($('sleep-release-tools').disabled,false,'recovery hold also needs manual release');
""")


def test_model_context_limits_are_read_only_server_values() -> None:
    html = DASHBOARD.read_text(encoding="utf-8")
    model_panel = html.split('id="model-config"', 1)[1].split('id="sleep-config"', 1)[0]
    by_id = {a["id"]: (t, a) for t, a in DashboardHTML(html).elements if "id" in a}
    assert by_id["model-context-limits"][0] == "p"
    assert 'id="model-context-limits"' in model_panel
    run_dashboard_javascript(
        r"""
await modelSettings.refreshSettings();
assert.match($('model-context-limits').textContent,/800000/);
assert.match($('model-context-limits').textContent,/1000000/);
assert.match($('model-context-limits').textContent,/Worker/);
assert.match($('model-context-limits').textContent,/Main/);
modelContext={worker_context_tokens:123,main_context_tokens:456};
await modelSettings.refreshSettings();
assert.match($('model-context-limits').textContent,/123/);
assert.match($('model-context-limits').textContent,/456/);
modelContext=undefined;await modelSettings.refreshSettings();
assert.doesNotMatch($('model-context-limits').textContent,/800000|1000000|123|456/);
assert.match($('model-context-limits').textContent,/—/);
assert.equal(calls.filter(c=>c.method==='PUT').length,0);
""",
        module="model",
    )


def test_dashboard_entire_inline_javascript_parses_offline() -> None:
    html = DASHBOARD.read_text(encoding="utf-8")
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is optional; static HTML contracts remain available")
    scripts = [part.split("</script>", 1)[0] for part in html.split("<script>")[1:]]
    assert scripts
    for script in scripts:
        result = subprocess.run(
            [node, "--check"],
            input=script,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=20,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
