"""Synthetic-only offline-player structure, provenance and injection guards."""
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest


PATH = Path(__file__).resolve().parents[2] / "scripts/build_intervention_player.py"
SPEC = importlib.util.spec_from_file_location("intervention_player", PATH)
player = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(player)


def payload():
    return {"version": "synthetic", "works": [{
        "id": "w1", "title": "Synthetic clip", "source_id": "fixture",
        "duration_seconds": 1, "seconds_per_quarter": 0.5,
        "notes": [{"id": "n1", "pitch": 60, "onset_qn": [0, 1], "duration_qn": [1, 1],
                   "onset_seconds": 0, "duration_seconds": 0.5}],
        "interventions": [{"id": "h1", "aspect": "harmony", "title": "Pitch",
                           "description": "Synthetic pitch edit", "location_seconds": 0,
                           "patches": [{"note_id": "n1", "pitch": 61}], "context_note_ids": ["n1"]}],
        "pairwise": [], "presets": [{"name": "One edit", "ids": ["h1"], "file_stem": "w1/h1"}],
        "original_wav": "w1/original.wav", "original_midi": "w1/original.mid",
    }]}


def test_embedded_payload_roundtrips_and_does_not_close_script(tmp_path):
    data = payload()
    data["works"][0]["title"] = '</script><script>alert("x")</script> & < >'
    path = tmp_path / "player.html"
    player.build_player(data, path)
    html = path.read_text()
    value = re.search(r'<script id="payload" type="application/json">(.*?)</script>', html, re.S).group(1)
    assert json.loads(value) == data
    assert '<script>alert("x")</script>' not in html
    assert "\\u003c/script" in html
    assert "innerHTML" not in html


def test_offline_controls_and_quality_boundary(tmp_path):
    path = tmp_path / "player.html"
    player.build_player(payload(), path)
    html = path.read_text()
    for name in ("workSelect", "playSelected", "playOriginal", "stopPlayback", "clearAll", "selectAll", "pianoRoll", "compatibility", "presets"):
        assert f'id="{name}"' in html
    assert 'src="https:' not in html and 'src="http:' not in html
    assert "fetch(" not in html
    assert "semantic_revalidation" in html
    assert "воспроизведение этой комбинации заблокировано" in html
    assert "не подтверждена человеком" in html
    assert "120 BPM" in html
    assert ".note_id" in html
    with pytest.raises(FileExistsError):
        player.build_player(payload(), path)


@pytest.mark.parametrize("path", ["https://example.com/a.wav", "../escape.wav", "/tmp/a.wav", "javascript:alert(1)", "folder\\escape.wav"])
def test_media_stays_local(path):
    data = payload()
    data["works"][0]["original_wav"] = path
    with pytest.raises(ValueError):
        player.validate_payload(data)


def test_unknown_note_and_preset_ids_fail():
    data = payload()
    data["works"][0]["interventions"][0]["patches"][0]["note_id"] = "missing"
    with pytest.raises(ValueError, match="stable note ID"):
        player.validate_payload(data)
    data = payload()
    data["works"][0]["presets"][0]["ids"] = ["missing"]
    with pytest.raises(ValueError, match="Preset"):
        player.validate_payload(data)


def test_duplicate_note_identity_fails():
    data = payload()
    data["works"][0]["notes"] *= 2
    with pytest.raises(ValueError, match="Note IDs"):
        player.validate_payload(data)


def test_control_has_no_interactive_selection_and_bulk_skips_deletion():
    node = shutil.which("node")
    if not node:
        bundled = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node"
        if bundled.exists():
            node = str(bundled)
    if not node:
        pytest.skip("Node runtime unavailable for JavaScript behavior check")
    # Execute the actual pure browser helpers with synthetic presets/events.
    names = ("presetSelectable", "bulkSelectionIds", "doseCounts", "presetLabel")
    functions = [re.search(r"function " + name + r"\([^\n]+\}\n", player.HTML).group(0) for name in names]
    source = "const labels={harmony:'Гармония',rhythm:'Ритмика',melody:'Мелодия'};\n" + "\n".join(functions)
    source += "\nconsole.log(JSON.stringify([presetSelectable({compatibility:'control',ids:[]}),presetSelectable({compatibility:'original',ids:[]}),bulkSelectionIds([{id:'H1',aspect:'harmony'},{id:'D1',aspect:'deletion'}]),doseCounts(1),doseCounts(4),presetLabel('harmony: 5 места')]));"
    result = subprocess.run([node, "-e", source], check=True, text=True, capture_output=True)
    assert json.loads(result.stdout) == [False, True, ["H1"], [1], [1, 3, 4], "Гармония: 5 мест"]
    assert "if(presetSelectable(p))" in player.HTML
    assert "Контроль доступен в WAV/MIDI" in player.HTML


def _node():
    node = shutil.which("node")
    bundled = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node"
    if not node and bundled.exists():
        node = str(bundled)
    if not node:
        pytest.skip("Node runtime unavailable for JavaScript behavior check")
    return node


DOM_HARNESS = r'''
const vm=require('node:vm'), assert=require('node:assert/strict');
class Element {
 constructor(tag){this.tag=tag;this.children=[];this.dataset={};this.value='';this.textContent='';this.disabled=false;this.src='';this.paused=true;}
 append(...children){const first=this.tag==='select'&&this.children.length===0;this.children.push(...children);if(first&&children.length)this.value=children[0].value;}
 replaceChildren(...children){this.children=[];if(this.tag==='select')this.value='';this.append(...children);}
 removeAttribute(name){this[name]='';}
 setAttribute(name,value){this[name]=value;}
 scrollIntoView(){}
 load(){this.loads=(this.loads||0)+1;}
 pause(){this.paused=true;}
 async play(){this.paused=false;this.onplay?.();}
 getContext(){return new Proxy({}, {get:()=>()=>{},set:()=>true});}
}
const nodes={},handlers={},buffers=[],sources=[],gainNodes=[],frames=new Map();let frameId=0;
for(const match of html.matchAll(/id="([^"]+)"/g))nodes[match[1]]=new Element(['workSelect','genreSelect','compareFocus','compareContext'].includes(match[1])?'select':'div');
nodes.payload.textContent=JSON.stringify(payload);
nodes.pianoRoll.width=1100;nodes.pianoRoll.height=180;
function walk(node){return [node,...node.children.flatMap(walk)];}
let releaseResume=null,holdResume=false;
class AudioContext {
 constructor(){this.currentTime=0;}
 resume(){return holdResume?new Promise(resolve=>{releaseResume=resolve;}):Promise.resolve();}
 createBuffer(channels,frames,sr){const buffer={sampleRate:sr,samples:new Float32Array(frames),getChannelData(){return this.samples;}};buffers.push(buffer);return buffer;}
 createBufferSource(){const source={connect(){},disconnect(){this.disconnected=true;},start(...args){this.started=true;this.startArgs=args;},stop(){this.stopped=true;}};sources.push(source);return source;}
 createGain(){const node={connect(){},disconnect(){this.disconnected=true;},gain:{value:1,events:[],cancelScheduledValues(t){this.events.push(['cancel',t]);},setValueAtTime(v,t){this.value=v;this.events.push(['set',v,t]);},linearRampToValueAtTime(v,t){this.value=v;this.events.push(['ramp',v,t]);}}};gainNodes.push(node);return node;}
}
const sandbox={console,setTimeout,TextEncoder,atob:value=>Buffer.from(value,'base64').toString('binary'),requestAnimationFrame:fn=>{frames.set(++frameId,fn);return frameId;},cancelAnimationFrame:id=>frames.delete(id),window:{AudioContext,addEventListener:(type,fn)=>handlers[type]=fn},document:{
 getElementById:id=>nodes[id],createElement:tag=>new Element(tag),
 querySelectorAll:()=>Object.values(nodes).flatMap(walk).filter(n=>n.dataset.edit)
}};
vm.createContext(sandbox);
vm.runInContext(html.match(/<\/script><script>([\s\S]*?)<\/script>/)[1],sandbox);
function evaluate(expression){return vm.runInContext(expression,sandbox);}
function links(){return walk(nodes.presets).filter(n=>n.tag==='a').map(n=>n.href);}
function listen(){return walk(nodes.presets).find(n=>n.tag==='button'&&n.textContent==='Слушать').onclick();}
async function scenario(){
__SCENARIO__
}
scenario().then(()=>console.log('passed')).catch(error=>{console.error(error);process.exitCode=1;});
'''


def run_dom_scenario(tmp_path, scenario, *, data=None, fixtures=None):
    from copy import deepcopy
    if data is None:
        data = payload()
        first = data["works"][0]
        for i in (2, 3):
            work = deepcopy(first)
            work.update(id=f"w{i}", title=f"Clip {i}", source_id=f"source{i}",
                        original_wav=f"w{i}/original.wav", original_midi=f"w{i}/original.mid")
            work["notes"][0]["pitch"] = 60 + i * 5
            work["interventions"][0]["patches"][0]["pitch"] = 61 + i * 5
            work["presets"][0]["file_stem"] = f"w{i}/h1"
            data["works"].append(work)
    out = tmp_path / "player.html"
    player.build_player(data, out)
    script = "const html=" + json.dumps(out.read_text()) + ";\nconst payload=" + json.dumps(data) + ";\n"
    script += "const fixtures=" + json.dumps(fixtures) + ";\n"
    script += DOM_HARNESS.replace("__SCENARIO__", scenario)
    # PCM and reference waveforms can exceed a platform's single-argument limit.
    script_path = tmp_path / "player-scenario.js"
    script_path.write_text(script)
    result = subprocess.run([_node(), str(script_path)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "passed"


@pytest.mark.parametrize("event", ["onchange", "oninput"])
def test_switch_updates_original_presets_and_edited_notes(tmp_path, event):
    run_dom_scenario(tmp_path, """
for(const i of [2,3,1]){
 nodes.workSelect.value='w'+i;nodes.workSelect.EVENT();
 assert.equal(nodes.originalAudio.src,'w'+i+'/original.wav');
 assert.ok(links().every(path=>path.startsWith('w'+i+'/')));
 evaluate("selected.add('h1');refresh()");
 assert.equal(evaluate('applyEdits(selectedItems())[0].pitch'),i===1?61:61+i*5);
 await listen();assert.equal(nodes.presetAudio.src,'w'+i+'/h1.wav');
 assert.equal(nodes.presetAudio.paused,false);
 assert.ok(nodes.presetTitle.textContent.startsWith(payload.works[i-1].title));
}
""".replace("EVENT", event))


def test_restored_dropdown_is_reconciled_on_pageshow(tmp_path):
    run_dom_scenario(tmp_path, """
// Browser history restores form value without firing change/input.
nodes.workSelect.value='w3';handlers.pageshow();
assert.equal(nodes.originalAudio.src,'w3/original.wav');
assert.ok(links().every(path=>path.startsWith('w3/')));
await listen();assert.equal(nodes.presetAudio.src,'w3/h1.wav');
""")


def test_switch_stops_both_media_and_clears_old_preset(tmp_path):
    run_dom_scenario(tmp_path, """
await listen();assert.equal(nodes.presetAudio.paused,false);
nodes.workSelect.value='w2';nodes.workSelect.onchange();
assert.equal(nodes.presetAudio.paused,true);assert.equal(nodes.presetAudio.src,'');
assert.equal(nodes.presetAudio.hidden,true);assert.equal(evaluate('selected.size'),0);
assert.equal(nodes.originalAudio.paused,true);
await nodes.originalAudio.play();await listen();assert.equal(nodes.originalAudio.paused,true);
await nodes.originalAudio.play();assert.equal(nodes.presetAudio.paused,true);
""")


def test_switch_during_async_synthesis_cancels_old_work(tmp_path):
    run_dom_scenario(tmp_path, """
holdResume=true;
const pending=evaluate('play(false)');
nodes.workSelect.value='w2';nodes.workSelect.onchange();
releaseResume();await pending;
assert.equal(buffers.length,0);assert.equal(sources.length,0);
holdResume=false;evaluate("selected.add('h1');refresh()");
await evaluate('play(false)');
assert.equal(sources.length,1);assert.equal(sources[0].started,true);
assert.ok(nodes.renderStatus.textContent.startsWith('Clip 2'));
nodes.workSelect.value='w3';nodes.workSelect.onchange();
assert.equal(sources[0].stopped,true);
""")


def test_ab_switch_preserves_synchronized_sources_and_audio_position(tmp_path):
    run_dom_scenario(tmp_path, """
await nodes.compareStart.onclick();
assert.equal(sources.length,2);assert.equal(buffers.length,2);
assert.equal(sources[0].loop,true);assert.equal(sources[1].loop,true);
assert.deepEqual(sources[0].startArgs,sources[1].startArgs);
assert.equal(sources[0].loopStart,sources[1].loopStart);
assert.equal(sources[0].loopEnd,sources[1].loopEnd);
assert.ok(buffers[0].samples.some((x,i)=>Math.abs(x-buffers[1].samples[i])>1e-4));
const start=evaluate('abSession.startedAt');evaluate('context.currentTime=.25');
nodes.compareB.onclick();
assert.equal(sources.length,2);assert.equal(evaluate('abSession.startedAt'),start);
assert.equal(gainNodes[0].gain.value,0);assert.equal(gainNodes[1].gain.value,1);
assert.equal(nodes.compareB['aria-pressed'],'true');assert.match(nodes.abStatus.textContent,/B — одна правка/);
nodes.compareA.onclick();assert.equal(gainNodes[0].gain.value,1);assert.equal(gainNodes[1].gain.value,0);
nodes.compareRestart.onclick();assert.equal(buffers.length,2);assert.equal(sources.length,4);
assert.equal(sources[0].stopped,true);assert.equal(sources[1].stopped,true);
assert.deepEqual(sources[2].startArgs,sources[3].startArgs);
assert.ok(sources[2].startArgs[0]>start);
nodes.compareStop.onclick();assert.equal(sources[2].stopped,true);assert.equal(sources[3].stopped,true);
assert.equal(evaluate('abSession'),null);assert.equal(frames.size,0);
""")


def test_ab_only_applies_focused_edit_and_cancels_on_track_switch(tmp_path):
    run_dom_scenario(tmp_path, """
evaluate("work.interventions.push({id:'other',aspect:'harmony',title:'Other',patches:[{note_id:'n1',pitch:100}]});selected.add('other');");
await nodes.compareStart.onclick();
assert.equal(sources.length,2);assert.equal(evaluate('abSession.itemId'),'h1');
assert.ok(nodes.abDescription.textContent.includes('Synthetic clip'));
nodes.workSelect.value='w2';nodes.workSelect.onchange();
assert.equal(sources[0].stopped,true);assert.equal(sources[1].stopped,true);
assert.equal(evaluate('abSession'),null);assert.equal(frames.size,0);
assert.ok(nodes.abDescription.textContent.includes('Clip 2'));
holdResume=true;const pending=nodes.compareStart.onclick();
nodes.workSelect.value='w3';nodes.workSelect.onchange();releaseResume();await pending;
assert.equal(sources.length,2);assert.equal(evaluate('abSession'),null);
""")


def test_ab_window_covers_original_and_shifted_note_with_clipped_context(tmp_path):
    run_dom_scenario(tmp_path, """
const check=(work,item,margin)=>JSON.parse(evaluate('JSON.stringify(comparisonWindow('+JSON.stringify(work)+','+JSON.stringify(item)+','+margin+'))'));
const w={duration_seconds:10,seconds_per_quarter:.5,notes:[{id:'n',onset_seconds:3,duration_seconds:2}]};
assert.deepEqual(check(w,{patches:[{note_id:'n',onset_qn:[13,2]}]},2),{start:1,end:7.25,changedStart:3,changedEnd:5.25});
assert.deepEqual(check(w,{patches:[{note_id:'n',onset_qn:[1,2],duration_qn:[20,1]}]},2),{start:0,end:10,changedStart:.25,changedEnd:10});
assert.deepEqual(check(w,{patches:[{note_id:'n',delete:true}]},1),{start:2,end:6,changedStart:3,changedEnd:5});
""")


def test_ab_uses_full_sustained_notes_and_identical_edge_fades(tmp_path):
    run_dom_scenario(tmp_path, """
evaluate("work.duration_seconds=10;work.notes[0].onset_seconds=3;work.notes[0].duration_seconds=1;work.notes.push({id:'sustain',pitch:48,onset_seconds:0,duration_seconds:8});");
nodes.compareContext.value='1';await nodes.compareStart.onclick();
assert.equal(sources[0].loopStart,2);assert.equal(sources[0].loopEnd,5);
assert.equal(buffers[0].samples.length,220500);
// Sustained original context retains its phase; only the target note differs.
const a=buffers[0].samples,b=buffers[1].samples,sr=22050;
assert.ok(a.slice(2.1*sr,2.2*sr).some(x=>Math.abs(x)>1e-5));
assert.deepEqual(a.slice(2.1*sr,2.2*sr),b.slice(2.1*sr,2.2*sr));
assert.equal(Math.abs(a[2*sr]),0);assert.equal(Math.abs(b[2*sr]),0);
assert.equal(Math.abs(a[5*sr-1]),0);assert.equal(Math.abs(b[5*sr-1]),0);
await nodes.originalAudio.play();assert.equal(evaluate('abSession'),null);
assert.ok(sources.every(s=>s.stopped));
""")


def test_harmonic_alternatives_allow_only_one_scenario(tmp_path):
    run_dom_scenario(tmp_path, """
evaluate("data.alternatives_only=true;work.interventions.push({...work.interventions[0],id:'h2'});loadWork();");
assert.equal(nodes.selectAll.hidden,true);
const inputs=walk(nodes.groups).filter(n=>n.tag==='input');
assert.equal(inputs.length,2);assert.ok(inputs.every(n=>n.type==='radio'));
inputs[0].checked=true;inputs[0].onchange();assert.equal(evaluate('selected.size'),1);
inputs[1].checked=true;inputs[1].onchange();assert.equal(evaluate('selected.size'),1);
assert.equal(evaluate("selected.has('h1')"),false);assert.equal(evaluate("selected.has('h2')"),true);
assert.ok(!walk(nodes.groups).some(n=>n.tag==='button'&&n.textContent.startsWith('Первые')));
""")


def genre_payload():
    """Small multitimbral fixtures with genuinely different source PCM bytes."""
    import base64
    from copy import deepcopy
    import numpy as np

    data = dict(version="synthetic-genres", renderer="DiagnosticGMFamilies@1.0.0", preview_gain=.02, works=[])
    for index, (genre, program, pitch) in enumerate((("classical", 0, 60), ("rock", 25, 52), ("electronic", 81, 72), ("hiphop", 33, 40), ("rock", 29, 57)), 1):
        work = deepcopy(payload()["works"][0])
        work.update(id=f"w{index}", title=f"Genre clip {index}", source_id=f"genre-fixture-{index}", genre=genre, duration_seconds=.2, background_gain=.15 + .025 * index, original_wav=f"w{index}/original.wav", original_midi=f"w{index}/original.mid")
        work["notes"][0].update(id=f"note:{index}", pitch=pitch, onset_qn=[1, 10], duration_qn=[3, 25], onset_seconds=.05, duration_seconds=.06, velocity=73, program=program, is_percussion=False)
        work["interventions"][0].update(location_seconds=.05, patches=[dict(note_id=f"note:{index}", pitch=pitch + 2)], context_note_ids=[f"note:{index}"])
        work["presets"][0]["file_stem"] = f"w{index}/h1"
        work["notes"].append(dict(id=f"drum:{index}", pitch=(36, 38, 42, 46, 45)[index - 1], onset_qn=[1, 50], duration_qn=[1, 50], onset_seconds=.01, duration_seconds=.01, velocity=85, program=0, is_percussion=True))
        pcm = (((np.arange(4410) * (123 + index * 71)) % 16001) - 8000).astype("<i2")
        work["background_pcm16"] = base64.b64encode(pcm.tobytes()).decode("ascii")
        if index == 2:
            work["preview_gain"] = .035
        data["works"].append(work)
    return data


def genre_expected(data):
    """Python reference independently sums signed PCM and applies one end fade."""
    import base64
    import numpy as np

    spec = importlib.util.spec_from_file_location("genre_player_reference", PATH.with_name("render_genre_preview.py"))
    synth = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(synth)
    references = {}
    for work in data["works"]:
        count = round(work["duration_seconds"] * 22050)
        gain = work.get("preview_gain", data["preview_gain"])
        background = np.frombuffer(base64.b64decode(work["background_pcm16"]), dtype="<i2").astype(np.float64) / 32768
        fade = np.ones(count)
        fade[-441:] = np.linspace(1, 0, 441)
        reference = {}
        for version in ("original", "edited", "background"):
            events = [dict(n) for n in work["notes"]]
            if version == "edited":
                events[0]["pitch"] += 2
            if version == "background":
                events = []
            # Longer render keeps its own clip-end fade outside this crop.
            notes, _ = synth.synthesize_preview(events, duration_seconds=.4, gain=gain)
            reference[version] = ((notes[:count] + background * work["background_gain"]) * fade).tolist()
        reference["gain"] = gain
        references[work["id"]] = reference
    return references


JS_ASSERT_WAVE = r'''
function assertWave(actual,expected,label){
 assert.equal(actual.length,expected.length,label+' length');
 let maximumError=0;
 for(let i=0;i<actual.length;i++)maximumError=Math.max(maximumError,Math.abs(actual[i]-expected[i]));
 assert.ok(maximumError<2e-8,label+' maximum error='+maximumError);
}
'''


def test_genre_renderer_matches_python_pcm_background_and_common_gain(tmp_path):
    data = genre_payload()
    run_dom_scenario(tmp_path, JS_ASSERT_WAVE + r'''
evaluate('context=new window.AudioContext()');
for(const source of payload.works){
 nodes.workSelect.value=source.id;nodes.workSelect.onchange();
 const original=await evaluate('renderBuffer(work,work.notes.map(n=>({...n})),generation)');
 const edited=await evaluate('renderBuffer(work,applyEdits([work.interventions[0]]),generation)');
 const background=await evaluate('renderBuffer(work,[],generation)');
 assertWave(original.buffer.samples,fixtures[source.id].original,source.id+' original');
 assertWave(edited.buffer.samples,fixtures[source.id].edited,source.id+' edited');
 assertWave(background.buffer.samples,fixtures[source.id].background,source.id+' background');
 assert.equal(original.gain,fixtures[source.id].gain);
 assert.equal(edited.gain,original.gain);
 assert.equal(background.gain,original.gain);
 assert.ok(original.buffer.samples.some((x,i)=>Math.abs(x-edited.buffer.samples[i])>1e-4));
 // Before the changed note, both versions contain exactly the same drums/background.
 assert.deepEqual(original.buffer.samples.slice(0,1000),edited.buffer.samples.slice(0,1000));
 assert.ok(background.buffer.samples.some(x=>x<0));
 assert.ok(background.buffer.samples.some(x=>x>0));
 assert.equal(Math.abs(original.buffer.samples.at(-1)),0);
}
assert.equal(evaluate('backgroundCache.size'),5);
''', data=data, fixtures=genre_expected(data))


def test_genre_filter_switches_notes_media_and_stops_old_sources(tmp_path):
    data = genre_payload()
    run_dom_scenario(tmp_path, JS_ASSERT_WAVE + r'''
assert.equal(nodes.genreLabel.hidden,false);
assert.equal(nodes.genreSelect.value,'');assert.equal(nodes.workSelect.children.length,5);
assert.ok(nodes.genreSelect.children.some(option=>option.value==='hiphop'));
assert.equal(evaluate('work.id'),'w1');
await listen();assert.equal(nodes.presetAudio.paused,false);
evaluate("selected.add('h1');refresh()");await evaluate('play(false)');
const firstSource=sources.at(-1);
nodes.genreSelect.value='rock';nodes.genreSelect.onchange();
assert.equal(evaluate('work.id'),'w2');assert.equal(nodes.workSelect.value,'w2');
assert.deepEqual(nodes.workSelect.children.map(option=>option.value),['w2','w5']);
assert.equal(firstSource.stopped,true);assert.equal(nodes.presetAudio.paused,true);
assert.equal(nodes.presetAudio.src,'');assert.equal(evaluate('selected.size'),0);
assert.equal(nodes.originalAudio.src,'w2/original.wav');assert.ok(links().every(path=>path.startsWith('w2/')));
assert.equal(evaluate('work.notes[0].pitch'),52);
await evaluate('play(true)');assertWave(buffers.at(-1).samples,fixtures.w2.original,'rock filtered original');
nodes.workSelect.value='w5';nodes.workSelect.onchange();await evaluate('play(true)');
assertWave(buffers.at(-1).samples,fixtures.w5.original,'other rock original');
nodes.genreSelect.value='hiphop';nodes.genreSelect.onchange();
assert.equal(evaluate('work.id'),'w4');assert.equal(evaluate('work.notes[0].pitch'),40);
assert.deepEqual(nodes.workSelect.children.map(option=>option.value),['w4']);
await evaluate('play(true)');assertWave(buffers.at(-1).samples,fixtures.w4.original,'hiphop filtered original');
nodes.genreSelect.value='';nodes.genreSelect.onchange();
assert.equal(evaluate('work.id'),'w4');assert.equal(nodes.workSelect.children.length,5);
''', data=data, fixtures=genre_expected(data))


def test_genre_switch_during_pending_ab_cannot_start_previous_source(tmp_path):
    data = genre_payload()
    run_dom_scenario(tmp_path, JS_ASSERT_WAVE + r'''
await nodes.compareStart.onclick();assert.equal(sources.length,2);
nodes.genreSelect.value='electronic';nodes.genreSelect.onchange();
assert.ok(sources.every(source=>source.stopped));assert.equal(evaluate('abSession'),null);
holdResume=true;const pending=nodes.compareStart.onclick();
nodes.genreSelect.value='hiphop';nodes.genreSelect.onchange();
releaseResume();await pending;assert.equal(sources.length,2);
assert.equal(evaluate('work.id'),'w4');assert.equal(evaluate('abSession'),null);
holdResume=false;await evaluate('play(true)');
assert.equal(sources.length,3);assertWave(buffers.at(-1).samples,fixtures.w4.original,'new source after cancelled AB');
''', data=data, fixtures=genre_expected(data))


def test_background_cache_is_scoped_by_work_and_revisit_preserves_audio(tmp_path):
    data = genre_payload()
    run_dom_scenario(tmp_path, JS_ASSERT_WAVE + r'''
evaluate('context=new window.AudioContext()');
for(const id of ['w1','w2','w4','w1','w5','w2']){
 nodes.workSelect.value=id;nodes.workSelect.onchange();
 const rendered=await evaluate('renderBuffer(work,[],generation)');
 assertWave(rendered.buffer.samples,fixtures[id].background,id+' cached background');
}
assert.equal(evaluate('backgroundCache.size'),4);
''', data=data, fixtures=genre_expected(data))
