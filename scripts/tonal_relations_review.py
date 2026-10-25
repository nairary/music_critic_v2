"""Validate a research-only analysis/event sidecar; never read source corpora.

Exact source times are rational strings. This checks references and provenance
fields, not the truth of an analyst's interpretation or dataset admission.
The sidecar is not a training TargetBundle and provides no tension score.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import html
import json
from pathlib import Path, PurePosixPath
import re
from urllib.parse import urlparse


class ReviewError(ValueError):
    """Invalid research sidecar."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReviewError(message)


def text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def fraction(value: object) -> Fraction:
    require(isinstance(value, str) and re.fullmatch(r"-?\d+/[1-9]\d*", value) is not None,
            "time must be an exact rational string")
    result = Fraction(value)
    require(value == f"{result.numerator}/{result.denominator}", "noncanonical rational")
    return result


def string_list(value: object, name: str) -> None:
    require(isinstance(value, list) and all(text(v) for v in value), f"invalid {name}")


def validate(data: dict) -> dict:
    require(isinstance(data, dict) and data.get("schema") == "tonal-relations-review@1", "invalid schema")
    require(text(data.get("title")), "title required")
    require(isinstance(data.get("cases"), list) and bool(data["cases"]), "cases required")
    cases, claims, events = set(), Counter(), 0
    for case in data["cases"]:
        require(isinstance(case, dict), "invalid case")
        require(text(case.get("id")) and case["id"] not in cases, "duplicate/empty case id")
        cases.add(case["id"])
        require(text(case.get("title")), "case title required")
        string_list(case.get("genres"), "genres")
        require(bool(case["genres"]), "genre required")
        require(case.get("membership") in {"train", "external_research_only", "excluded"}, "forbidden membership")
        require(case.get("time_unit") in {"quarter_note", "measured_seconds"}, "unknown time unit")
        require(text(case.get("time_origin")), "time origin required")
        string_list(case.get("limitations"), "limitations")
        require(isinstance(case.get("assets"), list), "assets required")
        for asset in case["assets"]:
            require(isinstance(asset, dict), "invalid asset")
            locator = asset.get("locator")
            require(text(locator) and not PurePosixPath(locator).is_absolute()
                    and ".." not in PurePosixPath(locator).parts
                    and ":" not in locator and "\\" not in locator, "relative asset locator required")
            require(isinstance(asset.get("sha256"), str)
                    and re.fullmatch(r"[0-9a-f]{64}", asset["sha256"]) is not None, "asset hash required")
        require(isinstance(case.get("events"), list), "events required")
        ids = set()
        for event in case["events"]:
            require(isinstance(event, dict), "invalid event")
            require(text(event.get("id")) and event["id"] not in ids, "duplicate/empty event id")
            ids.add(event["id"])
            fraction(event.get("start"))  # Negative pickups are allowed and keep their source origin.
            require(fraction(event.get("duration")) >= 0, "negative duration")
            require(type(event.get("pitch")) is int and 0 <= event["pitch"] <= 127, "invalid pitch")
            require(text(event.get("voice")) and text(event.get("source_ref")), "event provenance required")
        events += len(ids)
        require(case["membership"] != "excluded" or not ids, "excluded case has musical events")
        require(not ids or bool(case["assets"]), "events require source assets")
        require(isinstance(case.get("claims"), list) and bool(case["claims"]), "claims required")
        claim_ids = set()
        for claim in case["claims"]:
            require(isinstance(claim, dict), "invalid claim")
            require(text(claim.get("id")) and claim["id"] not in claim_ids, "duplicate/empty claim id")
            claim_ids.add(claim["id"])
            for field in ("source_anchor", "statement", "relation", "assessment"):
                require(text(claim.get(field)), f"claim {field} required")
            require(text(claim.get("source_url")), "source URL required")
            url = urlparse(claim["source_url"])
            require(url.scheme in {"https", "http"} and bool(url.netloc), "unsafe source URL")
            require(claim.get("status") in {"verified", "partial", "rejected"}, "unknown claim status")
            string_list(claim.get("alternatives"), "alternatives")
            string_list(claim.get("missing"), "missing")
            for field in ("context_event_ids", "source_window_event_ids"):
                if field in claim:
                    string_list(claim[field], field)
                    require(len(claim[field]) == len(set(claim[field]))
                            and set(claim[field]) <= ids, "invalid context event reference")
            require(isinstance(claim.get("participants"), list), "participants required")
            references = set()
            for participant in claim["participants"]:
                require(isinstance(participant, dict) and text(participant.get("role")), "participant role required")
                refs = participant.get("event_ids")
                string_list(refs, "event references")
                require(bool(refs) and len(refs) == len(set(refs)), "empty/duplicate event references")
                require(set(refs) <= ids, "dangling event reference")
                references.update(refs)
            require(claim["status"] != "verified" or bool(references), "verified claim needs event evidence")
            require(case["membership"] != "excluded" or claim["status"] == "rejected", "excluded claim must be rejected")
            require(claim["status"] == "verified" or bool(claim["missing"]), "incomplete claim needs missing reason")
            claims[claim["status"]] += 1
    return {"cases": len(cases), "events": events, "claim_statuses": dict(sorted(claims.items()))}


_VIEW = r'''<!doctype html>
<html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{font:16px/1.55 system-ui,sans-serif;color:#182735;background:#f5f4ef}body{margin:0}
header,main{max-width:1300px;margin:auto;padding:24px}h1{font-size:30px;line-height:1.15;margin:8px 0 14px}
.eyebrow{font:12px ui-monospace,monospace;letter-spacing:.12em;color:#586673}p{max-width:960px}
select,button{font:inherit;border:1px solid #b9c4cb;border-radius:7px;background:white;padding:8px 12px}
select{max-width:100%;width:680px}button{cursor:pointer;text-align:left}button.active{border:2px solid #146b8d}
#layout{display:grid;grid-template-columns:340px minmax(0,1fr);gap:24px}#claims{display:flex;flex-direction:column;gap:10px}
.panel{padding:20px;background:#fff;border:1px solid #d9dedf;border-radius:10px;min-width:0}
.muted{color:#596b78;font-size:14px}.badge{font:12px ui-monospace,monospace;color:#496273}
#plot{overflow:auto;border-top:1px solid #d9dedf;margin-top:16px}svg{display:block;min-width:600px;width:100%}
#assessment{border-left:3px solid #146b8d;padding-left:14px}a{color:#126b8d}li{margin:6px 0}
@media(max-width:850px){#layout{grid-template-columns:1fr}header,main{padding:16px}}
</style>
<header><div class="eyebrow">MUSIC CRITIC · T02b · ИССЛЕДОВАТЕЛЬСКИЙ ПИЛОТ</div>
<h1>__TITLE__</h1><p>Авторский разбор → исходные ноты → проверенная привязка.
Это просмотр свидетельств, не прогноз модели и не оценка качества музыки.
Синий цвет выделяет ноты выбранного утверждения; он не кодирует силу напряжения.</p>
<select id="case" aria-label="Произведение"></select><p id="meta" class="muted"></p></header>
<main><div id="layout"><aside id="claims"></aside><section class="panel">
<div class="badge" id="status"></div><h2 id="statement"></h2><p><a id="source" target="_blank" rel="noopener noreferrer"></a></p>
<p id="assessment"></p><div id="plot"></div><p id="axis" class="muted"></p>
<h3>Границы и альтернативы</h3><ul id="limits"></ul>
<details><summary>Участники и исходные IDs</summary><pre id="ids" style="white-space:pre-wrap;overflow-wrap:anywhere"></pre></details>
</section></div></main>
<script id="evidence" type="application/json">__DATA__</script>
<script>
"use strict";
const data=JSON.parse(document.getElementById('evidence').textContent), byId=id=>document.getElementById(id);
const frac=s=>{const a=s.split('/').map(Number);return a[0]/a[1]};
const status={verified:'Привязка проверена',partial:'Частичная привязка',rejected:'Привязка отклонена'};
const membership={train:'TRAIN',external_research_only:'Внешний исследовательский пример · вне обучения',excluded:'Исключён до чтения нот'};
const ns='http://www.w3.org/2000/svg';
function svgNode(name,attrs={},value){const n=document.createElementNS(ns,name);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v);if(value!==undefined)n.textContent=value;return n}
data.cases.forEach((c,i)=>{const o=document.createElement('option');o.value=i;o.textContent=c.genres.join(' / ')+' · '+c.title;byId('case').append(o)});
function selectCase(){const c=data.cases[Number(byId('case').value)];
 byId('meta').textContent=membership[c.membership]+' · '+c.events.length+' нотных событий в сохранённом фрагменте';
 byId('claims').replaceChildren();c.claims.forEach((cl,i)=>{const b=document.createElement('button');b.textContent=status[cl.status]+' — '+cl.statement;b.onclick=()=>selectClaim(c,i);byId('claims').append(b)});selectClaim(c,0)}
function selectClaim(c,i){const cl=c.claims[i];Array.from(byId('claims').children).forEach((b,j)=>b.classList.toggle('active',j===i));
 byId('status').textContent=status[cl.status]+' · '+cl.relation;byId('statement').textContent=cl.statement;
 byId('source').textContent=cl.source_anchor;byId('source').href=cl.source_url;byId('assessment').textContent=cl.assessment;
 byId('limits').replaceChildren();[...cl.missing,...cl.alternatives,...c.limitations].forEach(t=>{const li=document.createElement('li');li.textContent=t;byId('limits').append(li)});
 byId('ids').textContent=cl.participants.map(p=>p.role+': '+p.event_ids.join(', ')).join('\n')||'Нотные участники не установлены';
 const selected=new Set(cl.participants.flatMap(p=>p.event_ids));
 const contextIds=new Set([...selected,...(cl.context_event_ids||cl.source_window_event_ids||[])]);
 const context=c.events.filter(e=>contextIds.has(e.id));
 const bounds=context.length?context:c.events;
 byId('plot').replaceChildren();byId('axis').textContent='';if(!bounds.length){byId('plot').textContent='Музыкальные события не открывались или соответствие не найдено.';return}
 let lo=Math.min(...bounds.map(e=>frac(e.start))),hi=Math.max(...bounds.map(e=>frac(e.start)+frac(e.duration)));if(hi===lo)hi=lo+1;
 const pad=(hi-lo)*.03,notes=c.events.filter(e=>frac(e.start)<=hi+pad&&frac(e.start)+frac(e.duration)>=lo-pad);
 lo-=pad;hi+=pad;let p0=Math.min(...notes.map(e=>e.pitch))-2,p1=Math.max(...notes.map(e=>e.pitch))+2;
 const w=900,h=320,x=t=>50+(t-lo)/(hi-lo)*(w-70),y=p=>h-30-(p-p0)/(p1-p0)*(h-50);
 const svg=svgNode('svg',{viewBox:`0 0 ${w} ${h}`,role:'img','aria-label':'Нотные события выбранного участка'});
 for(let p=Math.ceil(p0/12)*12;p<=p1;p+=12){svg.append(svgNode('line',{x1:50,x2:w-20,y1:y(p),y2:y(p),stroke:'#e4e8e9'}));svg.append(svgNode('text',{x:8,y:y(p)+4,fill:'#617480','font-size':11},'C'+(p/12-1)))}
 for(let t=0;t<=5;t++){const v=lo+(hi-lo)*t/5;svg.append(svgNode('text',{x:x(v),y:h-8,fill:'#617480','font-size':11},v.toFixed(2)))}
 for(const e of notes){const rect=svgNode('rect',{x:x(frac(e.start)),y:y(e.pitch)-3,width:Math.max(2,x(frac(e.start)+frac(e.duration))-x(frac(e.start))),height:6,rx:2,fill:selected.has(e.id)?'#146b8d':'#cbd6dc'});rect.append(svgNode('title',{},`${e.id} · MIDI ${e.pitch} · ${e.voice}\n${e.start} + ${e.duration} · ${e.source_ref}`));svg.append(rect)}
 byId('plot').append(svg);byId('axis').textContent=(c.time_unit==='quarter_note'?'Ось X: четверти':'Ось X: измеренные секунды')+' · '+c.time_origin+'. Точные значения доступны при наведении.';
}
byId('case').onchange=selectCase;selectCase();
</script></html>'''


def render_html(data: dict) -> str:
    validate(data)
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return _VIEW.replace("__TITLE__", html.escape(data["title"])).replace("__DATA__", payload)


def write_html_once(data: dict, output: Path) -> None:
    rendered = render_html(data)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(rendered)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sidecar", type=Path)
    parser.add_argument("--html", type=Path, help="Create a new standalone review page (never overwrite)")
    args = parser.parse_args()
    data = json.loads(args.sidecar.read_text())
    result = validate(data)
    if args.html:
        write_html_once(data, args.html)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
