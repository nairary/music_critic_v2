"""Inspect research evidence across musical levels without estimating quality.

The map is not a TargetBundle, neural prediction or training route. It consumes
only a supplied sidecar, never a corpus. Source times retain exact rational units.
"""
from __future__ import annotations

import argparse
from collections import Counter
import html
import json
from pathlib import Path

if __package__:
    from . import tonal_relations_review as v1
else:  # Direct CLI execution from scripts/.
    import tonal_relations_review as v1


ReviewError = v1.ReviewError
OBJECT_KINDS = ("sonority", "sequence", "phrase", "section", "part", "piece")
LEVELS = ("note",) + OBJECT_KINDS
RELATION_LEVELS = set(LEVELS) | {"cross_level"}


def references(value: object, available: set[str], label: str, *, nonempty: bool = False) -> None:
    v1.string_list(value, label)
    v1.require(not nonempty or bool(value), f"empty {label}")
    v1.require(len(value) == len(set(value)) and set(value) <= available,
               f"duplicate or dangling {label}")


def validate(data: dict) -> dict:
    """Check sidecar mechanics, not musical truth or admission provenance."""
    v1.require(isinstance(data, dict) and data.get("schema") == "tonal-relations-map@1",
               "invalid relation-map schema")
    summary = v1.validate({**data, "schema": "tonal-relations-review@1"})
    summary.update(objects=0, relations=0)
    for case in data["cases"]:
        events = {event["id"]: event for event in case["events"]}
        event_ids = set(events)
        v1.require(isinstance(case.get("objects"), list), "objects required")
        objects = {}
        for obj in case["objects"]:
            v1.require(isinstance(obj, dict), "invalid object")
            identifier = obj.get("id")
            v1.require(v1.text(identifier) and identifier not in objects and identifier not in event_ids,
                       "duplicate/empty object id or collision with event id")
            v1.require(obj.get("kind") in OBJECT_KINDS, "unknown object kind")
            v1.require(v1.text(obj.get("label")) and v1.text(obj.get("basis")), "object label/basis required")
            start, end = v1.fraction(obj.get("start")), v1.fraction(obj.get("end"))
            v1.require(start < end, "object span must have positive duration")
            references(obj.get("event_ids"), event_ids, "object event references", nonempty=True)
            for event_id in obj["event_ids"]:
                event = events[event_id]
                onset, duration = v1.fraction(event["start"]), v1.fraction(event["duration"])
                offset = onset + duration
                v1.require(duration > 0 and onset < end and offset > start,
                           "object member must sound inside its half-open span")
                if obj["kind"] == "sonority":
                    v1.require(onset <= start and offset >= end,
                               "sonority member must cover its complete span")
            objects[identifier] = obj
        summary["objects"] += len(objects)
        v1.require(isinstance(case.get("relations"), list), "relations required")
        claim_ids = {claim["id"] for claim in case["claims"]}
        relation_ids = set()
        for relation in case["relations"]:
            v1.require(isinstance(relation, dict), "invalid relation")
            identifier = relation.get("id")
            v1.require(v1.text(identifier) and identifier not in relation_ids,
                       "duplicate/empty relation id")
            relation_ids.add(identifier)
            v1.require(v1.text(relation.get("kind")), "relation kind required")
            v1.require(relation.get("level") in RELATION_LEVELS, "unknown relation level")
            v1.require(relation.get("claim_id") in claim_ids, "dangling claim reference")
            v1.require(relation.get("alignment") in {"exact_event_ids", "partial", "unavailable"},
                       "unknown alignment")
            v1.require("confidence" in relation and relation["confidence"] is None,
                       "confidence must remain null")
            v1.require(relation.get("target_available") is False, "research relation cannot enable targets")
            references(relation.get("context_event_ids"), event_ids, "context event references")
            participants = relation.get("participants")
            v1.require(isinstance(participants, list), "relation participants required")
            v1.require(relation["alignment"] == "unavailable" or bool(participants),
                       "available alignment needs participants")
            for participant in participants:
                v1.require(isinstance(participant, dict) and v1.text(participant.get("role")),
                           "participant role required")
                references(participant.get("event_ids"), event_ids, "participant event references")
                references(participant.get("object_ids"), set(objects), "participant object references")
                v1.require(bool(participant["event_ids"] or participant["object_ids"]),
                           "empty participant references")
        summary["relations"] += len(relation_ids)
        v1.require(case["membership"] != "excluded" or (not objects and not relation_ids),
                   "excluded case cannot contain objects or relations")
        coverage = case.get("coverage")
        v1.require(isinstance(coverage, dict) and set(coverage) == set(LEVELS),
                   "coverage must describe every supported level")
        counts = Counter(obj["kind"] for obj in objects.values())
        counts["note"] = len(events)
        for level in LEVELS:
            item = coverage[level]
            v1.require(isinstance(item, dict) and v1.text(item.get("reason")), "coverage reason required")
            v1.require(item.get("status") in {"observed", "partial", "unavailable"}, "unknown coverage status")
            v1.require(item["status"] != "observed" or counts[level] > 0,
                       "observed coverage needs event/object evidence")
            v1.require(item["status"] != "unavailable" or counts[level] == 0,
                       "unavailable level cannot contain evidence")
            v1.require(case["membership"] != "excluded" or item["status"] == "unavailable",
                       "excluded coverage must remain unavailable")
    return summary


_VIEW = r'''<!doctype html>
<html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{font:15px/1.5 system-ui,sans-serif;color:#20333d;background:#f3f4f0}body{margin:0}header,main{max-width:1440px;margin:auto;padding:24px}
h1{font-size:30px;line-height:1.15;margin:8px 0}h2{font-size:21px}h3{font-size:16px}p{max-width:1050px}.muted{color:#536773;font-size:13px}
select,button{font:inherit;background:white;color:inherit;border:1px solid #bdc9cb;border-radius:6px;padding:8px;max-width:100%}button{cursor:pointer;text-align:left}button.active{border:2px solid #28708a;background:#f2f8f9}
select{width:100%}.panel{background:white;border:1px solid #d6dfe0;border-radius:9px;padding:18px;min-width:0}#layout{display:grid;grid-template-columns:290px minmax(0,1fr);gap:18px}#relations{display:flex;flex-direction:column;gap:7px}#relations button{overflow-wrap:anywhere;min-width:0;width:100%}
#map,#piano{overflow:auto}svg{display:block;width:100%;min-width:720px}#coverage{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:8px}#coverage article{padding:10px;border-left:3px solid #a8b7bd;background:#f4f7f7}#coverage strong{display:block}
#legend{display:flex;flex-wrap:wrap;gap:14px}#legend span{border-bottom:4px solid;padding:3px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.5 ui-monospace,monospace}a{color:#126b8d}li{margin:5px 0}.stack{display:grid;gap:18px}
@media(max-width:850px){#layout{grid-template-columns:1fr}header,main{padding:14px}}
</style>
<header><div class="muted">MUSIC CRITIC · T03 · КАРТА ИСТОЧНИКОВЫХ СВЯЗЕЙ</div>
<h1 id="title"></h1><p>Выберите связь или группу и проверьте её исходные ноты, контекст и основания.
Цвет обозначает роль участника; это не оценка напряжения, качества музыки или прогноз модели.
Точная ссылка на ноту сама по себе не подтверждает музыкальную интерпретацию.</p>
<label for="case">Нотная версия</label><select id="case"></select><p id="meta" class="muted"></p></header>
<main><div id="layout"><aside class="panel"><h2>Связи и утверждения</h2><div id="relations"></div>
<h3><label for="objects">Группы событий</label></h3><select id="objects"></select><p class="muted">Группы могут пересекаться. Ноты и группы — разные объекты; исходный voice не означает установленный инструмент.</p></aside>
<div class="stack"><section class="panel"><div id="status" class="muted"></div><h2 id="statement"></h2>
<p><a id="source" target="_blank" rel="noopener noreferrer"></a></p><p id="assessment"></p>
<label for="scope">Область просмотра</label><select id="scope"><option value="focus">Участок связи</option><option value="context">Весь доступный контекст</option></select><div id="legend"></div>
<div id="map"></div><p class="muted">Линии связывают выбранные объекты с общим отношением и показывают участие, не причинность.</p>
<div id="piano"></div><p id="axis" class="muted"></p><h3>Основания, границы и альтернативы</h3><ul id="limits"></ul>
<details open><summary>Участники, точное время и исходные IDs</summary><pre id="details"></pre></details></section>
<section class="panel"><h2>Доступность уровней</h2><div id="coverage"></div></section></div></div></main>
<script id="evidence" type="application/json">__DATA__</script>
<script>
"use strict";
const data=JSON.parse(document.getElementById('evidence').textContent);
const byId=id=>document.getElementById(id), ns='http://www.w3.org/2000/svg';
const levels=['note','sonority','sequence','phrase','section','part','piece'];
const names={note:'Ноты',sonority:'Созвучия',sequence:'Последовательности',phrase:'Фразы',section:'Разделы',part:'Партии',piece:'Произведение'};
const states={observed:'Наблюдается',partial:'Частично',unavailable:'Нет данных'};
const statuses={verified:'Привязка проверена',partial:'Частичная привязка',rejected:'Привязка отклонена'};
const alignments={exact_event_ids:'Точные исходные IDs',partial:'Частичное соответствие',unavailable:'Соответствие недоступно'};
const memberships={train:'TRAIN',external_research_only:'Внешний исследовательский пример · вне обучения',excluded:'Исключён до чтения нот'};
const palette=['#176c87','#80639d','#b57036','#407c65','#a8516c','#5969a4'];
let currentCase=null,currentSelection=null;
const brief=text=>text.length>95?text.slice(0,92)+'…':text;
const relationLabel=(c,r)=>brief(c.claims.find(cl=>cl.id===r.claim_id).statement);
const frac=s=>{const a=s.split('/').map(Number);return a[0]/a[1]};
const node=(tag,value)=>{const n=document.createElement(tag);if(value!==undefined)n.textContent=value;return n};
function svgNode(tag,attrs={},value){const n=document.createElementNS(ns,tag);for(const [key,val]of Object.entries(attrs))n.setAttribute(key,val);if(value!==undefined)n.textContent=value;return n}
function tooltip(n,text){n.append(svgNode('title',{},text));return n}
function selectable(n,action){n.setAttribute('tabindex','0');n.setAttribute('role','button');n.style.cursor='pointer';n.onclick=action;n.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();action()}};return n}
function evidenceView(c,selection){
 const objects=new Map(c.objects.map(o=>[o.id,o]));let participants=[],context=[];
 if(selection.type==='relation'){participants=selection.item.participants;context=selection.item.context_event_ids}
 if(selection.type==='object'){participants=[{role:'Выбранная группа',event_ids:[],object_ids:[selection.item.id]}];context=c.events.map(e=>e.id)}
 if(selection.type==='claim'){participants=selection.item.participants.map(p=>({...p,object_ids:[]}));context=[...(selection.item.context_event_ids||[]),...(selection.item.source_window_event_ids||[])]}
 const roles=new Map(),objectRoles=new Map(),roleNames=[];
 const add=(map,id,role)=>{if(!map.has(id))map.set(id,[]);if(!map.get(id).includes(role))map.get(id).push(role)};
 for(const p of participants){if(!roleNames.includes(p.role))roleNames.push(p.role);for(const id of p.event_ids)add(roles,id,p.role);for(const id of p.object_ids){add(objectRoles,id,p.role);for(const eventId of objects.get(id).event_ids)add(roles,eventId,p.role)}}
 const wanted=new Set([...roles.keys(),...(byId('scope').value==='context'?context:[])]);let bounds=c.events.filter(e=>wanted.has(e.id));if(!bounds.length)bounds=c.events;
 let lo=bounds.length?Math.min(...bounds.map(e=>frac(e.start))):0,hi=bounds.length?Math.max(...bounds.map(e=>frac(e.start)+frac(e.duration))):1;
 for(const id of objectRoles.keys()){lo=Math.min(lo,frac(objects.get(id).start));hi=Math.max(hi,frac(objects.get(id).end))}
 if(hi===lo)hi=lo+1;const pad=Math.max((hi-lo)*.1,c.time_unit==='quarter_note'?1:.25);lo-=pad;hi+=pad;
 const notes=c.events.filter(e=>frac(e.start)<hi&&frac(e.start)+frac(e.duration)>lo);
 return {participants,roles,objectRoles,roleNames,notes,lo,hi,objects,color:role=>palette[roleNames.indexOf(role)%palette.length]};
}
function detailsFor(c,view,selection){
 const lines=[];if(selection.type==='relation')lines.push('Тип связи в sidecar: '+selection.item.kind,'target_available=false · confidence=null');for(const p of view.participants){lines.push('Роль из источника/sidecar: '+p.role);for(const id of p.object_ids){const o=view.objects.get(id);lines.push('  '+o.id+' · '+names[o.kind]+' · ['+o.start+', '+o.end+') · '+o.label,'  Основание: '+o.basis,'  Участники: '+o.event_ids.join(', '))}for(const id of p.event_ids)lines.push('  Прямой участник: '+id)}
 for(const e of c.events.filter(e=>view.roles.has(e.id)))lines.push(e.id+' · MIDI '+e.pitch+' · '+e.start+' + '+e.duration+' · исходная группировка: '+e.voice+' · '+e.source_ref);
 if(selection.type==='object'&&selection.item.kind==='sonority')lines.push('Высоты участников (MIDI, регистр сохранён): '+c.events.filter(e=>selection.item.event_ids.includes(e.id)).map(e=>e.pitch).join(', '),'Это заявленная группа звучащих нот, не автоматически полная фактура или аккордовый символ.');
 return lines.join('\n')||'Нотные участники не установлены.';
}
function drawMap(c,view,selection){
 byId('map').replaceChildren();if(!c.events.length){byId('map').textContent='Нотные события недоступны.';return}
 const w=1050,h=390,x=t=>145+(t-view.lo)/(view.hi-view.lo)*(w-170),y=kind=>52+levels.indexOf(kind)*46;
 const svg=svgNode('svg',{viewBox:`0 0 ${w} ${h}`,role:'img','aria-label':'Карта уровней и участников'});
 const selectedPoints=[];
 for(const kind of levels){svg.append(svgNode('text',{x:8,y:y(kind)+4,fill:'#485c66','font-size':12},names[kind]));svg.append(svgNode('line',{x1:140,x2:w-20,y1:y(kind),y2:y(kind),stroke:'#e4e9e9'}));if(c.coverage[kind].status==='unavailable')svg.append(svgNode('text',{x:150,y:y(kind)-7,fill:'#6c7d86','font-size':11},'Нет данных'))}
 for(const e of view.notes){const selected=view.roles.get(e.id)||[],pos=x(frac(e.start));const n=svgNode('circle',{cx:pos,cy:y('note'),r:selected.length?5:3,fill:selected.length?view.color(selected[0]):'#c6d1d5','data-event-id':e.id});tooltip(n,e.id+' · MIDI '+e.pitch+' · '+e.start+' + '+e.duration+' · '+e.voice+' · '+e.source_ref);svg.append(n);if(selected.length)selectedPoints.push([pos,y('note')])}
 for(const o of c.objects){if(frac(o.start)>=view.hi||frac(o.end)<=view.lo)continue;const selected=view.objectRoles.get(o.id)||[];const left=x(Math.max(frac(o.start),view.lo)),right=x(Math.min(frac(o.end),view.hi));const rect=svgNode('rect',{x:left,y:y(o.kind)-10,width:Math.max(4,right-left),height:20,rx:3,fill:selected.length?view.color(selected[0]):'#dce4e7','data-object-id':o.id,'aria-label':o.label});tooltip(rect,o.id+' · '+o.label+' · ['+o.start+', '+o.end+')\n'+o.basis+'\n'+o.event_ids.join(', '));selectable(rect,()=>selectItem(c,{type:'object',item:o}));svg.append(rect);if(selected.length){selectedPoints.push([(left+right)/2,y(o.kind)]);for(const e of view.notes.filter(e=>o.event_ids.includes(e.id)))svg.append(svgNode('line',{x1:(left+right)/2,y1:y(o.kind),x2:x(frac(e.start)),y2:y('note'),stroke:view.color(selected[0]),'stroke-opacity':.25,'pointer-events':'none','data-membership':o.id+':'+e.id}))}}
 if(selection.type==='relation'&&selectedPoints.length){const hub=selectedPoints.reduce((sum,p)=>sum+p[0],0)/selectedPoints.length;for(const [px,py]of selectedPoints)svg.append(svgNode('line',{x1:hub,y1:18,x2:px,y2:py,stroke:'#647983','stroke-opacity':.45,'stroke-dasharray':'3 4','pointer-events':'none'}));svg.append(svgNode('text',{x:145,y:12,fill:'#425964','font-size':10},relationLabel(c,selection.item)))}
 byId('map').append(svg);
}
function drawPiano(c,view){
 byId('piano').replaceChildren();byId('axis').textContent='';if(!view.notes.length){byId('piano').textContent='Ноты для выбранной области недоступны.';return}
 const w=1050,h=290,p0=Math.min(...view.notes.map(e=>e.pitch))-2,p1=Math.max(...view.notes.map(e=>e.pitch))+2;
 const x=t=>55+(t-view.lo)/(view.hi-view.lo)*(w-75),y=p=>h-35-(p-p0)/(p1-p0)*(h-60),svg=svgNode('svg',{viewBox:`0 0 ${w} ${h}`,role:'img','aria-label':'Исходные ноты и окружающий контекст'});
 for(let p=Math.ceil(p0/12)*12;p<=p1;p+=12){svg.append(svgNode('line',{x1:55,x2:w-20,y1:y(p),y2:y(p),stroke:'#e4e9e9'}));svg.append(svgNode('text',{x:10,y:y(p)+3,'font-size':11,fill:'#667b86'},'C'+(p/12-1)))}
 for(let tick=0;tick<=5;tick++){const value=view.lo+(view.hi-view.lo)*tick/5;svg.append(svgNode('text',{x:x(value),y:h-8,'font-size':11,fill:'#667b86'},value.toFixed(2)))}
 for(const e of view.notes){const roles=view.roles.get(e.id)||[],stripes=roles.length?roles:[null];stripes.forEach((role,index)=>{const rect=svgNode('rect',{x:x(Math.max(view.lo,frac(e.start))),y:y(e.pitch)-4+index*8/stripes.length,width:Math.max(2,x(Math.min(view.hi,frac(e.start)+frac(e.duration)))-x(Math.max(view.lo,frac(e.start)))),height:8/stripes.length,fill:role===null?'#c6d1d5':view.color(role),'data-event-id':e.id,'data-role':role||'context'});tooltip(rect,e.id+' · MIDI '+e.pitch+' · '+e.start+' + '+e.duration+'\nИсходная группировка: '+e.voice+'\n'+e.source_ref+'\n'+(roles.join(', ')||'Окружающий контекст'));svg.append(rect)})}
 byId('piano').append(svg);byId('axis').textContent=(c.time_unit==='quarter_note'?'Время в четвертях':'Измеренные секунды')+' · '+c.time_origin+'. Точные исходные значения — при наведении. Серый — контекст, цвет — роль участника.';
}
function showClaim(cl){byId('source').textContent=cl?cl.source_anchor:'';if(cl)byId('source').href=cl.source_url;else byId('source').removeAttribute('href');byId('assessment').textContent=cl?cl.assessment:''}
function selectItem(c,selection){
 currentCase=c;currentSelection=selection;
 const item=selection.item,view=evidenceView(c,selection),cl=selection.type==='relation'?c.claims.find(cl=>cl.id===item.claim_id):selection.type==='claim'?item:null;
 byId('statement').textContent=cl?cl.statement:item.label;byId('status').textContent=selection.type==='relation'?names[item.level]||'Между уровнями':selection.type==='object'?names[item.kind]:'';
 if(cl)byId('status').textContent+=' · '+statuses[cl.status];if(selection.type==='relation')byId('status').textContent+=' · '+alignments[item.alignment]+' · Уверенность не оценивалась; в обучение не передано';
 showClaim(cl);byId('limits').replaceChildren();let limits=[...c.limitations];if(cl)limits=[...cl.missing,...cl.alternatives,...limits];if(selection.type==='object')limits=[item.basis,'Граница и название взяты из sidecar; инструмент и функция автоматически не выводятся.',...limits];for(const text of limits)byId('limits').append(node('li',text));
 byId('details').textContent=detailsFor(c,view,selection);byId('legend').replaceChildren();for(const role of view.roleNames){const label=node('span',role);label.style.borderColor=view.color(role);byId('legend').append(label)}byId('legend').append(node('span','Серый: окружающий контекст'));
 byId('objects').value=selection.type==='object'?item.id:'';for(const button of byId('relations').children)button.classList.toggle('active',button.dataset.key===selection.type+':'+item.id);
 drawMap(c,view,selection);drawPiano(c,view);
}
function selectCase(){
 const c=data.cases[Number(byId('case').value)];byId('meta').textContent=memberships[c.membership]+' · '+c.events.length+' событий · '+c.objects.length+' групп · '+c.relations.length+' связей';
 byId('relations').replaceChildren();for(const relation of c.relations){const claim=c.claims.find(cl=>cl.id===relation.claim_id),button=node('button',statuses[claim.status]+' · '+relationLabel(c,relation));button.title=claim.statement;button.dataset.key='relation:'+relation.id;button.onclick=()=>selectItem(c,{type:'relation',item:relation});byId('relations').append(button)}
 const linked=new Set(c.relations.map(r=>r.claim_id));for(const claim of c.claims.filter(cl=>!linked.has(cl.id))){const button=node('button',statuses[claim.status]+' · '+claim.statement);button.dataset.key='claim:'+claim.id;button.onclick=()=>selectItem(c,{type:'claim',item:claim});byId('relations').append(button)}
 byId('objects').replaceChildren();const placeholder=node('option','Выбрать группу');placeholder.value='';byId('objects').append(placeholder);for(const object of c.objects){const option=node('option',names[object.kind]+' · '+object.label);option.value=object.id;byId('objects').append(option)}byId('objects').disabled=!c.objects.length;byId('objects').onchange=()=>{const o=c.objects.find(o=>o.id===byId('objects').value);if(o)selectItem(c,{type:'object',item:o})};
 byId('coverage').replaceChildren();for(const level of levels){const entry=node('article');entry.append(node('strong',names[level]+' · '+states[c.coverage[level].status]),node('span',c.coverage[level].reason));byId('coverage').append(entry)}
 selectItem(c,c.relations.length?{type:'relation',item:c.relations[0]}:{type:'claim',item:c.claims[0]});
}
byId('title').textContent=data.title;data.cases.forEach((c,i)=>{const option=node('option',c.genres.join(' / ')+' · '+c.title);option.value=i;byId('case').append(option)});byId('scope').value='focus';byId('scope').onchange=()=>{if(currentCase)selectItem(currentCase,currentSelection)};byId('case').onchange=selectCase;selectCase();
</script></html>'''


def render_html(data: dict) -> str:
    """Render a self-contained inspector; no source retrieval or remote assets."""
    validate(data)
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    before_title, after_title = _VIEW.split("__TITLE__", 1)
    between, after_data = after_title.split("__DATA__", 1)
    return before_title + html.escape(data["title"]) + between + payload + after_data


def write_html_once(path: Path, data: dict) -> None:
    rendered = render_html(data)
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(rendered)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sidecar", type=Path)
    parser.add_argument("--html", type=Path, help="Create a standalone HTML file without overwriting")
    args = parser.parse_args()
    data = json.loads(args.sidecar.read_text(encoding="utf-8"))
    summary = validate(data)
    if args.html:
        write_html_once(args.html, data)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
