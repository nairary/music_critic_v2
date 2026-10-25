#!/usr/bin/env python3
"""Build an offline listening UI from an already prepared intervention payload.

This output-only builder never reads a corpus, model, or external service.
Titles/descriptions are displayed as text. Local media links are restricted to
relative paths. Musical validity and human preference remain unconfirmed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any


def _relative_media(value: Any) -> None:
    if not isinstance(value, str) or not value or any(c in value for c in ("\\", ":", "?", "#")):
        raise ValueError("Media paths must be plain relative paths")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or any(ord(c) < 32 for c in value):
        raise ValueError("Media paths must remain within the player directory")


def _positive_number(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def validate_payload(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict) or not isinstance(payload.get("works"), list) or not payload["works"]:
        raise ValueError("payload must contain a nonempty works list")
    work_ids = set()
    for work in payload["works"]:
        if not isinstance(work.get("id"), str) or work["id"] in work_ids:
            raise ValueError("Work IDs must be unique strings")
        work_ids.add(work["id"])
        _positive_number(work["duration_seconds"], "duration_seconds")
        _positive_number(work["seconds_per_quarter"], "seconds_per_quarter")
        if work["duration_seconds"] > 300:
            raise ValueError("Preview duration cannot exceed 300 seconds")
        if "preview_gain" in work:
            _positive_number(work["preview_gain"], "preview_gain")
        for key in ("original_wav", "original_midi"):
            if work.get(key):
                _relative_media(work[key])
        note_ids = set()
        for note in work["notes"]:
            if not isinstance(note.get("id"), str) or note["id"] in note_ids:
                raise ValueError("Note IDs must be unique strings within each work")
            note_ids.add(note["id"])
        interventions = work["interventions"]
        ids = {item["id"] for item in interventions}
        if len(ids) != len(interventions) or not all(isinstance(i, str) for i in ids):
            raise ValueError("Intervention IDs must be unique strings")
        for item in interventions:
            if item["aspect"] not in ("harmony", "rhythm", "melody", "deletion", "arrangement", "structure"):
                raise ValueError("Unknown intervention aspect")
            for patch in item["patches"]:
                if patch.get("note_id") not in note_ids:
                    raise ValueError("A patch references an unknown stable note ID")
        for pair in work.get("pairwise", []):
            if pair["left"] not in ids or pair["right"] not in ids:
                raise ValueError("Pairwise relation references an unknown intervention")
            if pair["status"] not in ("compatible", "semantic_revalidation", "conflict"):
                raise ValueError("Unknown pairwise status")
        for preset in work.get("presets", []):
            _relative_media(preset["file_stem"])
            if not set(preset["ids"]).issubset(ids):
                raise ValueError("Preset references an unknown intervention")


HTML = r'''<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Музыкальные правки — пилот</title>
<style>
:root{color-scheme:light;--ink:#182538;--muted:#617086;--line:#dce4eb;--bg:#f4f7fa;--blue:#235cba;--amber:#965400;--red:#a12232}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,sans-serif}main{max-width:1180px;margin:auto;padding:32px 24px 60px}h1{font-size:28px;line-height:1.2;margin:0 0 10px}h2{font-size:19px;margin:0 0 12px}h3{font-size:16px;margin:0}p{margin:6px 0}.muted,small{color:var(--muted)}.panel{background:white;border:1px solid var(--line);border-radius:14px;padding:20px;margin-top:18px}.row{display:flex;align-items:center;gap:10px;flex-wrap:wrap}.spread{justify-content:space-between}.cols{display:grid;grid-template-columns:1fr 1fr;gap:20px}button,select,.link{font:inherit;border-radius:8px;padding:8px 12px;border:1px solid #bfcbd7;background:white;color:var(--ink)}button,.link{cursor:pointer}button:hover,.link:hover{border-color:var(--blue)}button.primary{background:var(--blue);color:white;border-color:var(--blue)}button:disabled{opacity:.45;cursor:not-allowed}.link{display:inline-block;text-decoration:none;font-size:13px}select{max-width:100%;min-width:240px}audio{width:100%;margin:8px 0}.notice{border-radius:9px;padding:10px 13px;margin-top:12px;white-space:pre-line}.ok{background:#eaf5ee;color:#235135}.warning{background:#fff3d8;color:var(--amber)}.error{background:#ffe9ed;color:var(--red)}.badge{font-size:12px;border-radius:20px;padding:2px 9px;background:#edf2fa}.group{margin-top:24px}.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin-top:10px}.card{display:flex;gap:10px;align-items:flex-start;padding:13px;border:1px solid var(--line);border-radius:10px;background:white}.card:has(input:checked){border-color:#84a7dd;background:#f4f8ff}.card input{width:18px;height:18px;margin-top:3px;flex:none}.card label{cursor:pointer;width:100%}.card p{font-size:13px}.time{font-variant-numeric:tabular-nums;font-size:12px;color:var(--muted)}canvas{width:100%;height:180px;display:block;margin-top:10px;border:1px solid var(--line);border-radius:8px}.legend{font-size:12px;margin-top:6px}.presets{display:flex;flex-direction:column;gap:8px}.preset{display:flex;gap:7px;align-items:center;flex-wrap:wrap}.preset span{min-width:190px;font-size:13px}footer{font-size:12px;margin-top:24px;color:var(--muted)}#renderStatus{min-height:23px;font-size:13px}.count{font-variant-numeric:tabular-nums}ul{margin:4px 0;padding-left:20px}@media(max-width:720px){main{padding:22px 14px}.cols,.cards{grid-template-columns:1fr}.panel{padding:15px}h1{font-size:24px}}
.ab-controls button[aria-pressed="true"]{background:#235cba;color:white;border-color:#235cba}#abTimeline{height:76px}.ab-context select{min-width:110px}#compareFocus{flex:1;min-width:220px}.card-body{flex:1;min-width:0}.card-body button{margin-top:7px;font-size:13px}
</style></head><body><main>
<h1 id="appTitle">Музыкальные правки: послушать и сравнить</h1>
<p class="muted">Изменения нот с известным местом и механизмом. Оценка «стало хуже» пока не подтверждена человеком.</p>
<section class="panel"><div class="row spread"><label id="genreLabel">Жанр <select id="genreSelect" aria-label="Жанр"></select></label><label>Произведение <select id="workSelect" aria-label="Произведение"></select></label><span id="workSource" class="badge"></span></div><p id="tempoNote" class="muted"></p><p id="synthNote" class="muted">Звук — одинаковый диагностический клавишный синтез. Тембр, выразительность и исходная аранжировка не воспроизводятся. Громкость между вариантами не нормализуется отдельно.</p></section>
<section class="panel" id="comparePanel"><h2>Сравнение одного места · A/B</h2>
<p class="muted">A — оригинал, B — только одна указанная правка. Остальная музыка одинакова; флажки ниже не влияют на это сравнение.</p>
<div class="row"><select id="compareFocus" aria-label="Место для сравнения"></select><label class="ab-context">Контекст с каждой стороны <select id="compareContext" aria-label="Контекст"><option value="1">1 с</option><option value="2" selected>2 с</option><option value="4">4 с</option></select></label></div>
<p id="abDescription"></p><div class="row ab-controls"><button id="compareStart" class="primary">▶ Слушать петлю</button><button id="compareA" aria-pressed="true">A · Оригинал</button><button id="compareB" aria-pressed="false">B · Изменение</button><button id="compareRestart">С начала петли</button><button id="compareStop">Стоп</button></div>
<canvas id="abTimeline" width="1100" height="76" aria-label="Участок изменения и позиция прослушивания"></canvas><p id="abStatus" role="status" aria-live="polite"></p>
<p class="muted"><small>Петля повторяется. Переключайте A/B во время звучания: позиция сохраняется. Оранжевый участок — изменённые ноты; линия — текущая позиция. Для сравнения атаки нажмите «С начала петли».</small></p></section>
<section class="panel cols"><div><h2>Оригинальные ноты</h2><audio id="originalAudio" controls preload="none"></audio><div class="row"><a id="originalMidi" class="link" download>MIDI оригинала</a><button id="playOriginal">Синтез оригинала</button></div></div><div><h2>Выбранные правки <span id="selectedCount" class="count badge">0</span></h2><div class="row"><button id="playSelected" class="primary">Синтезировать и слушать</button><button id="stopPlayback">Стоп</button></div><p id="renderStatus" class="muted" aria-live="polite"></p><p class="muted"><small>Воспроизведение запускается только по нажатию. Выбранные правки применяются к стабильным ID нот; файлы оригинала остаются неизменными.</small></p></div></section>
<section class="panel"><div class="row spread"><h2>Места изменений</h2><div class="row"><button id="clearAll">Снять все</button><button id="selectAll">Все H/R/M</button></div></div><div id="compatibility" role="status" aria-live="polite"></div><canvas id="pianoRoll" width="1100" height="180" aria-label="Ноты оригинала и выбранные изменения"></canvas><p class="legend">Серый — оригинал · синий — изменённая нота · красный — удаление · вертикальные линии — каждые 5 секунд.</p><p id="alternativeNote" class="muted"></p><div id="groups"></div></section>
<section class="panel"><h2>Готовые варианты</h2><p id="presetTitle" aria-live="polite">Выберите вариант для прослушивания.</p><audio id="presetAudio" controls preload="none" hidden></audio><div id="presets" class="presets"></div><p class="muted"><small>Число выбранных мест — доза вмешательства, а не доказанная шкала музыкального качества. Комбинации с жёлтым предупреждением требуют повторной музыкальной оценки.</small></p></section>
<footer id="sourceScope">Локальный пилот · без отправки данных в сеть · только подготовленные TRAIN-фрагменты · без оценивания моделью и без обучения критика.</footer>
</main><script id="payload" type="application/json">__PAYLOAD__</script><script>
'use strict';
__GENRE_SYNTH__
const data=JSON.parse(document.getElementById('payload').textContent);
const $=id=>document.getElementById(id);
const labels={harmony:'Гармония',rhythm:'Ритмика',melody:'Мелодия',deletion:'Удаление нот',arrangement:'Фактура и партии',structure:'Порядок тактов'};
function presetSelectable(p){return p.compatibility!=='control';}
function bulkSelectionIds(items){return items.filter(i=>i.aspect!=='deletion').map(i=>i.id);}
function doseCounts(count){return [...new Set([1,3,5].map(n=>Math.min(n,count)))];}
function presetLabel(name){return name.replace(/^(harmony|rhythm|melody):/,(_,key)=>labels[key]+':').replace(/5 места$/,'5 мест');}
const aspects=Object.keys(labels), selected=new Set();
let work=data.works[0], context=null, activeSource=null, generation=0;
const genreLabels={classical:'Классика',rock:'Рок',electronic:'Электроника',hiphop:'Хип-хоп'};
const backgroundCache=new Map();
let abSession=null,abMode='A',abFrame=null;
function fraction(x){if(!Array.isArray(x)||x.length!==2||!Number.isInteger(x[0])||!Number.isInteger(x[1])||x[1]<=0)throw new Error('Некорректное точное время');return x[0]/x[1];}
function el(tag,text,cls){const x=document.createElement(tag);if(text!==undefined)x.textContent=text;if(cls)x.className=cls;return x;}
function secondsText(t){return Number(t).toFixed(1)+' с';}
function selectedItems(){return work.interventions.filter(x=>selected.has(x.id));}
function applyEdits(items){
 const map=new Map(work.notes.map(n=>[n.id,{...n}])), writes=new Map(), deleted=new Set(), touched=new Set();
 for(const item of items)for(const patch of item.patches){
  const id=patch.note_id;if(!map.has(id))throw new Error('Неизвестный ID ноты: '+id);
  if(patch.delete){if(touched.has(id))throw new Error('Удаление и изменение одной ноты: '+id);deleted.add(id);continue;}
  if(deleted.has(id))throw new Error('Изменение удалённой ноты: '+id);
  touched.add(id);const note=map.get(id);
  for(const key of ['pitch','onset_qn','duration_qn'])if(Object.hasOwn(patch,key)){
   const wk=id+'\u0000'+key,value=JSON.stringify(patch[key]);
   if(writes.has(wk)&&writes.get(wk)!==value)throw new Error('Две разные правки одного поля ноты: '+id);
   writes.set(wk,value);note[key]=patch[key];
   if(key==='onset_qn')note.onset_seconds=fraction(patch[key])*work.seconds_per_quarter;
   if(key==='duration_qn')note.duration_seconds=fraction(patch[key])*work.seconds_per_quarter;
  }
 }
 return [...map.values()].filter(n=>!deleted.has(n.id));
}
function compatibility(){
 const hard=[],soft=[];
 for(const p of work.pairwise||[])if(selected.has(p.left)&&selected.has(p.right)){
  const left=work.interventions.find(x=>x.id===p.left).title,right=work.interventions.find(x=>x.id===p.right).title;
  const line=left+' + '+right+': '+(p.reasons||[]).join('; ');
  if(p.status==='conflict')hard.push(line);else if(p.status==='semantic_revalidation')soft.push(line);
 }
 try{applyEdits(selectedItems());}catch(error){hard.push(error.message);}
 return {hard,soft};
}
function refresh(){
 const state=compatibility(),box=$('compatibility');$('selectedCount').textContent=selected.size;
 $('playSelected').disabled=state.hard.length>0;
 if(state.hard.length){box.className='notice error';box.textContent='Конфликт: воспроизведение этой комбинации заблокировано.\n'+state.hard.join('\n');}
 else if(state.soft.length){box.className='notice warning';box.textContent='Технически совместимо; музыкальный эффект комбинации нужно проверить заново.\n'+state.soft.join('\n');}
 else{box.className='notice ok';box.textContent=selected.size?'Технических конфликтов выбранных правок не обнаружено. Музыкальная оценка остаётся открытой.':'Правки не выбраны: можно послушать оригинал или включить отдельные места.';}
 for(const input of document.querySelectorAll('input[data-edit]'))input.checked=selected.has(input.dataset.edit);
 drawRoll();
}
function stop(){generation++;stopAB();if(activeSource){try{activeSource.stop();}catch(_){}activeSource=null;}$('originalAudio').pause();$('presetAudio').pause();$('renderStatus').textContent='';$('playOriginal').disabled=false;refresh();}
function drawRoll(){
 const canvas=$('pianoRoll'),c=canvas.getContext('2d'),W=canvas.width,H=canvas.height;
 c.clearRect(0,0,W,H);const pitches=work.notes.map(n=>n.pitch),lo=Math.min(...pitches)-2,hi=Math.max(...pitches)+2;
 const y=p=>H-25-(p-lo)/(hi-lo||1)*(H-40),x=t=>40+t/work.duration_seconds*(W-50);
 c.font='11px system-ui';c.fillStyle='#617086';
 for(let t=0;t<=work.duration_seconds;t+=5){c.strokeStyle='#e7edf3';c.beginPath();c.moveTo(x(t),8);c.lineTo(x(t),H-21);c.stroke();c.fillText(t+'s',x(t)-5,H-5);}
 for(let p=Math.ceil(lo/12)*12;p<=hi;p+=12)c.fillText('C'+(p/12-1),2,y(p)+3);
 const draw=(n,color)=>{c.fillStyle=color;c.fillRect(x(n.onset_seconds),y(n.pitch)-2,Math.max(2,n.duration_seconds/work.duration_seconds*(W-50)),4);};
 work.notes.forEach(n=>draw(n,'#bac6d2'));
 const ids=new Set(selectedItems().flatMap(i=>i.patches.map(p=>p.note_id)));
 try{for(const n of applyEdits(selectedItems()))if(ids.has(n.id))draw(n,'#235cba');}catch(_){}
 for(const item of selectedItems())for(const p of item.patches)if(p.delete){const n=work.notes.find(n=>n.id===p.note_id);if(n)draw(n,'#ca354b');}
}
function loadWork(){
 selected.clear();stop();$('workSource').textContent=(work.genre?genreLabels[work.genre]+' · ':'')+(work.source_id||work.id);
 $('workSelect').value=work.id;
 $('selectAll').hidden=Boolean(data.alternatives_only);
 $('alternativeNote').textContent=data.alternatives_only?'Это отдельные сценарии. Выбирайте один: некоторые из них изменяют одни и те же ноты.':'';
 const focus=$('compareFocus');focus.replaceChildren();
 for(const item of work.interventions){const option=el('option',item.id+' · '+labels[item.aspect]+' · '+secondsText(item.location_seconds));option.value=item.id;focus.append(option);}
 if(work.interventions.length)focus.value=work.interventions[0].id;
 abMode='A';updateABPreview();
 $('presetAudio').removeAttribute('src');$('presetAudio').load();$('presetAudio').hidden=true;
 $('presetTitle').textContent='Выберите вариант: '+work.title;
 $('tempoNote').textContent=work.tempo_note||'Для прослушивания условно принято 120 BPM; это не подтверждённый исходный темп.';
 $('originalAudio').src=work.original_wav||'';$('originalAudio').hidden=!work.original_wav;
 $('originalMidi').href=work.original_midi||'';$('originalMidi').hidden=!work.original_midi;
 const groups=$('groups');groups.replaceChildren();
 for(const aspect of aspects){
  const items=work.interventions.filter(i=>i.aspect===aspect);if(!items.length)continue;
  const group=el('div',undefined,'group'),head=el('div',undefined,'row spread');
  head.append(el('h3',labels[aspect]));const buttons=el('div',undefined,'row');
  for(const count of (data.alternatives_only?[]:doseCounts(items.length))){const b=el('button','Первые '+count);b.title='Выбрать '+count+' отдельных мест этой категории';b.onclick=()=>{stop();for(const i of items)selected.delete(i.id);for(const i of items.slice(0,count))selected.add(i.id);refresh();};buttons.append(b);}
  head.append(buttons);group.append(head);const cards=el('div',undefined,'cards');
  items.forEach((item,index)=>{const card=el('div',undefined,'card'),input=el('input');input.type=data.alternatives_only?'radio':'checkbox';if(data.alternatives_only)input.name='harmonic-alternative';input.id='edit-'+aspect+'-'+index;input.dataset.edit=item.id;
   const label=el('label');label.htmlFor=input.id;label.append(el('strong',item.title),el('div',secondsText(item.location_seconds)+' · '+item.id,'time'),el('p',item.description));
   input.onchange=()=>{const checked=input.checked;stop();if(data.alternatives_only)selected.clear();if(checked)selected.add(item.id);else selected.delete(item.id);refresh();};const body=el('div',undefined,'card-body'),compare=el('button','Сравнить A/B');compare.onclick=()=>{$('compareFocus').value=item.id;updateABPreview();$('comparePanel').scrollIntoView({behavior:'smooth',block:'start'});startAB();};body.append(label,compare);card.append(input,body);cards.append(card);});
  group.append(cards);groups.append(group);
 }
 const presets=$('presets');presets.replaceChildren();
 for(const p of work.presets||[]){const row=el('div',undefined,'preset');row.append(el('span',presetLabel(p.name)));
  const listen=el('button','Слушать');listen.onclick=()=>playPreset(p);row.append(listen);
  if(presetSelectable(p)){const b=el('button','Выбрать');b.onclick=()=>{stop();selected.clear();for(const id of p.ids)selected.add(id);refresh();};row.append(b);}
  else row.append(el('small','Контроль доступен в WAV/MIDI'));
  for(const ext of ['wav','mid']){const a=el('a',ext==='wav'?'WAV':'MIDI','link');a.href=p.file_stem+'.'+ext;a.download='';row.append(a);}presets.append(row);
  const failed=(p.observable_checks||[]).filter(c=>!c.passed);
  if(failed.length||p.all_observable_conditions_hold===false)presets.append(el('p','Часть ожидаемых эффектов в комбинации не сохранилась: '+(failed.map(c=>c.id).join(', ')||'см. протокол')+'. Это не подтверждённая сумма ухудшений.','notice warning'));
 }
 if(!(work.presets||[]).length)presets.append(el('p','Для этого фрагмента подготовлены только интерактивные правки.','muted'));
 refresh();
}
function syncWork(force=false){
 const next=data.works.find(w=>w.id===$('workSelect').value)||data.works[0];
 if(!force&&next.id===work.id)return;
 work=next;loadWork();
}
async function playPreset(p){
 stop();const sourceWork=work,myGeneration=generation,audio=$('presetAudio');
 audio.src=p.file_stem+'.wav';audio.hidden=false;
 $('presetTitle').textContent=sourceWork.title+' — '+presetLabel(p.name);
 try{await audio.play();}catch(error){if(generation===myGeneration)$('presetTitle').textContent=sourceWork.title+' — '+error.message;}
}
function focusItem(){return work.interventions.find(i=>i.id===$('compareFocus').value);}
function comparisonWindow(sourceWork,item,contextSeconds){
 const starts=[],ends=[];
 for(const patch of item.patches){
  const note=sourceWork.notes.find(n=>n.id===patch.note_id);if(!note)throw new Error('Неизвестная нота сравнения');
  starts.push(note.onset_seconds);ends.push(note.onset_seconds+note.duration_seconds);
  if(!patch.delete){const onset=patch.onset_qn?fraction(patch.onset_qn)*sourceWork.seconds_per_quarter:note.onset_seconds;
   const duration=patch.duration_qn?fraction(patch.duration_qn)*sourceWork.seconds_per_quarter:note.duration_seconds;
   starts.push(onset);ends.push(onset+duration);}
 }
 const changedStart=Math.max(0,Math.min(...starts)),changedEnd=Math.min(sourceWork.duration_seconds,Math.max(...ends));
 return {start:Math.max(0,changedStart-contextSeconds),end:Math.min(sourceWork.duration_seconds,changedEnd+contextSeconds),changedStart,changedEnd};
}
function currentWindow(){const item=focusItem();return item?comparisonWindow(work,item,Number($('compareContext').value)||2):null;}
function drawAB(position){
 const range=abSession?.range||currentWindow(),canvas=$('abTimeline'),c=canvas.getContext('2d'),W=canvas.width;
 c.clearRect(0,0,W,canvas.height);if(!range)return;
 const x=t=>20+(t-range.start)/Math.max(.001,range.end-range.start)*(W-40);
 c.fillStyle='#edf2fa';c.fillRect(20,10,W-40,32);
 c.fillStyle='#f6c46c';c.fillRect(x(range.changedStart),10,Math.max(3,x(range.changedEnd)-x(range.changedStart)),32);
 c.font='13px system-ui';c.fillStyle='#617086';c.fillText(secondsText(range.start),20,65);c.fillText(secondsText(range.end),W-65,65);
 c.fillStyle='#965400';c.fillText('Изменение: '+secondsText(range.changedStart),Math.min(W-220,Math.max(20,x(range.changedStart))),26);
 if(position!==undefined){c.strokeStyle='#235cba';c.lineWidth=3;c.beginPath();c.moveTo(x(position),4);c.lineTo(x(position),45);c.stroke();}
}
function updateABButtons(){
 for(const mode of ['A','B'])$('compare'+mode).setAttribute('aria-pressed',String(mode===abMode));
 $('compareStart').disabled=!focusItem();$('compareRestart').disabled=!abSession;
}
function updateABPreview(){
 const item=focusItem();$('abDescription').textContent=item?work.title+' — '+item.title+'. '+item.description:'Нет правок для сравнения.';
 $('abStatus').textContent=item?'Нажмите «Слушать петлю», затем переключайте A/B.':'';
 updateABButtons();drawAB();
}
function stopAB(){
 if(abFrame!==null){cancelAnimationFrame(abFrame);abFrame=null;}
 if(abSession){for(const source of abSession.sources){try{source.stop();}catch(_){}source.disconnect();}for(const gain of abSession.gains)gain.disconnect();abSession=null;}
 $('abStatus').textContent='';updateABButtons();drawAB();
}
function selectAB(mode){
 abMode=mode;
 if(abSession){const now=context.currentTime;
  abSession.gains.forEach((node,index)=>{const param=node.gain;param.cancelScheduledValues(now);param.setValueAtTime(param.value,now);param.linearRampToValueAtTime((mode==='A'?0:1)===index?1:0,now+.015);});
  $('abStatus').textContent=abSession.title+' · '+abSession.itemId+' · '+(mode==='A'?'A — оригинал':'B — одна правка')+' · петля '+secondsText(abSession.range.start)+'–'+secondsText(abSession.range.end);
 }
 updateABButtons();
}
function startABSources(buffers,range,title,itemId){
 const when=context.currentTime+.03,sources=[],gains=[];
 // Both streams share a scheduled start and loop bounds; A/B only changes gain.
 try{for(let i=0;i<2;i++){
  const source=context.createBufferSource(),gain=context.createGain();sources.push(source);gains.push(gain);
  source.buffer=buffers[i];source.loop=true;source.loopStart=range.start;source.loopEnd=range.end;
  gain.gain.setValueAtTime((abMode==='A'?0:1)===i?1:0,context.currentTime);
  source.connect(gain);gain.connect(context.destination);source.start(when,range.start);
 }}catch(error){for(const source of sources){try{source.stop();}catch(_){}source.disconnect();}for(const gain of gains)gain.disconnect();throw error;}
 abSession={sources,gains,buffers,range,title,itemId,startedAt:when};selectAB(abMode);
 const frame=()=>{if(!abSession)return;const length=range.end-range.start,elapsed=Math.max(0,context.currentTime-when);drawAB(range.start+elapsed%length);abFrame=requestAnimationFrame(frame);};
 frame();
}
async function startAB(){
 stop();const item=focusItem();if(!item)return;
 const sourceWork=work,range=currentWindow(),token=generation,original=work.notes.map(n=>({...n})),edited=applyEdits([item]);
 $('compareStart').disabled=true;$('abStatus').textContent='Готовлю две синхронные версии…';
 try{
  const AudioContextClass=window.AudioContext||window.webkitAudioContext;if(!AudioContextClass)throw new Error('WebAudio недоступен; используйте готовые WAV.');
  context=context||new AudioContextClass();await context.resume();if(generation!==token)return;
  const a=await renderBuffer(sourceWork,original,token);if(!a)return;
  const b=await renderBuffer(sourceWork,edited,token);if(!b)return;
  if(generation!==token)return;
  softenLoopEdges([a.buffer,b.buffer],range);
  startABSources([a.buffer,b.buffer],range,sourceWork.title,item.id);
 }catch(error){if(generation===token)$('abStatus').textContent=error.message;}
 finally{if(generation===token)updateABButtons();}
}
function restartAB(){if(!abSession)return;const {buffers,range,title,itemId}=abSession;stop();startABSources(buffers,range,title,itemId);}
function softenLoopEdges(buffers,range){
 // Identical 8ms fades suppress cut clicks, without per-version normalization.
 for(const buffer of buffers){const samples=buffer.getChannelData(0),sr=buffer.sampleRate;
  const first=Math.ceil(range.start*sr),last=Math.min(samples.length,Math.floor(range.end*sr));
  const count=Math.min(Math.round(.008*sr),Math.floor((last-first)/4));
  for(let j=0;j<count;j++){const weight=count>1?j/(count-1):0;samples[first+j]*=weight;samples[last-1-j]*=weight;}
 }
}

async function renderBuffer(sourceWork,events,myGeneration){
  const sr=22050,frames=Math.floor(sourceWork.duration_seconds*sr+.5),buffer=context.createBuffer(1,frames,sr),samples=buffer.getChannelData(0);
  const gain=sourceWork.preview_gain??data.preview_gain??(0.9/(1.555*Math.max(1,sourceWork.notes.length)));
  if(data.renderer==='DiagnosticGMFamilies@1.0.0'){
   for(let i=0;i<events.length;i++){addGenreNote(samples,sr,events[i],gain);if(i%20===0){await new Promise(resolve=>setTimeout(resolve,0));if(generation!==myGeneration)return null;}}
   if(sourceWork.background_pcm16){
    let bg=backgroundCache.get(sourceWork.id);if(!bg){const raw=atob(sourceWork.background_pcm16),bytes=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);const view=new DataView(bytes.buffer);bg=new Float32Array(raw.length/2);for(let i=0;i<bg.length;i++)bg[i]=view.getInt16(2*i,true)/32768;backgroundCache.set(sourceWork.id,bg);}
    for(let i=0;i<Math.min(frames,bg.length);i++)samples[i]+=bg[i]*(sourceWork.background_gain||0);
   }
   finishGenreSamples(samples,sr);let peak=0;for(const x of samples)peak=Math.max(peak,Math.abs(x));if(peak>1)throw new Error('Клиппинг общей смеси');return {buffer,peak,gain};
  }
  const harmonics=[[1,1],[2,.28],[3,.16],[4,.08],[5,.035]];
  events.sort((a,b)=>a.onset_seconds-b.onset_seconds||a.pitch-b.pitch||a.duration_seconds-b.duration_seconds||(a.velocity??80)-(b.velocity??80));
  for(let ni=0;ni<events.length;ni++){
   const n=events[ni],velocity=n.velocity??80,duration=n.duration_seconds,start=Math.floor(n.onset_seconds*sr+.5);
   if(duration===0||velocity===0||start>=frames)continue;
   if(start<0||duration<0||!Number.isFinite(start)||!Number.isFinite(duration)||!Number.isInteger(n.pitch)||n.pitch<0||n.pitch>127)throw new Error('Некорректное событие после правки');
   const f=440*Math.pow(2,(n.pitch-69)/12);if(f>=.45*sr)throw new Error('Нота слишком высокая для выбранной частоты синтеза; используйте MIDI.');
   const count=Math.min(frames-start,Math.ceil((duration+.08)*sr));
   for(let j=0;j<count;j++){const t=j/sr,envelope=Math.min(t/.003,1)*Math.max(0,Math.min(1,(duration+.08-t)/.08));let voice=0;
    for(const [h,a]of harmonics)if(f*h<.45*sr)voice+=a*Math.exp(-t*(.65+.30*(h-1)))*Math.sin(2*Math.PI*f*h*t);
    samples[start+j]+=gain*(velocity/127)*envelope*voice;
   }
   if(ni%20===0){await new Promise(resolve=>setTimeout(resolve,0));if(generation!==myGeneration)return null;}
  }
  const fade=Math.min(frames,Math.round(.02*sr));for(let j=0;j<fade;j++)samples[frames-fade+j]*=fade>1?1-j/(fade-1):1;
  let peak=0;for(const x of samples)peak=Math.max(peak,Math.abs(x));if(peak>1)throw new Error('Обнаружен клиппинг; нужно уменьшить общий gain всех вариантов.');
  if(generation!==myGeneration)return null;
  return {buffer,peak,gain};
}
async function play(useOriginal){
 stop();const state=compatibility();if(!useOriginal&&state.hard.length)return;
 const sourceWork=work,myGeneration=generation,editCount=selected.size,events=useOriginal?sourceWork.notes.map(n=>({...n})):applyEdits(selectedItems());
 $('playOriginal').disabled=true;$('playSelected').disabled=true;$('renderStatus').textContent='Подготавливаю диагностический звук…';
 try{
  const AudioContextClass=window.AudioContext||window.webkitAudioContext;if(!AudioContextClass)throw new Error('WebAudio недоступен в этом браузере. Используйте готовые WAV.');
  context=context||new AudioContextClass();await context.resume();
  if(generation!==myGeneration)return;
  const rendered=await renderBuffer(sourceWork,events,myGeneration);if(!rendered)return;
  const {buffer,peak,gain}=rendered;
  activeSource=context.createBufferSource();activeSource.buffer=buffer;activeSource.connect(context.destination);activeSource.start();
  $('renderStatus').textContent=sourceWork.title+' · '+(useOriginal?'Оригинал':'Правок: '+editCount)+' · '+secondsText(sourceWork.duration_seconds)+' · peak '+peak.toFixed(3)+' · общий gain '+gain.toFixed(4);
  activeSource.onended=()=>{if(generation===myGeneration){activeSource=null;$('playOriginal').disabled=false;refresh();}};
 }catch(error){if(generation===myGeneration){$('renderStatus').textContent=error.message;$('playOriginal').disabled=false;refresh();}}
}
function fillWorks(){const old=$('workSelect').value;const genre=$('genreSelect').value;const available=data.works.filter(w=>!genre||w.genre===genre);$('workSelect').replaceChildren();for(const w of available){const option=el('option',(w.genre?genreLabels[w.genre]+' · ':'')+w.title);option.value=w.id;$('workSelect').append(option);}if(available.some(w=>w.id===old))$('workSelect').value=old;else if(available.length)$('workSelect').value=available[0].id;}
$('genreSelect').append(Object.assign(el('option','Все жанры'),{value:''}));for(const genre of Object.keys(genreLabels))if(data.works.some(w=>w.genre===genre)){$('genreSelect').append(Object.assign(el('option',genreLabels[genre]),{value:genre}));}
$('genreLabel').hidden=!data.works.some(w=>w.genre);fillWorks();$('genreSelect').onchange=()=>{fillWorks();syncWork(true);};
if(data.title)$('appTitle').textContent=data.title;
if(data.source_scope)$('sourceScope').textContent=data.source_scope;
if(data.renderer==='DiagnosticGMFamilies@1.0.0')$('synthNote').textContent='Диагностические тембры различают инструменты и ударные. Это не исходное звучание записи; роли и ограничения указаны для каждого произведения.';
$('compareFocus').onchange=()=>{stop();updateABPreview();};$('compareContext').onchange=()=>{stop();updateABPreview();};
$('compareStart').onclick=startAB;$('compareA').onclick=()=>selectAB('A');$('compareB').onclick=()=>selectAB('B');$('compareRestart').onclick=restartAB;$('compareStop').onclick=stop;
$('workSelect').onchange=()=>syncWork();$('workSelect').oninput=()=>syncWork();
window.addEventListener('pageshow',()=>{fillWorks();syncWork(true);});
$('clearAll').onclick=()=>{stop();selected.clear();refresh();};$('selectAll').onclick=()=>{stop();selected.clear();bulkSelectionIds(work.interventions).forEach(id=>selected.add(id));refresh();};
$('playSelected').onclick=()=>play(false);$('playOriginal').onclick=()=>play(true);$('stopPlayback').onclick=stop;
function mediaStarted(other){generation++;stopAB();if(activeSource){try{activeSource.stop();}catch(_){}activeSource=null;}$(other).pause();$('renderStatus').textContent='';$('playOriginal').disabled=false;refresh();}
$('originalAudio').onplay=()=>mediaStarted('presetAudio');$('presetAudio').onplay=()=>mediaStarted('originalAudio');
syncWork(true);
</script></body></html>'''


def build_player(payload: dict[str, Any], output_path: Path) -> None:
    """Validate and exclusively write an offline player; preserve old outputs."""
    validate_payload(payload)
    serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    # Script data is not HTML: escaping '<' prevents a title from closing script.
    serialized = serialized.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    serialized = serialized.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as output:
        template = HTML.replace("__PAYLOAD__", serialized)
        if payload.get("renderer") == "DiagnosticGMFamilies@1.0.0":
            import importlib.util
            spec = importlib.util.spec_from_file_location("genre_preview_for_player", Path(__file__).with_name("render_genre_preview.py"))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            template = template.replace("__GENRE_SYNTH__", module.browser_synth_js())
        else:
            template = template.replace("__GENRE_SYNTH__", "")
        output.write(template)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("payload", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build_player(json.loads(args.payload.read_text()), args.output)


if __name__ == "__main__":
    main()
