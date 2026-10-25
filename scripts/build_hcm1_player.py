"""Self-contained local listening page from a registered diagnostic receipt."""
from __future__ import annotations
from dataclasses import replace
from hashlib import sha256
import argparse
import json
from pathlib import Path

from music_critic.data import loads_piece,RationalTime as Q
from scripts.hcm1_diagnostic_data import read_bound,query_from_json,audio_events,qn_seconds
from scripts.render_genre_preview import browser_synth_js


def script_json(value):
    return json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':')).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026').replace('\u2028','\\u2028').replace('\u2029','\\u2029')


def display_span(piece,query):
    hits=[i for i,b in enumerate(piece.bars) if b.start_qn<query.end and b.start_qn+b.duration_qn>query.start]
    first=max(0,hits[0]-1);last=min(len(piece.bars)-1,hits[-1]+1)
    return piece.bars[first].start_qn,piece.bars[last].start_qn+piece.bars[last].duration_qn


def listening_query(piece,variant):
    from scripts.hcm1_diagnostic_data import pair_query
    from music_critic.pll.contracts import MaskQuery
    try:return pair_query(piece,variant)[0]
    except ValueError:
        left={n.note_id:n for n in piece.notes};right={n.note_id:n for n in variant.notes}
        def content(n):return None if n is None else (n.pitch,n.onset_qn,n.duration_qn)
        notes=[n for key in left.keys()|right.keys() if content(left.get(key))!=content(right.get(key))
               for n in (left.get(key),right.get(key)) if n is not None]
        if not notes:raise ValueError('no listening edit')
        start=min(n.onset_qn for n in notes);end=max(n.onset_qn for n in notes)
        bars=[b for b in piece.bars if b.start_qn<=end and b.start_qn+b.duration_qn>start]
        return MaskQuery(bars[0].start_qn,bars[-1].start_qn+bars[-1].duration_qn,tuple(t.track_id for t in piece.tracks))


def generation_audio(piece,query,raw,start,end):
    # Keep observed notes and fill only the query-owned attacks. HCM does not
    # predict velocity/program: explicit neutral velocity80, observed instrument.
    observed=replace(piece,notes=tuple(n for n in piece.notes if not(n.track_id in query.track_ids and query.start<=n.onset_qn<query.end)))
    events,duration=audio_events(observed,start,end);origin=qn_seconds(piece,start)
    for i,e in enumerate(raw):
        onset=query.start+Q(**e['onset']);finish=onset+Q(**e['duration']);track=piece.tracks[e['track']]
        events.append(dict(id=f'generated:{i}',pitch=e['pitch'],velocity=80,program=track.program or 0,
            is_percussion=e['kind']>=2,onset_seconds=max(0,qn_seconds(piece,onset)-origin),
            duration_seconds=max(0,qn_seconds(piece,min(finish,end))-qn_seconds(piece,max(onset,start)))))
    return events,duration


def payload(cases_path,receipt_path,record_path):
    from music_critic.experiments.registry import load_record
    cases_path=Path(cases_path);case_raw=cases_path.read_bytes();raw=Path(receipt_path).read_bytes()
    cases=json.loads(case_raw);receipt=json.loads(raw)
    if cases['status']!='completed' or receipt['status']!='completed':raise ValueError('completed evidence required')
    record=load_record(record_path)
    if record['status']!='completed' or not any(a['sha256']==sha256(raw).hexdigest() and a['size_bytes']==len(raw) for a in record['artifacts']):
        raise ValueError('receipt not bound to terminal evaluation')
    binding=record['hcm_diagnostic']
    if binding!=receipt['binding'] or binding['cases_sha256']!=sha256(case_raw).hexdigest():
        raise ValueError('scores and listening cases differ')
    access=record['data_access']
    update=receipt['results']['parent_update']
    if update not in (10000,20000) or record['results']['parent_update']!=update or any(access[k] is not False for k in ('test_inputs_accessed','test_targets_accessed','test_metrics_computed','test_used_for_selection')):
        raise ValueError('page checkpoint or access claim not established')
    results=receipt['results'];scores={r['id']:r for r in results['scores']};gens={r['id']:r for r in results['generations']}
    rows=[]
    for case in cases['cases']:
        # Show all40 pairs and the fixed validation restoration cases. TRAIN
        # comparison remains in the score summary rather than listening cards.
        if case['split']=='train':continue
        row=dict(case);row['score']=scores[case['id']];row['audio']={};row['gain']=.012
        if case['id'] in gens:row['generation']=gens[case['id']]
        if 'original' not in case:
            row['audio_reason']=case.get('reason','No admitted query');rows.append(row);continue
        piece=loads_piece(read_bound(cases_path.parent,case['original']).decode())
        variant=loads_piece(read_bound(cases_path.parent,case['variant']).decode()) if case['kind']=='pair' else None
        q=query_from_json(case['query']) if 'query' in case else listening_query(piece,variant)
        start,end=display_span(piece,q);duration=qn_seconds(piece,end)-qn_seconds(piece,start)
        if not 0<duration<=300:
            row['audio_reason']='Listening excerpt exceeds300seconds';rows.append(row);continue
        row['mask_seconds']=[qn_seconds(piece,q.start)-qn_seconds(piece,start),qn_seconds(piece,q.end)-qn_seconds(piece,start)]
        row['duration_seconds']=duration;row['original_piece_id']=piece.piece_id
        row['audio']['original']=audio_events(piece,start,end)[0]
        if case['kind']=='pair':
            row['audio']['variant']=audio_events(variant,start,end)[0]
        if case['id'] in gens:
            row['generation']=gens[case['id']]
            if gens[case['id']]['status']=='completed':row['audio']['generated']=generation_audio(piece,q,gens[case['id']]['events'],start,end)[0]
        rows.append(row)
    return dict(schema='HCM1ListeningPage@1',cases=rows,all_scores=results['scores'],
        parent_update=update,
        source_receipt_sha256=sha256(raw).hexdigest(),
        cases_sha256=sha256(case_raw).hexdigest(),
        generation_render='velocity80 and observed instrument; HCM predicts onset/track/kind/pitch/duration only')


def attach_review_context(data,previous,ratings,comparison):
    from scripts.hcm1_listening_ratings import review_ratings
    review_ratings(ratings,previous)
    if data['cases_sha256']!=comparison['cases_sha256']:raise ValueError('comparison cases differ')
    old={r['id']:r for r in previous['cases']};scores={r['id']:r['score'] for r in comparison['cases']}
    for row in data['cases']:
        prior=old.get(row['id']);row['new_example']=prior is None
        if prior is not None:
            if any(row.get(k)!=prior.get(k) for k in ('original','variant','query','context_sha256')):
                raise ValueError('prior audio identity or mask changed')
            if row['id'] in ratings['ratings']:row['prior_rating']=dict(ratings['ratings'][row['id']])
        row['comparison_score']=scores[row['id']]
    data['prior_ratings_source_receipt_sha256']=ratings['source_receipt_sha256']
    data['prior_rating_count']=len(ratings['ratings'])
    data['comparison_receipt_sha256']=comparison['source_receipt_sha256']
    data['comparison_update']=comparison.get('parent_update',10000)
    return data


def page(data):
    update=f"{data.get('parent_update',10000):,}".replace(',',' ')
    return HTML.replace('__SYNTH__',browser_synth_js()).replace('__UPDATE__',update).replace('__DATA__',script_json(data))


HTML=r'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HCM-1 | Музыка и контекст</title>
<style>
:root{--ink:#162b35;--muted:#63757c;--bg:#f4f2eb;--line:#d9dfdb;--accent:#16726c;--warm:#d59b48}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:var(--bg);font:15px/1.5 system-ui,sans-serif}header{padding:30px 5vw 24px;border-bottom:1px solid var(--line)}header small{letter-spacing:.18em;color:var(--accent);font-weight:700}h1{font-size:34px;margin:8px 0}p{max-width:900px}.subtle{color:var(--muted)}.layout{display:grid;grid-template-columns:300px minmax(0,1fr);max-width:1500px;margin:auto;min-height:70vh}aside{padding:24px;border-right:1px solid var(--line)}main{padding:28px 36px}button,select,input{font:inherit;border:1px solid var(--line);background:white;border-radius:8px;padding:9px 13px;color:var(--ink)}button{cursor:pointer}button:hover{border-color:var(--accent)}button.active,.primary{background:var(--accent);color:white}input,select{width:100%;margin-bottom:10px}.list{max-height:65vh;overflow:auto}.item{display:block;width:100%;text-align:left;margin:7px 0}.item small{display:block;font-size:11px;opacity:.8}.topline{display:flex;gap:12px;align-items:center;justify-content:space-between}.tag{padding:4px 10px;border-radius:20px;background:#e4ece6;font-size:12px}.panel{background:#fff;border:1px solid var(--line);border-radius:14px;padding:20px;margin:18px 0}.controls{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0}canvas{display:block;width:100%;height:200px;background:#f8faf8;border-radius:8px}#seek{margin:8px 0;accent-color:var(--accent)}h2{font-size:25px;overflow-wrap:anywhere;margin:12px 0}h3{font-size:16px;margin:0 0 10px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.5 ui-monospace,monospace}.metrics{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.metric{background:#f4f6f3;padding:14px;border-radius:8px}.metric b{display:block;font-size:25px}.ratings{display:flex;flex-wrap:wrap;gap:8px}details{margin:14px 0}summary{cursor:pointer}#error{color:#9d391c}.empty{padding:30px}.tiny{font-size:12px}footer{border-top:1px solid var(--line);padding:20px 5vw;color:var(--muted);font-size:12px}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:7px;border-bottom:1px solid var(--line)}@media(max-width:800px){.layout{grid-template-columns:1fr}aside{border-right:0;border-bottom:1px solid var(--line)}.list{max-height:170px}main{padding:20px}.metrics{grid-template-columns:1fr}h1{font-size:28px}}
</style>
<header><small>HCM-1 / ПРОВЕРКА НА __UPDATE__ ШАГОВ</small><h1>Послушать. Сравнить. Оценить.</h1><p class="subtle">Один музыкальный контекст — два заполнения скрытого участка. Сначала послушайте и сформулируйте своё мнение, затем раскройте оценки модели.</p><details><summary>Что проверяет этот просмотр</summary><p>Модель оценивает вероятность нот внутри маски по окружающей музыке. Более вероятный вариант не обязательно лучше на слух. Аудио синтезировано из нот одним диагностическим синтезатором; это не исходная запись исполнения. Громкость двух версий одинакова. Небольшая выборка не даёт общей оценки качества.</p><div id="summary"></div></details></header>
<div class="layout"><aside><select id="kind"><option value="pair">Оригинал и правка</option><option value="restoration">Восстановление новой музыки</option><option value="all">Все примеры</option></select><select id="subset"><option value="new">Новые примеры</option><option value="all">Все примеры</option><option value="rated">Мои прошлые оценки</option></select><input id="search" placeholder="Поиск по произведению или жанру"><div class="subtle tiny" id="count"></div><div id="list" class="list"></div><button id="export" style="margin-top:18px">Скачать мои оценки</button><p class="subtle tiny">Оценки сохраняются в этом браузере. Для надёжного сохранения скачайте JSON.</p></aside><main id="main"></main></div>
<footer>Модель после __UPDATE__ шагов. Примеры из заранее выбранных отложенных наборов. Итоговый тестовый набор и резерв кампании не использованы. PLL здесь — условная оценка заданного блока, а не вероятность произведения целиком.</footer>
<script id="data" type="application/json">__DATA__</script><script>
'use strict';const DATA=JSON.parse(document.getElementById('data').textContent);
__SYNTH__
const $=id=>document.getElementById(id),fmt=x=>x==null?'—':Number(x).toFixed(3),labels={original:'Оригинал',variant:'С правкой',generated:'Восстановление HCM'},genres={electronic:'Электроника',hiphop:'Хип-хоп',classical:'Классика',jazz:'Джаз',folk:'Фолк',pop:'Поп',rock:'Рок'},aspects={harmony:'Гармония',rhythm:'Ритм',melody:'Мелодия',structure:'Структура'};
const storageKey='hcm1-ratings:'+DATA.source_receipt_sha256;let ratings={};try{ratings=JSON.parse(localStorage.getItem(storageKey)||'{}')}catch{}
let current=null,version='original',ctx=null,node=null,started=0,offset=0,playing=false,buffers=new Map(),frame=0,request=0,pending=false;
function reasonText(reason){return ({'changed span exceeds four bars':'Изменённый участок занимает больше четырёх тактов — он выходит за рамки этой проверки.','no bounded window preserving masked targets':'Не удалось подготовить допустимое окно с неизменными скрываемыми нотами.','example_wall_budget':'Достигнут предел времени на один пример.','token_budget':'Достигнут предел длины ответа.'})[reason]||reason||'Причина не указана'}
function el(tag,text,cls){const x=document.createElement(tag);if(text!=null)x.textContent=text;if(cls)x.className=cls;return x}
function displayName(c){return c.kind==='pair'?c.work_id:((c.source==='pop909_cl'?'POP909-CL':'HookTheory')+' · пример '+(Number(c.id.split(':')[2])+1))}
function visible(){const needle=$('search').value.toLowerCase();return DATA.cases.filter(c=>($('kind').value==='all'||c.kind===$('kind').value)&&($('subset').value==='all'||($('subset').value==='new'?c.new_example!==false:!!c.prior_rating))&&JSON.stringify([displayName(c),c.work_id,c.genre,genres[c.genre],c.aspect,aspects[c.aspect],c.source,c.description?.title,c.description?.id]).toLowerCase().includes(needle)).sort((a,b)=>Number(b.generation?.status==='completed')-Number(a.generation?.status==='completed'))}
function list(){const rows=visible();$('list').replaceChildren();$('count').textContent='Примеров: '+rows.length;if(!rows.length){stop(true);current=null;$('main').replaceChildren(el('p','Нет примеров для выбранных фильтров.','empty'));return}for(const c of rows){const b=el('button',displayName(c),'item'+(current?.id===c.id?' active':''));b.append(el('small',(aspects[c.aspect]||c.source)+(c.description?.id?' '+c.description.id:'')+' · '+(c.kind==='pair'?(genres[c.genre]||c.genre):('Тактов: '+c.mask_bars))+(c.generation?.status==='completed'?' · есть восстановление HCM':'')+(c.new_example?' · новый':'')+(c.prior_rating?' · ваша оценка сохранена':'')+(c.supported?'':' · ограничение')));b.onclick=()=>select(c);$('list').append(b)}if(rows.length&&!rows.some(c=>c.id===current?.id))select(rows[0])}
function position(){return playing?Math.min(current.duration_seconds,offset+ctx.currentTime-started):offset}
function stop(reset=false){request++;pending=false;if(playing)offset=position();playing=false;if(node){node.onended=null;node.stop();node=null}if(reset)offset=0;cancelAnimationFrame(frame);if($('play'))$('play').textContent='▶ Слушать';draw()}
async function bufferFor(v){const key=current.id+':'+v;if(buffers.has(key))return buffers.get(key);const sr=44100,samples=new Float64Array(Math.round(current.duration_seconds*sr));for(const e of current.audio[v])addGenreNote(samples,sr,e,current.gain);finishGenreSamples(samples,sr);let peak=0;for(const x of samples)peak=Math.max(peak,Math.abs(x));if(peak>=1)throw Error('Перегрузка звука: воспроизведение остановлено. Общую громкость нужно снизить для всех версий.');const b=ctx.createBuffer(1,samples.length,sr);b.copyToChannel(Float32Array.from(samples),0);buffers.set(key,b);return b}
async function play(){if(playing||pending){stop();return}const token=++request;pending=true;try{ctx=ctx||new AudioContext();await ctx.resume();if(token!==request)return;const b=await bufferFor(version);if(token!==request)return;if(offset>=b.duration)offset=0;node=ctx.createBufferSource();node.buffer=b;node.connect(ctx.destination);started=ctx.currentTime;playing=true;node.onended=()=>{if(token===request&&playing){playing=false;offset=0;$('play').textContent='▶ Слушать';draw()}};node.start(0,offset);$('play').textContent='Ⅱ Пауза';tick()}catch(e){if(token===request){$('error').textContent=e.message;stop()}}finally{if(token===request)pending=false}}

async function switchTo(v){const was=playing||pending;stop();version=v;document.querySelectorAll('[data-version]').forEach(b=>b.classList.toggle('active',b.dataset.version===v));draw();if(was)await play()}
function tick(){if(!current)return;const p=position();if($('seek'))$('seek').value=p;if($('time'))$('time').textContent=p.toFixed(1)+' / '+current.duration_seconds.toFixed(1)+' с';draw();if(playing)frame=requestAnimationFrame(tick)}
function draw(){const cv=$('roll');if(!cv||!current?.audio[version])return;const ratio=devicePixelRatio||1,w=cv.clientWidth,h=200;cv.width=w*ratio;cv.height=h*ratio;const g=cv.getContext('2d');g.scale(ratio,ratio);g.clearRect(0,0,w,h);const events=current.audio[version],pitches=events.map(e=>e.pitch),low=Math.min(48,...pitches)-2,high=Math.max(84,...pitches)+2;const x=t=>t/current.duration_seconds*w,y=p=>(high-p)/(high-low)*h;const [a,b]=current.mask_seconds;g.fillStyle='#f5dfb3';g.fillRect(x(a),0,x(b-a),h);g.fillStyle='#6d644d';g.font='11px system-ui';g.fillText('маска',x(a)+5,15);for(let p=low;p<=high;p+=12){g.strokeStyle='#e1e7e2';g.beginPath();g.moveTo(0,y(p));g.lineTo(w,y(p));g.stroke()}for(const e of events){g.fillStyle=e.onset_seconds>=a&&e.onset_seconds<b?'#b97c27':'#44877e';g.fillRect(x(e.onset_seconds),y(e.pitch),Math.max(2,x(e.duration_seconds)),Math.max(2,h/(high-low)*.7))}g.strokeStyle='#172e3a';g.beginPath();g.moveTo(x(position()),0);g.lineTo(x(position()),h);g.stroke()}
function metric(label,value){const d=el('div',label,'metric');d.append(el('b',fmt(value)));return d}
function reveal(c,box){box.replaceChildren();box.append(el('h3','Модель: '+DATA.parent_update+' шагов'));if(c.comparison_score?.supported){const previous=c.comparison_score;const compare=el('table');const head=el('tr');for(const text of ['Шаги',c.kind==='pair'?'Δ NLL':'NLL на событие',c.kind==='pair'?'Δ на событие':''])head.append(el('th',text));compare.append(head);for(const [step,r] of [[DATA.comparison_update,previous],[DATA.parent_update,c.score]]){const tr=el('tr');for(const value of [step,c.kind==='pair'?fmt(r.delta_nll):fmt(r.original?.nats_per_event),c.kind==='pair'?fmt(r.delta_nats_per_event):''])tr.append(el('td',String(value)));compare.append(tr)}box.append(compare)}const score=c.score;if(!score.supported){box.append(el('p','Оценка недоступна: '+reasonText(score.reason||c.reason||'ограничение контекста')));return}const grid=el('div',null,'metrics');grid.append(metric('NLL оригинала ↓',score.original.nll));if(c.kind==='pair'){grid.append(metric('NLL правки ↓',score.variant.nll),metric('Δ = правка − оригинал',score.delta_nll))}else{grid.append(metric('NLL на событие ↓',score.original.nats_per_event),metric('NLL на четверть ↓',score.original.nats_per_qn))}box.append(grid);box.append(el('p',c.kind==='pair'?'Δ > 0: модель предпочитает оригинальное заполнение. Δ < 0: предпочитает правку. Близость к нулю сама по себе не означает одинаковое качество.':'Меньше NLL — выше вероятность исходных нот. Это оценка восстановления, а не красоты музыки.','subtle'));const table=el('table');for(const v of ['original','variant'])if(score[v]){const tr=el('tr');for(const x of [labels[v],score[v].target_events+' событий',fmt(score[v].nats_per_event)+' на событие',fmt(score[v].nats_per_qn)+' на четверть'])tr.append(el('td',x));table.append(tr)}box.append(table);if(c.hcm_training_overlap)box.append(el('p',c.hcm_training_overlap.known_overlap?'Есть известное пересечение с HCM TRAIN. Этот пример не подтверждает обобщение на новую музыку.':'Известное совпадение с HCM TRAIN не найдено. Межкорпусные дубликаты и знакомство энкодера C3 с произведением полностью не исключены.','tiny subtle'))}
function select(c){stop(true);current=c;version='original';buffers.clear();const main=$('main');main.replaceChildren();const top=el('div',null,'topline');top.append(el('span',c.kind==='pair'?(aspects[c.aspect]||c.aspect):'Восстановление','tag'),el('span',genres[c.genre]||c.genre||c.source,'subtle'));main.append(top,el('h2',displayName(c)));const audio=el('section',null,'panel');if(Object.keys(c.audio).length){audio.append(el('p','Золотым выделен участок, скрытый от модели. При переключении сохраняется позиция.','subtle tiny'));const cv=el('canvas');cv.id='roll';audio.append(cv);const seek=el('input');seek.id='seek';seek.type='range';seek.min=0;seek.max=c.duration_seconds;seek.step=.05;seek.value=0;seek.oninput=()=>{const was=playing;stop();offset=Number(seek.value);draw();if(was)play()};audio.append(seek);const controls=el('div',null,'controls');const pb=el('button','▶ Слушать','primary');pb.id='play';pb.onclick=play;controls.append(pb);for(const v of Object.keys(c.audio)){const b=el('button',labels[v],v===version?'active':'');b.dataset.version=v;b.onclick=()=>switchTo(v);controls.append(b)}const only=el('button','К началу маски');only.onclick=()=>{stop();offset=c.mask_seconds[0];play()};controls.append(only);audio.append(controls);const clock=el('span','0.0 / '+c.duration_seconds.toFixed(1)+' с','tiny subtle');clock.id='time';audio.append(clock)}else audio.append(el('p','Аудио для этого слота недоступно: '+reasonText(c.audio_reason)));const error=el('p');error.id='error';audio.append(error);main.append(audio);
if(c.prior_rating){const names={better:'лучше',worse:'хуже',same:'так же',unsure:'не уверен'};main.append(el('p','Ваша прошлая оценка этой пары: «'+names[c.prior_rating.rating]+'». Она сохранена отдельно от нового просмотра.','subtle'))}if(c.kind==='pair'){const description=el('section',null,'panel');description.append(el('h3','Что изменено'));const d=c.description||{};for(const key of ['title','rationale','description','uncertainty'])if(d[key])description.append(el('p',String(d[key])));description.append(el('p','Изменённых нот: '+(d.patches?.length||0)+'. Ожидаемое ухудшение — предположение автора правки; ваше мнение может отличаться.','subtle tiny'));main.append(description)}
if(c.generation){const g=el('p',c.generation.status==='completed'?'Восстановление HCM доступно в переключателе. Динамика сгенерированных нот задана одинаковой; модель её не предсказывает.':'Восстановление HCM не завершено: '+reasonText(c.generation.reason||c.generation.status),'subtle');main.append(g)}
if(c.kind==='pair'||(c.generation?.status==='completed'&&c.audio.generated)){const rate=el('section',null,'panel');rate.append(el('h3',c.kind==='pair'?'Как правка повлияла на музыку?':'Насколько убедительно звучит восстановление?'));const buttons=el('div',null,'ratings');const options=c.kind==='pair'?[['better','Лучше'],['worse','Хуже'],['same','Так же'],['unsure','Не уверен']]:[['plausible','Убедительно'],['implausible','Неубедительно'],['unsure','Не уверен']];for(const [v,label] of options){const b=el('button',label,ratings[c.id]?.rating===v?'active':'');b.onclick=()=>{ratings[c.id]={rating:v,at:new Date().toISOString(),score_revealed:!!c.revealed};try{localStorage.setItem(storageKey,JSON.stringify(ratings))}catch{}buttons.querySelectorAll('button').forEach(x=>x.classList.toggle('active',x===b))};buttons.append(b)}rate.append(buttons);main.append(rate);}else{main.append(el('p','Генерация для этого блока не проводилась или не завершена. Оценивать восстановление пока нечего.','subtle'));}const scores=el('section',null,'panel');const revealButton=el('button','Показать оценку HCM');revealButton.onclick=()=>{c.revealed=true;reveal(c,scores)};scores.append(revealButton);main.append(scores);const raw=el('details');raw.append(el('summary','Подробности примера и ограничения'));raw.append(el('pre',JSON.stringify({id:c.id,work_id:c.work_id,query:c.query,reason:c.reason,context_sha256:c.context_sha256,overlap:c.hcm_training_overlap,description:c.description},null,2)));main.append(raw);list();draw()}
$('kind').onchange=list;$('subset').onchange=list;$('search').oninput=list;window.addEventListener('resize',draw);$('export').onclick=()=>{const blob=new Blob([JSON.stringify({schema:'HCM1HumanListeningRatings@1',source_receipt_sha256:DATA.source_receipt_sha256,ratings},null,2)],{type:'application/json'});const a=el('a');a.href=URL.createObjectURL(blob);a.download='hcm1-listening-ratings.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
const summary=el('table');for(const split of ['train','validation','campaign_head_dev']){const rows=DATA.all_scores.filter(r=>r.split===split),ok=rows.filter(r=>r.supported);const tr=el('tr');tr.append(el('th',split),el('td',ok.length+' / '+rows.length+' поддержаны'));summary.append(tr)}$('summary').append(summary);if(DATA.prior_rating_count)$('summary').append(el('p','Сохранено прошлых оценок: '+DATA.prior_rating_count+'. Их можно найти в фильтре «Мои прошлые оценки».'));list();
</script></html>'''


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cases',required=True);p.add_argument('--receipt',required=True);p.add_argument('--record',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();from music_critic.pll.cli import _admit_output
    _admit_output(a.output);out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:f.write(page(payload(a.cases,a.receipt,a.record)))
