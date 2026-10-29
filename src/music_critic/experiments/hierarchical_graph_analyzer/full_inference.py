"""Novel unlabelled canonical/MIDI inference, without numerical theory exports."""
from hashlib import sha256
import io
import time
from pathlib import Path

import torch

from .cache import (BASE_NODES, LIMITS, RawGraphProfile, assemble_raw_tile,
                    attach_global_bars, pair)


def encode_raw_work(piece, encoder, *, device='cpu', wall_seconds, limits=None):
    """Use frozen C3 only as an observed-note encoder; retain all physical IDs.

    No unsupported tile is silently omitted. One exceeded bound rejects the
    whole piece, so downstream predictions never imply complete missing music.
    This seam accepts a native encoder for fixture tests; deployment verifies its
    checkpoint using PLLContextEncoder.from_pretrained.
    """
    from music_critic.relation_probe.data import sanitize
    from music_critic.pll.contracts import MaskQuery
    from music_critic.pll.graph import build_tile_graph
    from music_critic.pll.masking import build_observed_tiles
    from music_critic.pll.observed_view import ObservedMusicView
    from .full_data import assemble_work_raw
    if not isinstance(wall_seconds,(int,float)) or wall_seconds<=0:raise ValueError('positive raw encoding wall cap required')
    start_clock=time.monotonic();piece=sanitize(piece)
    if not piece.bars:raise ValueError('raw inference requires observed meter bars')
    if len(piece.bars)>LIMITS['max_global_bars']:raise ValueError('raw inference global_bar_capacity')
    bounds=dict(limits or {})
    if set(bounds)-{'max_tiles','max_notes'}:raise ValueError('unknown whole raw capacity')
    if len(piece.notes)>bounds.get('max_notes',200000):raise ValueError('raw inference max_notes')
    profile=RawGraphProfile(context_bars=max(32,len(piece.bars)))
    start=piece.bars[0].start_qn;end=piece.bars[-1].start_qn+piece.bars[-1].duration_qn
    view=ObservedMusicView(piece,MaskQuery(start,end,tuple(t.track_id for t in piece.tracks)),tuple(range(len(piece.bars))))
    tiles=build_observed_tiles(view,profile)
    if len(tiles)>bounds.get('max_tiles',2048):raise ValueError('raw inference max_tiles')
    previous=encoder.training;encoder.eval();payloads=[];contexts={}
    try:
        with torch.no_grad():
            for tile in tiles:
                if time.monotonic()-start_clock>=wall_seconds:raise TimeoutError('raw inference encoding wall cap')
                graph=build_tile_graph(tile,profile)
                flags=torch.tensor(tile.boundary_flags,dtype=torch.float32,device=device).reshape(-1,2)
                states=encoder._encode_graph(graph.to(device),flags)
                states={name:value.detach().cpu() for name,value in states.items()}
                for i,(_,identity) in enumerate(graph.keys['bar']):
                    if identity in tile.core_bar_ids:contexts[identity]=states['bar'][i]
                payloads.append(assemble_raw_tile(piece,tile,graph,states))
    finally:encoder.train(previous)
    ids=tuple(b.bar_id for b in piece.bars)
    if set(ids)!=set(contexts):raise ValueError('raw inference incomplete C3 bar coverage')
    context=dict(ids=ids,states=torch.stack([contexts[i] for i in ids]),
        spans=[[pair(b.start_qn.to_fraction()),pair((b.start_qn+b.duration_qn).to_fraction())] for b in piece.bars])
    raw=assemble_work_raw(piece,((attach_global_bars(p,context),p['metadata']) for p in payloads),**bounds)
    raw['coverage']['raw_encoding_wall_seconds']=time.monotonic()-start_clock
    return raw


def _verified_bytes(path, expected_sha256, expected_size_bytes, *, max_bytes):
    path=Path(path)
    if path.is_symlink() or not path.is_file():raise ValueError('regular non-symlink input required')
    if type(expected_size_bytes) is not int or expected_size_bytes<1 or expected_size_bytes>max_bytes:raise ValueError('explicit bounded input size required')
    if path.stat().st_size!=expected_size_bytes:raise ValueError('input size mismatch')
    value=path.read_bytes()
    if sha256(value).hexdigest()!=expected_sha256:raise ValueError('input SHA256 mismatch')
    return value


def load_full_weights(model,path,*,expected_sha256,expected_size_bytes,bindings):
    """Authenticated weight-only inference load. No optimizer or RNG mutation."""
    from .runtime import _model_signature
    from .full_runtime import full_checkpoint_bindings
    blob=_verified_bytes(path,expected_sha256,expected_size_bytes,max_bytes=2*1024**3)
    payload=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=True)
    if payload.get('schema')!='HiergraphCheckpoint@1' or payload.get('bindings')!=full_checkpoint_bindings(model,bindings):raise ValueError('full inference checkpoint bindings mismatch')
    if payload.get('model_signature')!=_model_signature(model):raise ValueError('full inference checkpoint architecture mismatch')
    old=model.state_dict();state=payload.get('model',{})
    if set(state)!=set(old):raise ValueError('full inference checkpoint state keys mismatch')
    for name,value in state.items():
        if not isinstance(value,torch.Tensor) or value.shape!=old[name].shape or value.dtype!=old[name].dtype or not bool(torch.isfinite(value).all()):raise ValueError('full inference checkpoint tensor mismatch')
    model.load_state_dict(state,strict=True)
    return dict(sha256=expected_sha256,size_bytes=expected_size_bytes,optimizer_restored=False)


def load_raw_input(source):
    """Authenticate one raw source before dispatching the existing importer."""
    from music_critic.data import loads_piece
    blob=_verified_bytes(source['path'],source['sha256'],source['size_bytes'],max_bytes=128*1024**2)
    if source['format']=='canonical_json':return loads_piece(blob)
    if source['format']=='midi':
        from music_critic.adapters.midi import MidiAdapterConfig,load_midi_piece
        piece=load_midi_piece(source['path'],config=MidiAdapterConfig(dataset_name='hiergraph_unlabelled_inference'))
        parsed=[p.checksum_sha256 for p in piece.provenance if p.kind=='source' and p.source=='standard_midi_file']
        if parsed!=[source['sha256']]:raise ValueError('parsed MIDI source differs from verified input SHA')
        return piece
    raise ValueError('canonical_json or midi input required')


def run_full_inference(config,output,*,seed=17):
    """Bounded registered inference on a canonical JSON or MIDI source only."""
    from music_critic.data import loads_piece
    from music_critic.pll.encoder import PLLContextEncoder
    from music_critic.experiments.harmonic_event_reasoner.evidence import atomic_json
    from .full_runtime import make_full_model,infer_full,validate_full_config
    validate_full_config(config,'infer');out=Path(output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    torch.set_num_threads(config.get('cpu_threads',1));torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
    try:
        piece=load_raw_input(config['input'])
        c3=config['c3_checkpoint']
        encoder=PLLContextEncoder.from_pretrained(c3['path'],expected_sha256=c3['sha256'],expected_size_bytes=c3['size_bytes'],freeze_policy='frozen').to(config['device'])
        raw=encode_raw_work(piece,encoder,device=config['device'],wall_seconds=config['wall_seconds']-(time.monotonic()-start),limits=config.get('raw_limits'))
        del encoder
        model=make_full_model(config['head_sizes'],config,seed)
        weights=load_full_weights(model,**config['model_checkpoint'])
        if time.monotonic()-start>=config['wall_seconds']:raise TimeoutError('inference wall cap before whole graph')
        prediction=infer_full(model,raw,work_id=config['work_id'],field_values=config.get('field_values'),edge_threshold=config.get('edge_threshold',.5))
        atomic_json(out/'prediction.json',prediction)
        result=dict(status='completed',summary='Unlabelled whole-piece inference completed; no target bundle read.',work_id=config['work_id'],source_piece_id=piece.piece_id,weights=weights,wall_seconds=time.monotonic()-start,coverage=prediction['coverage'],test_locked=True)
    except BaseException as error:
        result=dict(status='aborted' if isinstance(error,KeyboardInterrupt) else 'failed',summary='Whole-piece inference did not complete.',error=f'{type(error).__name__}: {error}',wall_seconds=time.monotonic()-start)
        atomic_json(out/'result.json',result);raise
    atomic_json(out/'result.json',result);return result
