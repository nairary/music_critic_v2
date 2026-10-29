"""Native CUDA-only synthetic gradient/replay contract, no corpus or quality claim."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import tempfile
import unittest
import torch
from music_critic.experiments.hierarchical_graph_analyzer.model import HAnalyzer
from music_critic.experiments.hierarchical_graph_analyzer.runtime import initialize_shared_parameters,train_steps,save_checkpoint,load_checkpoint
from test_hiergraph_model import synthetic_raw

@unittest.skipUnless(torch.cuda.is_available(),'native CUDA required')
class CudaReplayTests(unittest.TestCase):
    def test_all_five_exact_optimizer_replay_on_synthetic_cuda(self):
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        target=dict(ids=['h'],spans=[[0.,.5]],fields={'H.function':dict(values=[1],mask=[True])},
                    work_id='synthetic',split='fit',view_id='v',scope_id='harmonies',core_span=[0,1],halo_span=[0,1],known_boundary_indices=[1])
        samples=[(synthetic_raw(),target)];bindings=dict(source_sha256='1'*64,config_sha256='2'*64,data_sha256='3'*64)
        for kind in ('E1','E2','E3','E4','E5'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as directory:
                def make():
                    m=HAnalyzer(kind,{'H.function':3},input_width=12,hidden=16,rounds=2,max_events=3).cuda()
                    initialize_shared_parameters(m,17)
                    return m,torch.optim.Adam(m.parameters(),lr=.0003)
                m,o=make();first=train_steps(m,samples,o,steps=2,seed=17)
                path=Path(directory)/'state.pt';b=save_checkpoint(path,m,o,first['state'],bindings=bindings)
                final=train_steps(m,samples,o,steps=2,seed=17,state=first['state'])
                r,ro=make();state=load_checkpoint(path,r,ro,bindings=bindings,expected_sha256=b['sha256'])
                replay=train_steps(r,samples,ro,steps=2,seed=17,state=state)
                self.assertEqual(final['state'],replay['state'])
                self.assertEqual(final['state']['applied_updates'],4)
                for name,value in m.state_dict().items():torch.testing.assert_close(value,r.state_dict()[name],rtol=0,atol=0)
                del m,o,r,ro
        torch.cuda.empty_cache()
