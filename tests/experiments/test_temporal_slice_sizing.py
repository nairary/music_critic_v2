from types import SimpleNamespace
import pytest
import torch
from scripts.size_temporal_graphs import train_rows, dimensions


def test_test_gate_precedes_metadata_read():
    with pytest.raises(ValueError, match='TEST forbidden'):
        train_rows({'records':[{'split':'test'}]})


def test_size_counts_tensors_and_relation_multiplicity():
    class Base:
        stores=[{'x':torch.zeros(2,3), 'text':'metadata'}]
        node_types=['note']
        edge_types=[('note','next','note')]
        def __getitem__(self, key):
            return SimpleNamespace(num_nodes=2) if key=='note' else SimpleNamespace(edge_index=torch.tensor([[0],[1]]))
    graph=SimpleNamespace(base=Base(),slice_features=torch.zeros(3,14),keys={'slice':range(3)},
                          edges={('note','starts','slice'):torch.tensor([[0,1],[0,2]])})
    result=dimensions(graph)
    assert result['slice_nodes']==3 and result['extension_edges']==2 and result['base_edges']==1
    assert result['total_tensor_bytes']==24+168+32
