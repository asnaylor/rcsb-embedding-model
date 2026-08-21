from contextlib import contextmanager
from types import SimpleNamespace

import torch
from typer.testing import CliRunner

from foldmatch.cli.inference import app
from foldmatch.modules.chain_module import ChainModule
from foldmatch.modules.esm_module import EsmModule
from foldmatch.utils.profiling import cuda_profile_stage


def test_cuda_profile_stage_is_noop_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    executed = False

    with cuda_profile_stage("cpu-stage", nvtx=True, memory=True):
        executed = True

    assert executed


def test_esm_module_profiles_encode_and_forward(monkeypatch):
    stages = []

    @contextmanager
    def record_stage(name, **kwargs):
        stages.append((name, kwargs))
        yield

    monkeypatch.setattr(
        "foldmatch.modules.esm_module.cuda_profile_stage",
        record_stage,
    )

    class FakeEsm:
        def encode(self, protein):
            return f"encoded-{protein}"

        def forward_and_sample(self, encoded, config):
            return SimpleNamespace(per_residue_embedding=torch.ones(3, 4))

    module = EsmModule(FakeEsm(), nvtx=True, profile_memory=True)
    embeddings, names = module.predict_step([("protein", "sample-id")], 7)

    assert embeddings[0].shape == (3, 4)
    assert names == ("sample-id",)
    assert [name for name, _ in stages] == [
        "esm.encode batch=7 sample=0 id=sample-id",
        "esm.forward_and_sample batch=7 sample=0 id=sample-id",
    ]
    assert all(options["nvtx"] for _, options in stages)
    assert all(options["memory"] for _, options in stages)


def test_chain_module_profiles_aggregator_with_shape(monkeypatch):
    stages = []

    @contextmanager
    def record_stage(name, **kwargs):
        stages.append((name, kwargs))
        yield

    monkeypatch.setattr(
        "foldmatch.modules.chain_module.cuda_profile_stage",
        record_stage,
    )

    class FakeAggregator(torch.nn.Module):
        def forward(self, x, mask):
            return x.sum(dim=1)

    module = ChainModule(
        FakeAggregator(),
        nvtx=True,
        profile_memory=True,
        profile_stage="assembly",
    )
    x = torch.ones(2, 5, 4)
    mask = torch.zeros(2, 5, dtype=torch.bool)
    embeddings, names = module.predict_step(((x, mask), ("a", "b")), 3)

    assert embeddings.shape == (2, 4)
    assert names == ("a", "b")
    assert stages[0][0] == "assembly.aggregator batch=3 shape=2x5x4"


def test_complete_embedding_help_lists_profiling_flags():
    result = CliRunner().invoke(app, ["complete-embedding", "--help"])

    assert result.exit_code == 0
    assert "--nvtx" in result.stdout
    assert "--profile-memory" in result.stdout
