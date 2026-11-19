from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from typer.testing import CliRunner

from foldmatch.cli.embedding import app as embedding_app
from foldmatch.cli.inference import app
from foldmatch.modules.assembly_complete_module import AssemblyCompleteModule
from foldmatch.modules.chain_complete_module import ChainCompleteModule
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
        "foldmatch.utils.profiling.cuda_profile_stage",
        record_stage,
    )

    class FakeEsm:
        def encode(self, protein):
            return "encoded"

        def forward_and_sample(self, encoded, config):
            return SimpleNamespace(per_residue_embedding=torch.ones(3, 4))

    module = EsmModule(FakeEsm(), nvtx=True, profile_memory=True)
    # With coordinates, so ESM3's geometric attention is left on (no transformer blocks touched).
    protein = SimpleNamespace(coordinates=torch.zeros(1, 37, 3))
    embeddings, names = module.predict_step([(protein, "sample-id")], 7)

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


class _FakeEsm:
    """ESM3 stand-in: per-residue embedding of ``protein.n`` rows plus BOS and EOS."""

    def encode(self, protein):
        return protein

    def forward_and_sample(self, encoded, config):
        return SimpleNamespace(per_residue_embedding=torch.ones(encoded.n + 2, 4))


class _FakeAggregator(torch.nn.Module):
    def forward(self, x, mask):
        return x.sum(dim=1)


def _protein(n):
    # With coordinates, so ESM3's geometric attention is left on (no transformer blocks touched).
    return SimpleNamespace(n=n, coordinates=torch.zeros(n, 37, 3))


@pytest.fixture
def recorded_stages(monkeypatch):
    stages = []

    @contextmanager
    def record_stage(name, **kwargs):
        stages.append((name, kwargs))
        yield

    # per_residue_embedding imports it lazily; the complete modules import it at module level.
    monkeypatch.setattr("foldmatch.utils.profiling.cuda_profile_stage", record_stage)
    monkeypatch.setattr("foldmatch.modules.chain_complete_module.cuda_profile_stage", record_stage)
    monkeypatch.setattr("foldmatch.modules.assembly_complete_module.cuda_profile_stage", record_stage)
    return stages


def test_chain_complete_module_profiles_esm_and_aggregator(recorded_stages):
    module = ChainCompleteModule(_FakeEsm(), _FakeAggregator(), nvtx=True, profile_memory=True)
    embeddings, names = module.predict_step([(_protein(3), "1ABC.A"), (_protein(5), "1ABC.B")], 2)

    assert embeddings.shape == (2, 4)
    assert names == ("1ABC.A", "1ABC.B")
    assert [name for name, _ in recorded_stages] == [
        "esm.encode batch=2 sample=0 id=1ABC.A",
        "esm.forward_and_sample batch=2 sample=0 id=1ABC.A",
        "esm.encode batch=2 sample=1 id=1ABC.B",
        "esm.forward_and_sample batch=2 sample=1 id=1ABC.B",
        "chain.aggregator batch=2 shape=2x7x4",
    ]
    assert all(options["nvtx"] and options["memory"] for _, options in recorded_stages)


def test_assembly_complete_module_profiles_each_chain_and_aggregator(recorded_stages):
    module = AssemblyCompleteModule(_FakeEsm(), _FakeAggregator(), nvtx=True, profile_memory=True)
    embeddings, names = module.predict_step([([_protein(3), _protein(4)], "3J8H-1")], 0)

    assert embeddings.shape == (1, 4)
    assert names == ("3J8H-1",)
    assert [name for name, _ in recorded_stages] == [
        "esm.encode batch=0 id=3J8H-1 chain=0",
        "esm.forward_and_sample batch=0 id=3J8H-1 chain=0",
        "esm.encode batch=0 id=3J8H-1 chain=1",
        "esm.forward_and_sample batch=0 id=3J8H-1 chain=1",
        "assembly.aggregator batch=0 shape=1x11x4",
    ]


def test_complete_modules_do_not_profile_by_default(recorded_stages):
    ChainCompleteModule(_FakeEsm(), _FakeAggregator()).predict_step([(_protein(3), "a")], 0)
    AssemblyCompleteModule(_FakeEsm(), _FakeAggregator()).predict_step([([_protein(3)], "b")], 0)

    # The aggregator ranges are entered (as no-ops); the ESM path skips profiling entirely.
    assert [name.split()[0] for name, _ in recorded_stages] == ["chain.aggregator", "assembly.aggregator"]
    assert not any(options["nvtx"] or options["memory"] for _, options in recorded_stages)


@pytest.mark.parametrize("command", [
    ["from-structures", "residue"],
    ["from-structures", "chain"],
    ["from-structures", "assembly"],
    ["from-sequences", "residue"],
    ["from-sequences", "chain"],
])
def test_fm_embedding_help_lists_profiling_flags(command):
    result = CliRunner().invoke(embedding_app, [*command, "--help"])

    assert result.exit_code == 0
    assert "--nvtx" in result.stdout
    assert "--profile-memory" in result.stdout
