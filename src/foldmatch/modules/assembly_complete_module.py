import logging

from lightning import LightningModule
from torch import cat

from foldmatch.utils.data import collate_seq_embeddings
from foldmatch.utils.esm.loaders import per_residue_embedding
from foldmatch.utils.model import release_mps_cache
from foldmatch.utils.profiling import cuda_profile_stage, esm_profile_kwargs, shape_label

logger = logging.getLogger(__name__)

class AssemblyCompleteModule(LightningModule):

    def __init__(
            self,
            res_model,
            aggregator_model,
            max_res_n=0,
            nvtx: bool = False,
            profile_memory: bool = False,
    ):
        super().__init__()
        self.esm3 = res_model
        self.aggregator =  aggregator_model
        self.max_res_n = max_res_n
        self.nvtx = nvtx
        self.profile_memory = profile_memory

    def on_predict_start(self):
        logger.info(f"ESM + Aggregator device: {self.device} | {self.trainer.global_rank + 1}/{self.trainer.world_size}")
        if (self.nvtx or self.profile_memory) and self.device.type != "cuda":
            logger.warning("CUDA profiling requested, but the ESM + aggregator module is not running on CUDA")

    def predict_step(self, prot_batch, batch_idx):
        assembly_embeddings = []
        assembly_names = []
        for esm_chains, name in prot_batch:
            prot_embeddings = []
            assembly_n_res = 0
            for chain_idx, esm_prot in enumerate(esm_chains):
                embeddings = per_residue_embedding(self.esm3, esm_prot, **esm_profile_kwargs(
                    nvtx=self.nvtx,
                    profile_memory=self.profile_memory,
                    label=f"batch={batch_idx} id={name} chain={chain_idx}",
                    logger=logger,
                ))
                prot_embeddings.append(embeddings)
                assembly_n_res += embeddings.shape[0]
                if 0 < self.max_res_n < assembly_n_res:
                    break
            assembly_embeddings.append(cat(prot_embeddings, dim=0))
            assembly_names.append(name)
        res_batch_embedding, res_batch_mask = collate_seq_embeddings(assembly_embeddings)
        with cuda_profile_stage(
                f"assembly.aggregator batch={batch_idx} shape={shape_label(res_batch_embedding)}",
                nvtx=self.nvtx,
                memory=self.profile_memory,
                logger=logger,
        ):
            assembly_batch_embedding = self.aggregator(res_batch_embedding, res_batch_mask)
        release_mps_cache(self.device)
        return assembly_batch_embedding, tuple(assembly_names)