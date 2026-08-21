import logging

from esm.sdk.api import SamplingConfig
from lightning import LightningModule

from foldmatch.utils.profiling import cuda_profile_stage

logger = logging.getLogger(__name__)

class EsmModule(LightningModule):

    def __init__(
            self,
            model,
            nvtx: bool = False,
            profile_memory: bool = False,
    ):
        super().__init__()
        self.esm3 = model
        self.nvtx = nvtx
        self.profile_memory = profile_memory

    def on_predict_start(self):
        logger.info(f"ESM device: {self.device} | {self.trainer.global_rank + 1}/{self.trainer.world_size}")
        if (self.nvtx or self.profile_memory) and self.device.type != "cuda":
            logger.warning("CUDA profiling requested, but the ESM module is not running on CUDA")

    def predict_step(self, prot_batch, batch_idx):
        embeddings = []
        names = []
        for sample_idx, (esm_prot, name) in enumerate(prot_batch):
            embeddings.append(
                self.__compute_embeddings(esm_prot, str(name), batch_idx, sample_idx)
            )
            names.append(name)
        return tuple(embeddings), tuple(names)

    def __compute_embeddings(self, esm_prot, name, batch_idx, sample_idx):
        label = f"batch={batch_idx} sample={sample_idx} id={name}"
        with cuda_profile_stage(
                f"esm.encode {label}",
                nvtx=self.nvtx,
                memory=self.profile_memory,
                logger=logger,
        ):
            encoded = self.esm3.encode(esm_prot)

        with cuda_profile_stage(
                f"esm.forward_and_sample {label}",
                nvtx=self.nvtx,
                memory=self.profile_memory,
                logger=logger,
        ):
            output = self.esm3.forward_and_sample(
                encoded,
                SamplingConfig(return_per_residue_embeddings=True),
            )
        return output.per_residue_embedding
