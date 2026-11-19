import logging

from lightning import LightningModule

from foldmatch.utils.profiling import cuda_profile_stage, shape_label

logger = logging.getLogger(__name__)

class ChainModule(LightningModule):

    def __init__(
            self,
            model,
            nvtx: bool = False,
            profile_memory: bool = False,
            profile_stage: str = "chain",
    ):
        super().__init__()
        self.aggregator = model
        self.nvtx = nvtx
        self.profile_memory = profile_memory
        self.profile_stage = profile_stage

    def on_train_start(self):
        logger.info(f"Aggregator device: {self.device} | {self.trainer.global_rank + 1}/{self.trainer.world_size}")

    def on_predict_start(self):
        logger.info(f"Aggregator device: {self.device} | {self.trainer.global_rank + 1}/{self.trainer.world_size}")
        if (self.nvtx or self.profile_memory) and self.device.type != "cuda":
            logger.warning("CUDA profiling requested, but the aggregator module is not running on CUDA")

    def predict_step(self, batch, batch_idx):
        (x, x_mask), dom_id = batch
        label = f"{self.profile_stage}.aggregator batch={batch_idx} shape={shape_label(x)}"
        with cuda_profile_stage(
                label,
                nvtx=self.nvtx,
                memory=self.profile_memory,
                logger=logger,
        ):
            embeddings = self.aggregator(x, x_mask)
        return embeddings, dom_id
