import hydra
from omegaconf import DictConfig, OmegaConf



@hydra.main(config_path="configs", config_name="get_candidates", version_base = None)
def main(config: DictConfig):
    config = OmegaConf.to_container(config, resolve = True)
    