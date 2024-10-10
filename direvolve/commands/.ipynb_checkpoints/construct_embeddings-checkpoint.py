import hydra
import sys
sys.path.append("src")
from omegaconf import DictConfig, OmegaConf
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@hydra.main(config_path = "configs", 
            config_name = "construct", 
            version_base = None)
def main(config: DictConfig):
    logging.info("Generating embedding...")
    config    = OmegaConf.to_container(config, resolve = True)
    ## run the PLM
    hydra.utils.instantiate(config["embeddings"])
    
if __name__ == "__main__":
    main()