import hydra
import sys
sys.path.append("src")
from omegaconf import DictConfig, OmegaConf
import logging
import submitit
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@hydra.main(config_path = "configs", 
            config_name = "construct", 
            version_base = None)
def main(config: DictConfig):
    config    = OmegaConf.to_container(config, resolve = True)
    
    ## run the PLM
    logging.info(f"Generating embeddings: save at {config['embeddings']['outputdir']}")
    hydra.utils.instantiate(config["embeddings"])
    
if __name__ == "__main__":
    main()