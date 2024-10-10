import hydra
import sys
sys.path.append("src")
from omegaconf import DictConfig, OmegaConf
import logging
import os
import torch
from tqdm import tqdm
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def load_state_dict(model, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["model-state-dict"])
    del checkpoint
    return model

@hydra.main(config_path = "configs", 
            config_name = "predict", 
            version_base = None)
def main(config: DictConfig):
    config    = OmegaConf.to_container(config, resolve = True)
    
    ## run the PLM
    logging.info(f"Getting model, dataloader...")
    loader = hydra.utils.instantiate(config["loader"])
    model  = hydra.utils.instantiate(config["model"]["inst"]).to(config["device"])
    
    pairs  = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc = "Predicting: ")):
            names, xs = batch
            labels    = model(xs.to(config["device"]))
            pairs    += list(zip(list(names),
                                 labels.squeeze().cpu().numpy().tolist()))
    df = pd.DataFrame(pairs, columns = ["name", "predictions"])
    df.to_csv(config["model"]["outputfile"], sep = "\t", index = None)
    
if __name__ == "__main__":
    main()