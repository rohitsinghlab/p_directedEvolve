import hydra
import sys
sys.path.append("src")
from omegaconf import DictConfig, OmegaConf
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def load_state_dict(model, filename):
    model.load_state_dict(filename)
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
            pairs    += list(zip(names.tolist(),
                                 labels.squeeze().cpu().numpy().tolist()))
    df = pd.DataFrame(pairs, columns = ["name", "predictions"])
    df.to_csv(config["outputfile"], sep = "\t", index = None)
    
if __name__ == "__main__":
    main()