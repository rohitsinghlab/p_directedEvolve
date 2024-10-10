import sys
sys.path.append("src")
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import functools
from Bio import SeqIO, Seq, SeqRecord

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@hydra.main(config_path="configs", config_name="get_candidates", version_base = None)
def main(config: DictConfig):    
    logging.info("Getting the pandas file.")
    config            = OmegaConf.to_container(config, resolve = True)
    no_mutations      = config["candidates"]["no_mutations"]
    originalseq       = str(list(hydra.utils.instantiate(config["1jpl_sequence"]))[0].seq)
        
    logging.info("Getting the candidates.")
    greedy_candidates = hydra.utils.instantiate(config["candidates"])
    greedy_candidates.to_csv(f"{config['outputurl']}.tsv", sep = "\t", index = None)
    
    mutationcols = functools.reduce(lambda x, y: x + y,
                                    [[f"location_{i}", 
                                      f"residue_{i}",
                                      f"phenotype_{i}"]
                                    for i in range(no_mutations)
                                    ])
    records = []
    for entry in greedy_candidates[mutationcols].values:
        seqlist  = list(originalseq)
        entryn   = "1jpl"
        allpheno = 0
        for i in range(no_mutations):
            loc, res, pheno = entry[3*i], entry[3*i+1], entry[3*i+2]
            seqlist[int(loc)-1]  = res
            entryn         += f":{loc}-{res}"
            allpheno       += pheno
        record = SeqRecord.SeqRecord(id = entryn, 
                          name          = entryn,
                          seq           = Seq.Seq("".join(seqlist)),
                          description   = f"phenotype_sum={allpheno}")
        records.append(record)
    SeqIO.write(records, f"{config['outputurl']}.fasta", "fasta")
    return


if __name__ == "__main__":
    main()
