import esm
import torch
from Bio import SeqIO
from tqdm import tqdm
from glob import glob
import h5py 

def save_esm_embeddings(recordsfile, outputdir, esm, device = 0):
    """
    Takes in a fasta file of sequences, and saves the embedding of 
    individual sequence in a separate torch-save file. 
    
    Works for both esm2 and esm1v
    """
    esmmodel, esmalph = esm
    esmmodel          = esmmodel.to(device)
    bc                = esmalph.get_batch_converter()
    records = list(SeqIO.parse(recordsfile, "fasta"))
    with torch.no_grad():
        for rec in tqdm(records, desc = "generating ESM embedding"):
            seq  = str(rec.seq)
            name = rec.id
            data = [(name, seq)]
            _, _, batch_tokens = bc(data)
            emb = esmmodel(batch_tokens.to(device), repr_layers=[33], 
                           return_contacts=True)["representations"][33]
            with h5py.File(f"{outputdir}/{name}.h5", "w") as hf:
                hf.create_dataset(name, 
                                  data = emb[:, 1:-1, :].cpu().numpy())
        
def _compute_raygun_embedding(esmfile, 
                             outputdir, 
                             raymodel, 
                             device):
    """
    Unlike above, it takes in one torch-file of the esm embedding, 
    uses it to compute the raygun embedding and saves it to a new file.
    
    Takes in the raymodel as input.
    """
    name = esmfile.split("/")[-1]
    if os.path.exists(f"{outputdir}/{name}"):
        return 
    embedding = torch.load(esmfile)
    embed     = raymodel.encoder(embedding.to(device))
    with h5py.File(f"{outputdir}/{name}", "w") as hf:
        hf.create_dataset(name.split(".")[0], 
                          data = embed[:, 1:-1, :].cpu().numpy())
    return 

def compute_raygun_embeddings(esmdir, 
                             outputdir, 
                             ray, 
                             device = 0):
    """
    takes in a directory of esm-2 embeddings and utilizes it to 
    compute the raygun embedding.
    """
    raymodel, _, _ = ray
    raymodel       = raymodel.to(device)
    esmfiles       = glob(f"{esmdir}/*.sav")
    for f in tqdm(esmfiles, desc = "Computing Raygun embeddings"):
        _compute_raygun_embedding(f, outputdir, raymodel, device)

class EmbeddingData(torch.utils.data.Dataset):
    def __init__(self, embeddingfolder):
        self.embfiles = glob(f"{embeddingfolder}/*.h5")
        return 
    
    def __len__(self):
        return len(self.embfiles)
    
    def __getitem__(self, idx):
        with h5py.File(self.embfiles[idx], "r") as hf:
            name = list(hf.keys())[0]
            emb  = torch.tensor(hf.get(name)[:]).squeeze()
        return name, emb