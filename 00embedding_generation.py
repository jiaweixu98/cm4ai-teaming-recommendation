import json
import os
from tqdm import tqdm
import torch
import numpy as np
from transformers import AutoTokenizer
from adapters import AutoAdapterModel


save_dir = '/data/jx4237data/TKG/new_TKG/TKG_new_BAI'

# Open the file and load the JSON data
with open(os.path.join(save_dir, 'paper_nodes.json'), 'r') as file:
    data = json.load(file)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_aug2023refresh_base')

# Load base model
model = AutoAdapterModel.from_pretrained('allenai/specter2_aug2023refresh_base')

# Load the adapter(s) as per the required task, provide an identifier for the adapter in load_as argument and activate it
model.load_adapter("allenai/specter2_aug2023refresh", source="hf", load_as="proximity", set_active=True)
# Other possibilities: allenai/specter2_aug2023refresh_<classification|regression|adhoc_query>

docs = []
ids = []

for pmid in tqdm(data):
    title = data[pmid]['features']['Title']
    abstract = data[pmid]['features']['Abstract']
    docs.append(str(title) + tokenizer.sep_token + str(abstract))
    ids.append(pmid)

del data
# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

batch_size = 100  # Define an appropriate batch size
chunk_size = 100000  # Number of document id-embedding pairs per file
all_embeddings = []
all_ids = []

# Function to process a single batch
def process_batch(batch):
    inputs = tokenizer(batch, padding=True, truncation=True,
                       return_tensors="pt", return_token_type_ids=False, max_length=512)
    inputs = inputs.to(device)  # Move inputs to GPU
    with torch.no_grad():  # Disable gradient calculations
        output = model(**inputs)
    batch_embeddings = output.last_hidden_state[:, 0, :].cpu()  # Move embeddings back to CPU
    return batch_embeddings

chunk_index = 0
# Process the dataset in batches and store embeddings in npz
for i in tqdm(range(0, len(docs), batch_size)):
    batch_docs = docs[i:i + batch_size]
    batch_ids = ids[i:i + batch_size]
    batch_embeddings = process_batch(batch_docs)
    
    all_embeddings.append(batch_embeddings)
    all_ids.extend(batch_ids)

    # Save embeddings and ids to npz file in chunks
    if len(all_ids) >= chunk_size:
        all_embeddings_np = torch.cat(all_embeddings, dim=0).numpy()
        all_ids_np = np.array(all_ids)
        np.savez_compressed(os.path.join(save_dir+'/embeddings', f'tkg_embeddings_chunk_{chunk_index}.npz'), embeddings=all_embeddings_np, ids=all_ids_np)
        
        # Reset lists to free up memory
        all_embeddings = []
        all_ids = []
        chunk_index += 1

# Save any remaining embeddings and ids to npz file
if len(all_ids) > 0:
    all_embeddings_np = torch.cat(all_embeddings, dim=0).numpy()
    all_ids_np = np.array(all_ids)
    np.savez_compressed(os.path.join(save_dir+'/embeddings', f'tkg_embeddings_chunk_{chunk_index}.npz'), embeddings=all_embeddings_np, ids=all_ids_np)

print("Embeddings have been saved to files")
