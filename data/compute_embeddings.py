from pykeen.triples import TriplesFactory
import pandas as pd
from pykeen.models import TransE
from torch.optim import Adam
from pykeen.training import SLCWATrainingLoop
import numpy as np

triples_df = pd.read_csv('./data/train.txt', sep='\t', header=None, names=['subject', 'predicate', 'object', 'positive']).dropna()
# Create a dataset from the triples
triples_factory = TriplesFactory.from_labeled_triples(triples=triples_df.values)
# Fit the model
models = []

embedding_dim = 256

models.append(
    TransE(triples_factory=triples_factory,
            embedding_dim=embedding_dim).to('cuda:0')
)
model = models[0]
optimizer = Adam(params=model.get_grad_params())

training_loop = SLCWATrainingLoop(
    model=model,
    triples_factory=triples_factory,
    optimizer=optimizer,
)

training_loop.train(
    triples_factory=triples_factory,
    num_epochs=50,
    batch_size=256,
)

entity_embeddings = model.entity_representations[0](indices=None).detach().cpu().numpy()
embeddings = []
with open('./data/entities_graph_emb.tsv') as f:
    for line in f:
        entity = line.strip().split('\t')[1]
        if entity in triples_factory.entity_to_id:
            embeddings.append(entity_embeddings[triples_factory.entity_to_id[entity]])
        else:
            embeddings.append(np.zeros(embedding_dim))    
    
embeddings = np.array(embeddings)
np.save('./data/embeddings_TransE.npy', embeddings)