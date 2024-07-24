import pandas as pd
from sklearn.model_selection import KFold
from pykeen.models import TransE, TransH, TransD, RotatE, RGCN
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pykeen.predict import predict_triples
from torch.optim import Adam
from pykeen.training import SLCWATrainingLoop

kg_dataset = pd.read_csv('./data/train.txt', sep='\t', header=None, names=['subject', 'predicate', 'object', 'positive']).dropna()
dataset = pd.read_csv('./data/hypotheses.csv', sep=',', header=None, names=['subject', 'predicate', 'object', 'label', 'score']).dropna()

# Check what entities appear the most including both subjects and objects
entities = pd.concat([dataset['subject'], dataset['object']])
entities = entities.value_counts().head(20)
metrics = []

training_factory = TriplesFactory.from_labeled_triples(triples=kg_dataset[['subject', 'predicate', 'object']].values)
embedding_dim = 16
model = RGCN(triples_factory=training_factory, embedding_dim=embedding_dim, num_layers=1).to('cuda:0')
optimizer = Adam(params=model.get_grad_params())

training_loop = SLCWATrainingLoop(
    model=model,
    triples_factory=training_factory,
    optimizer=optimizer,
)

training_loop.train(
    triples_factory=training_factory,
    num_epochs=100,
    batch_size=256,
)

# 4-fold validation using each time 5 of the 20 entities
kf = KFold(n_splits=4, shuffle=True, random_state=1)
for train_index, test_index in kf.split(entities):
    train_entities = entities.iloc[train_index].index.to_list()
    test_entities = entities.iloc[test_index].index.to_list()
    print("Test entities: ", test_entities)
    # triples containing training entities and label = 1 -> ./data/train.tsv
    # triples containing test entities -> ./data/test.tsv
    train = dataset[dataset['subject'].isin(train_entities) | dataset['object'].isin(train_entities) & dataset['label'] == 1]
    test = dataset[dataset['subject'].isin(test_entities) | dataset['object'].isin(test_entities)]
    testing_factory = TriplesFactory.from_labeled_triples(triples=test[['subject', 'predicate', 'object']].values)
    scored_triples = []
    predictions = predict_triples(model=model, triples=testing_factory.triples, triples_factory=training_factory)
    for triple, score in zip(predictions.result.cpu().numpy(), predictions.scores.cpu().numpy()):
        s = training_factory.entity_id_to_label[triple[0]]
        p = training_factory.relation_id_to_label[triple[1]]
        o = training_factory.entity_id_to_label[triple[2]]
        # take label column of matching triple in  test
        label = test[(test['subject'] == s) & (test['predicate'] == p) & (test['object'] == o)]['label'].values[0]
        scored_triple = {'subject': s, 'predicate': p, 'object': o, 'label': label, 'score': score}
        scored_triples.append(scored_triple)
    # Sort the scored triples by score
    result = pd.DataFrame(scored_triples).sort_values(by='score', ascending=False)
        
    # compute precision at k = 5, 10, 15, 20
    for k in [5, 10, 15, 20]:
        precision = result['label'].head(k).sum() / k
        metrics.append({'k': k, 'precision': precision})

# average each precision at k across folds
metrics = pd.DataFrame(metrics).groupby('k').mean()
print(metrics)
