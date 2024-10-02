import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score
import itertools
import os
from tqdm import tqdm

dataset = pd.read_csv('./data/hypotheses_processed.csv', sep=',', header=0)
use_features_scicheck = True
use_features_freq = False
use_features_sim = False

clf = RandomForestClassifier(n_estimators=2000, random_state=1337)

features_groups = {
    '1_features_scicheck': ['scicheck_score'],
    '2_features_freq': ['num_connections_train_subject', 'num_connections_train_object', 'freq_abstracts_subject', 'freq_abstracts_object', 'harmomic_mean_frequences', 'freq_abstracts_s&o'],
    '3_features_sim': ['max_cos_sim', 'min_emb_sim'],
    '4_features_emb': ['graph_emb_similarity', 'word_emb_similarity']
}

target = 'label'

metrics = []

# Check what entities appear the most including both subjects and objects
entities = pd.concat([dataset['s'], dataset['o']])
entities = entities.value_counts().head(20)

os.makedirs('./results', exist_ok=True)

kf = KFold(n_splits=4, shuffle=True, random_state=1)

# Generate all possible combinations of feature groups
all_combinations = []
for r in range(1, len(features_groups) + 1):
    all_combinations += list(itertools.combinations(features_groups.keys(), r))

for combination in tqdm(all_combinations):
    tqdm.write(f"Combination: {combination}")
    features = []
    for group in combination:
        features += features_groups[group]

    for train_index, test_index in kf.split(entities):
        train_entities = entities.iloc[train_index].index.to_list()
        test_entities = entities.iloc[test_index].index.to_list()
        
        train = dataset[dataset['s'].isin(train_entities) | dataset['o'].isin(train_entities)]
        test = dataset[dataset['s'].isin(test_entities) | dataset['o'].isin(test_entities)]

        clf.fit(train[features], train[target])
        test['predicted'] = clf.predict_proba(test[features])[:, 1]
        sorted_test = test.sort_values(by='predicted', ascending=False)

        for k in [5, 10, 15, 20]:
            top_k = sorted_test.head(k)
            precision = precision_score(top_k[target], top_k['predicted'].round())
            row = {'K': k, 'Precision': precision}
            for key in features_groups.keys():
                row[key] = key in combination
            metrics.append(row)
            
results_columns = list(features_groups.keys()) + ['K', 'Precision']
results_df = pd.DataFrame.from_records(metrics, columns=results_columns)

results_df.to_csv("./results/metrics_summary.csv", index=False)