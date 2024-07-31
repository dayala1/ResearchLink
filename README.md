# ResearchLink

Code needed to compute the features used by ResearchLink, condensed into a single file for convenience.

The hypotheses must be in the file `data/hypotheses.csv` with the following format: `subject,predicate,object,label,scicheck_score`. Then, execute `run.py` to generate a processed file with all features computed for each hypothesis, which can then be passed on to a classifier.

A number of precomputations are needed for different features. In the `data/` folder, we provide all of them for the CSKG-600 dataset.

## Contents of the repository
* **/CSKG-600.csv** contains our validation dataset, containing 600 manually verified research hypotheses.
* **/data** contains files needed to run ResearchLink, mostly containing metadata needed to compute features:
  * **/data/compute_embeddings.py** is a script used to compute graph-embeddings of each entity in the graph.
  * **/data/embeddings_<technique>.npy** contains the pre-computed graph embeddings of each entity in the graph.
  * **/data/entities.txt** contains a list of entities in the graph including their degree (total, in-degree, out-degree) and type.
  * **/data/entities_graph_emb.tsv** contains a list of entities with the same index as their corresponding embedding.
  * **/data/entity2count.json** contains the number of apparition of each entity in research abstracts, per year.
  * **/data/hypotheses.csv** contains the score assigned by SciCheck to each triple in the CSKG-600 dataset.
  * **/data/max_cos_sim.csv** contains the maximum cosine similarity of each triple in the CSKG-600 dataset to any other triple in the graph.
  * **/data/pair2freq.pkl** contains the number of times pairs of entities appear together in research abstracts, per year.
  * **/data/train.txt** contains the entire knowledge graph that can be used for training of relevant techniques.
* **/test_baselines.py** is a script used to test different baseline models on the CSKG-600 dataset.
* **/run.py** is the script that computes and stores ResearchLink's features, storing them in **hypotheses_processed.csv**.

## How to use

1. First, install the required libraries using ```pip install -r requirements.txt```.
2. Compute the desired graph embeddings by running the compute_embeddings.py script.
3. Use the run.py script to compute and store ResearchLink's features.
4. Train any desired model from the computed features.

## ResearchLink Algorithm for Classifying Research Hypotheses

### Input
- Knowledge Graph (KG)
- Number of top entities (N)
- SciCheck threshold (0.5)

### Output
- Classified triples as true or false hypotheses

### Algorithm

1. **Generate Candidate Triples**
   - Identify top N entities in KG.
   - For each entity \( e_i \) in top N entities:
     - Retrieve its type and possible relations.
     - For each relation \( r \), find compatible entities in top N.
     - Generate triples \( (e_i, r, t) \) not already in KG.

2. **Initial Pruning with SciCheck**
   - Train SciCheck on KG to compute confidence scores \( c \) for candidate triples.
   - Prune triples with \( c < 0.5 \) and keep the rest.

3. **Pre-compute Information for Features**
   - Compute TransH embeddings, degrees, occurrences, and joint occurrences for entities in KG.

4. **Compute Features**
   - For each labelled triple \( (s, p, o) \) and pruned triple \( (s, p, o) \):
     - Compute features \( f_1, f_2, \ldots, f_8 \).

5. **Train Random Forest Model**
   - Train Random Forest model on labelled triples using computed features.

6. **Apply Model for Inference**
   - For each pruned triple \( (s, p, o) \):
     - Apply Random Forest model to compute classification.
     - Label the triple as true or false hypothesis.
