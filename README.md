# ResearchLink

Code needed to compute the features used by ResearchLink, condensed into a single file for convenience.

The hypotheses must be in the file `data/hypotheses.csv` with the following format: `subject,predicate,object,label,scicheck_score`. Then, execute `run.py` to generate a processed file with all features computed for each hypothesis, which can then be passed on to a classifier.

A number of precomputations are needed for different features. In the `data/` folder, we provide all of them for the CSKG-600 dataset.

The algorithm pseudocode for the application of ResearchLink is as follows:

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
