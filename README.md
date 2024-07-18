# cm4ai-teaming-recommendation
Bridge2AI CM4AI Data Generation Project. Teaming recommendation alalgorithms.

# Teaming Recommendation Algorithms for Bridge2AI

## Motivation

In modern science, effective teaming is crucial for addressing challenging research problems, especially those requiring efforts from multiple fields and areas of expertise, such as the Bridge2AI project. Our Talent Knowledge Graph (TKG) for Bridge2AI is a heterogeneous graph where both nodes and edges have distinct types. This allows for the promotion of teaming using historical publication data and based on the researchers’ needs. The first step in this process is obtaining accurate representations of authors in the TKG.

Traditional Graph Neural Networks (GNNs) are limited to homogeneous graphs, meaning they cannot differentiate between different types of nodes and edges and treat them uniformly. To address this limitation, we deploy heterogeneous GNNs, which can handle heterogeneous graphs and obtain embeddings of nodes. These embeddings serve as the foundation for teaming recommendation algorithms and can be used to identify similar scientists, datasets, biomedical methods, and bio entities, thus contributing to the identification of potential users to test our datasets.

In this report, we present the results of four baselines of heterogeneous graph neural networks implemented on TKG and their evaluations: 1) Heterogeneous Graph Transformers (HGT) (Hu et al., 2020), 2) Heterogeneous Graph Neural Network (HetGNN) (Zhang et al., 2019), 3) SPECTER2 aggregation (Cohan et al., 2020), and 4) Metapath2vec (Dong et al., 2017). We use these embeddings to identify potential collaboration recommendations for researchers.

We emphasize that this is ongoing work; this report demonstrates how we achieve good representations of authors in TKG. Another key step is representing researchers' collaboration needs and utilizing TKG to reason about potential future fair collaborations while providing explainability. We are actively combining the reasoning abilities of Large Language Models with the factual information in the Talent Knowledge Graph to fulfill fair teaming recommendations.

## 2\. Description of the Dataset

To obtain a trustworthy researcher dataset that well represents the Bridge2AI project, we collected 122 ORCIDs from researchers within the Bridge2AI Project, 107 of which were from the Bridge2AI kick-off meeting, and manually supplemented another 15 important researchers’ ORCIDs. We manually reviewed all 122 Bridge2AI researchers' profiles in our Talent Knowledge Graph (TKG) to ensure proper author disambiguation. Subsequently, we collected all 12,000 papers published by these 122 Bridge2AI researchers from the PubMed Knowledge Graph (from the beginning until the end of 2021; we will update it with the newest data by July 10th, 2024).

We further identified 35,000 co-authors of the 122 Bridge2AI researchers, obtaining a total of 1.7 million papers authored by these 35,000 co-authors. These 1.7 million papers are primarily used to understand the scientific background of all 35,000 authors. These 35,000 authors form the backbone of the Talent Knowledge Graph. We utilized publication, patent, and funding datasets for enriched information.

Our core focus is the 35,000 authors, which include diverse Bridge2AI talents and all their collaborators throughout their careers. Our goal is to represent these 35,000 authors well through heterogeneous graph learning, obtaining precise high-dimensional embedding representations that are highly useful. With these, we can support AI/ML-enabled CM4AI Bridge2AI workforce initiatives, such as providing better teaming recommendations for the current Bridge2AI teams and recommending potential users for our developed dataset.

We defined four types of nodes in our heterogeneous Talent Knowledge Graph.

**Paper**

- **Paper features:** AuthorNum, PubYear, CitedCount, Title, Abstract, Venue, is_core (we have 12K core papers, and all the info about these papers' authors is known; if is_core = 0, for this paper, we may only have the info of one author out of our 35K focused authors), Field of Study.
- **Paper neighbors:** Author {with author order, affiliations in this paper}, Mesh, citing papers, reference papers

**Author**

- **Author features:** BeginYear, FullName, Gender, Race (Gender and Race are only for reference when evaluating fair teaming)
- **Author neighbors:** Papers

**Mesh**

- **Mesh features:** MeshTerm
- **Mesh neighbors:** papers

**Venue**

- **Venue features:** VenueName
- **Venue neighbors:** papers

## 3\. Teaming Recommendation Algorithms on TKG

The first step is to obtain accurate author node embeddings in the TKG. We tested four heterogeneous graph learning methods and evaluated them on our proposed future collaboration link prediction task. The four heterogeneous graph neural methods are:

1. Heterogeneous Graph Transformers (HGT) (Hu et al., 2020)
    1. We use Metapath2vec network embeddings as input node features. The paper label (the research field to which the paper belongs) serves as the training task to obtain the embeddings.
2. Heterogeneous Graph Neural Network (HetGNN) (Zhang et al., 2019)
    1. We use the semantic embeddings encoded from the paper’s title and abstract by SPECTER2, along with the Metapath2vec network embeddings, as input node features. Negative sampling is used for training.
3. SPECTER2 aggregation (Cohan et al., 2020)
    1. For this baseline, we simply aggregate each author’s paper semantic embeddings encoded from the paper’s title and abstract by SPECTER2 as the author’s embedding.
4. Metapath2vec (Dong et al., 2017)
    1. This baseline uses random walks based on metapaths (e.g., Author-paper-Author) to ensure that the semantic relationships between different types can be properly combined into the Skip-gram algorithms to obtain embeddings. Negative sampling is also used here for training.

## 4\. Results

For the evaluation, we designed a link prediction task: who will collaborate in the future? With the author embeddings learned by the previous methods, we split the TKG into a training set and a test set by a specific year. Both the training set and test set consist of (author A, author B, collaborated or not) tuples. We trained a binary classifier using the obtained author embeddings on the training set and tested it on the test set. We report the results for the split years 2015 and 2018; see Table 1 for details. We used OpenHGNN (Han et al., 2022), a toolkit for Heterogeneous Graph Neural Networks, and DGL (Deep Graph Library) to implement our four heterogeneous graph learning baselines, i.e., HetGNN, HGT, SPECTER2 (Agg.), and Metapath2vec.

We want to emphasize that predicting who will collaborate in the future is a complex task and is key for the teaming algorithm. People may want to collaborate with similar authors, researchers with complementary expertise, or simply famous researchers. Therefore, this evaluation setup cannot fully reflect the complex real-world scenario. When implementing a user-centered system, the algorithms should consider user needs and have interactivity and explainability. We are still working on this by utilizing LLM+TKG.

As shown in Table 1, Metapath2vec achieves better metrics, but all four baselines do not reach high performance on the future collaboration link prediction. We speculate that this is because the motivations for future teaming are diverse, and the Bridge2AI project and our dataset are interdisciplinary: sometimes people collaborate for similarity, and sometimes for complementarity, which may mislead the model.

Table 1. Link prediction results. Split notation in the data denotes train/test data split years or ratios.

| Data<sub>split</sub> | Metrics | HetGNN | HGT | SPECTER2 (Agg.) | Metapath2vec |
| --- | --- | --- | --- | --- | --- |
| TKG_2015_split | Micro_F1 | 0.538 | 0.502 | 0.532 | **0.610** |
| Macro_f1 | 0.537 | 0.493 | 0.532 | **0.610** |
| TKG_2018_split | Micro_f1 | 0.504 | 0.503 | 0.501 | **0.616** |
| Macro_f1 | 0.503 | 0.494 | 0.501 | **0.615** |

## 5\. Conclusion and Ongoing Work

In this stage, we have implemented state-of-the-art heterogeneous graph learning methods to obtain node (e.g., authors, papers, venues, etc.) representations, which serve as the foundation for fair teaming recommendations. However, teaming in real academic circles is highly diverse and contextual. To provide appropriate recommendations, we must understand the user’s specific need for collaboration, whether it is to address a specific problem in the current project or to explore future relevant directions. Therefore, we must capture the users’ needs, which can be done by allowing users to input in the form of natural language.

Apart from the factual data provided by TKG, we must have an agent with the knowledge to understand the scenario and provide recommendations as well as explainability based on the relevant information retrieved from TKG. We are working on this TKG+LLM direction for our next step in teaming recommendations.

# References

Cohan, A., Feldman, S., Beltagy, I., Downey, D., & Weld, D. S. (2020). SPECTER: Document-level Representation Learning using Citation-informed Transformers. _ACL_.

Dong, Y., Chawla, N. V., & Swami, A. (2017). metapath2vec: Scalable Representation Learning for Heterogeneous Networks. _Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining_, 135–144. <https://doi.org/10.1145/3097983.3098036>

Han, H., Zhao, T., Yang, C., Zhang, H., Liu, Y., Wang, X., & Shi, C. (2022). OpenHGNN: An Open Source Toolkit for Heterogeneous Graph Neural Network. _Proceedings of the 31st ACM International Conference on Information & Knowledge Management_, 3993–3997. <https://doi.org/10.1145/3511808.3557664>

Hu, Z., Dong, Y., Wang, K., & Sun, Y. (2020). Heterogeneous Graph Transformer. _Proceedings of The Web Conference 2020_, 2704–2710. <https://doi.org/10.1145/3366423.3380027>

Zhang, C., Song, D., Huang, C., Swami, A., & Chawla, N. V. (2019). Heterogeneous Graph Neural Network. _Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining_, 793–803. <https://doi.org/10.1145/3292500.3330961>
