# Methodology

## 1. Overview

This chapter presents a complete methodology for automated biological species identification that requires no supervised training data. The approach rests on three pillars: visual feature extraction from a domain-specific foundation model, unsupervised clustering of those features, and zero-shot text-based annotation of the resulting clusters with a structured uncertainty quantification framework. Together, these stages form a pipeline that takes a collection of unlabeled specimen images as input and produces species-level identifications, each accompanied by a calibrated confidence assessment that tells a domain expert which annotations can be trusted automatically and which require human verification.

The pipeline proceeds through a sequence of well-defined stages. First, each specimen image is passed through a vision-language foundation model to obtain a dense embedding vector that captures visual and semantic information about the organism. Second, these embedding vectors are grouped through unsupervised clustering, where images depicting the same species naturally form coherent clusters in the embedding space. Third, each cluster receives a species-level name through zero-shot text matching, where the cluster centroid is compared against text embeddings of candidate species names enriched with taxonomic context. Fourth, a composite uncertainty score is computed from four independent signals, and the result is used to triage every annotation into one of three confidence tiers. Finally, an interactive review interface allows a domain expert to verify, correct, or reassign annotations, and these expert decisions are merged back into the final dataset with full provenance tracking.

The following sections describe each stage in formal detail, including the mathematical formulations, the design rationale behind key decisions, and the specific parameters used throughout.


## 2. Feature Extraction with BioCLIP-2

### 2.1 Model Architecture and Training Background

The pipeline uses BioCLIP-2 as its primary feature extractor, a vision-language foundation model built on the Vision Transformer (ViT-L/14) architecture. BioCLIP-2 was pretrained on TreeOfLife-200M, a large-scale dataset of approximately 200 million organism images paired with structured taxonomic text sourced from the Global Biodiversity Information Facility (GBIF). This training dataset and its taxonomic text supervision distinguish BioCLIP-2 from general-purpose vision-language models such as CLIP, which was trained on generic internet image-caption pairs. Because BioCLIP-2 has learned to associate visual features of organisms with their taxonomic descriptions, its embedding space is inherently organized along biological axes of variation, making it well-suited for fine-grained species discrimination without any task-specific fine-tuning.

The ViT-L/14 backbone divides each input image into a grid of 14 by 14 pixel patches, projects each patch into a 768-dimensional token embedding, and processes the full sequence through a stack of transformer layers with self-attention. The final representation is a single 768-dimensional vector obtained from the classification token, which aggregates information from all spatial positions in the image and captures both local texture patterns and global structural features of the organism.

### 2.2 Image Preprocessing

Each input image undergoes a standardized preprocessing pipeline dictated by the model's training configuration. The image is loaded in RGB format, resized and center-cropped to the resolution expected by ViT-L/14, and then normalized using the channel-wise mean and standard deviation statistics from the training set. Corrupted or unreadable image files are detected during loading and replaced with zero tensors of the appropriate dimensions, so that the pipeline can proceed without interruption while logging warnings for later inspection.

### 2.3 Embedding Extraction and Normalization

Images are processed in batches of configurable size (default 64) to balance GPU memory usage against throughput. For a batch of images $\{I_1, I_2, \ldots, I_B\}$, the forward pass through the image encoder $f_\theta$ produces a matrix of raw embeddings:

$$\mathbf{Z} = \begin{bmatrix} f_\theta(I_1) \\ f_\theta(I_2) \\ \vdots \\ f_\theta(I_B) \end{bmatrix} \in \mathbb{R}^{B \times 768}$$

Each embedding vector is then L2-normalized to project it onto the unit hypersphere:

$$\hat{\mathbf{z}}_i = \frac{\mathbf{z}_i}{\max\left(\|\mathbf{z}_i\|_2,\; \epsilon\right)}$$

where $\epsilon = 10^{-8}$ is a small constant that prevents division by zero in the degenerate case of a zero vector (which can arise from corrupted image replacements). After normalization, every embedding lies on the surface of the unit hypersphere in $\mathbb{R}^{768}$, and cosine similarity between any two embeddings reduces to their dot product. This normalization is essential for the text matching stage, where cosine similarity serves as the scoring function.

After extraction, a validation check scans the full feature matrix for NaN values. If any are detected, the pipeline raises an error, because NaN propagation from a single corrupted embedding could silently compromise all downstream computations.

### 2.4 Foundation Model Comparison

To validate the choice of BioCLIP-2 as the primary extractor, the pipeline supports comparative evaluation against three alternative foundation models:

- **BioCLIP** (ViT-B/16): The predecessor to BioCLIP-2, trained on TreeOfLife-10M with a smaller backbone and fewer training images. It produces 512-dimensional embeddings.
- **CLIP** (ViT-L/14): The general-purpose vision-language model from OpenAI, trained on 400 million internet image-caption pairs. It shares the same ViT-L/14 architecture as BioCLIP-2 but lacks domain-specific biological training.
- **DINOv2** (ViT-L/14): A self-supervised vision model trained with a distillation objective on LVD-142M. It produces strong visual features but has no text encoder, which means it cannot participate in the zero-shot annotation stage and is evaluated only on clustering quality.

All four models are accessed through the same extraction interface, and their embeddings are evaluated on identical clustering benchmarks to isolate the effect of pretraining data and objective on species discrimination capability.


## 3. Text Embedding with Taxonomic Prompt Ensembling

### 3.1 Motivation for Prompt Ensembling

In a contrastive vision-language model, the similarity between an image embedding and a text embedding depends on how well the text description matches the model's internal concept of the depicted entity. No single text prompt perfectly captures the visual and semantic identity of a biological species in embedding space. A bare scientific name like "Aglia tau" activates different aspects of the model's knowledge than a taxonomically contextualized description like "Aglia tau, family Saturniidae, order Lepidoptera." By encoding multiple complementary prompts for each species and averaging the resulting embeddings, the pipeline obtains a more stable and representative text centroid that reduces the variance introduced by any single prompt formulation. This approach follows the prompt ensembling strategy introduced in the original CLIP paper.

### 3.2 Template Design with Taxonomic Placeholders

Each prompt template is a string containing placeholders that are filled with species-specific taxonomic information. The default template set is:

1. `"{name}"` -- the bare scientific name
2. `"a photo of {name}"` -- a descriptive framing
3. `"a photo of {name}, a moth"` -- with common group context
4. `"{name}, family {family}, order {order}"` -- full taxonomic hierarchy
5. `"a species of genus {genus}, family {family}"` -- genus-family structure

The placeholders are resolved as follows. The species name `\{name\}$ is derived from the folder name or candidate list entry, with underscores replaced by spaces and numeric prefixes stripped (to handle datasets like CUB-200 that prepend class indices). The genus $\{genus\}$ is parsed as the first word of the binomial scientific name, since by taxonomic convention the first word of a binomial always denotes the genus. The family $\{family\}$ and order $\{order\}$ fields are retrieved from a taxonomy table that is populated by one of three sources: a precomputed CSV file, a GBIF Species Match API query, or a simple name-parsing fallback. If a required field is missing for a particular species (for example, if no family information is available), the templates that reference that field are silently skipped, and the ensemble averages over the remaining valid templates.

### 3.3 Encoding and Averaging

For a given species $s$ with $T_s$ valid filled templates $\{p_1, p_2, \ldots, p_{T_s}\}$, the text encoder $g_\phi$ produces one embedding per template:

$$\mathbf{e}_t = g_\phi(p_t) \in \mathbb{R}^{768}, \quad t = 1, \ldots, T_s$$

Each embedding is individually L2-normalized, then the ensemble mean is computed and re-normalized:

$$\bar{\mathbf{e}}_s = \frac{1}{T_s} \sum_{t=1}^{T_s} \hat{\mathbf{e}}_t, \qquad \hat{\bar{\mathbf{e}}}_s = \frac{\bar{\mathbf{e}}_s}{\|\bar{\mathbf{e}}_s\|_2}$$

The re-normalization after averaging is necessary because the mean of unit vectors is not itself a unit vector. The resulting text centroid $\hat{\bar{\mathbf{e}}}_s$ represents the consensus location of species $s$ in the shared vision-language embedding space. The full matrix of text centroids $\mathbf{E} \in \mathbb{R}^{S \times 768}$, where $S$ is the number of candidate species, is used in the annotation stage.


## 4. Candidate Species List Construction

Before annotation can proceed, the pipeline must assemble a list of candidate species against which the unknown specimens will be compared. This list serves as the vocabulary of possible identifications. The pipeline supports three modes of candidate construction, each suited to different practical scenarios.

### 4.1 GBIF Mode

In the default mode, the pipeline queries the GBIF Occurrence API to retrieve a list of species that have been recorded in a specified geographic region and taxonomic group. The query is parameterized by a taxon key (for example, 797 for Lepidoptera) and a geographic region (for example, Europe). In balanced mode, the query proceeds family by family: for each family within the target order, the pipeline retrieves a configurable number of common species (default 40 per family) and a smaller number of rare or infrequently observed species (default 10 per family). This balanced sampling ensures that the candidate list covers the taxonomic breadth of the target group rather than being dominated by a few hyper-abundant species. Reference images for each candidate species are retrieved from GBIF occurrence records and used for visual comparison in the review interface.

### 4.2 Local Mode

When the input images are organized in subfolders named after species, the pipeline uses the folder names directly as the candidate list. This mode requires no internet access and is appropriate when working with curated datasets where the folder structure represents the known or suspected species composition. The genus is parsed from the first word of each folder name, but family and order information is not available unless supplemented by a taxonomy CSV. Reference images are sampled from the dataset itself.

### 4.3 CSV Mode

In this mode, the candidate species list is read from a CSV file that contains columns for the species name and optionally the genus, family, and order. This mode supports offline workflows where a taxonomist has prepared a list of species that are likely to be present in the collection.


## 5. Dimensionality Reduction

### 5.1 The Dual-Path Architecture

High-dimensional embedding vectors (768 dimensions) present two distinct challenges for clustering algorithms. Partition-based methods such as K-Means and Gaussian Mixture Models operate on variance and distance in the feature space, and their performance can degrade when many dimensions carry little discriminative signal. Density-based methods such as HDBSCAN rely on local density estimation, which suffers from the concentration of distances phenomenon in high dimensions, where the relative contrast between nearest-neighbor and farthest-neighbor distances diminishes as dimensionality grows.

Empirical evaluation on the EU-Moth and CUB-200 benchmark datasets revealed that no single reduction strategy works well for all clustering algorithms. PCA, which maximizes retained variance along orthogonal axes, aligns naturally with the variance-based objectives of partition methods and produces stable results for K-Means, Spectral Clustering, Agglomerative Clustering, and Gaussian Mixture Models. However, Spectral Clustering and GMM fail on UMAP-reduced features due to graph disconnection and singular covariance matrices, respectively. Conversely, UMAP, which preserves local neighborhood topology, aligns with the local-density objectives of DBSCAN and HDBSCAN and achieves substantially lower noise fractions and comparable or better clustering quality than PCA for these methods.

Based on these findings, the pipeline employs a dual-path reduction architecture:

- **Path A (Partition methods):** PCA reduction to 50 dimensions
- **Path B (Density methods):** UMAP reduction to 100 dimensions

A critical design constraint governs both paths: dimensionality reduction is applied only for clustering, never for text matching. The cosine similarity between image and text embeddings is defined only in the original 768-dimensional CLIP embedding space, and projecting either modality into a lower-dimensional subspace would destroy the learned alignment between vision and language.

### 5.2 PCA Reduction

Principal Component Analysis finds the linear projection that maximizes the retained variance. Given the centered feature matrix $\mathbf{X} \in \mathbb{R}^{N \times 768}$, PCA computes the eigendecomposition of the covariance matrix $\mathbf{C} = \frac{1}{N-1}\mathbf{X}^\top \mathbf{X}$ and selects the top $d$ eigenvectors $\mathbf{V}_d \in \mathbb{R}^{768 \times d}$ corresponding to the $d$ largest eigenvalues $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_d$. The reduced features are:

$$\mathbf{X}_{\text{PCA}} = \mathbf{X} \mathbf{V}_d \in \mathbb{R}^{N \times d}$$

The fraction of total variance retained is:

$$R = \frac{\sum_{j=1}^{d} \lambda_j}{\sum_{j=1}^{768} \lambda_j}$$

With $d = 50$, the pipeline typically retains between 78% and 86% of the total variance across the tested datasets, which provides sufficient signal for partition-based clustering while removing noise dimensions.

### 5.3 UMAP Reduction

Uniform Manifold Approximation and Projection constructs a weighted k-nearest-neighbor graph in the high-dimensional space and then optimizes a low-dimensional layout that preserves the topological structure of this graph. The key parameters are:

- **n_components = 100:** The target dimensionality. A higher value than is typical for visualization (2 or 3) is chosen because the goal here is clustering rather than visualization, and 100 dimensions provide enough room to preserve the complex neighborhood structure of fine-grained species embeddings.
- **n_neighbors = 15:** The number of nearest neighbors used to construct the high-dimensional graph. This parameter controls the balance between local and global structure preservation. A value of 15 provides a reasonable balance for datasets of approximately 2000 images.
- **min_dist = 0.1:** The minimum distance between points in the low-dimensional space. A moderate value prevents points from collapsing together while allowing the formation of tight clusters.
- **metric = cosine:** The distance metric used in the high-dimensional space. Cosine distance is the natural choice for L2-normalized embeddings, where direction carries more information than magnitude.

The UMAP objective can be understood intuitively as follows. For each point $\mathbf{x}_i$, UMAP defines a fuzzy set membership over its neighbors in the high-dimensional space. It then seeks a low-dimensional embedding where a similar fuzzy set membership holds, minimizing the cross-entropy between the two fuzzy topological representations. The result is a layout that contracts local neighborhoods (points that are close in 768 dimensions remain close in 100 dimensions) while allowing global distances to distort as needed.


## 6. Clustering

### 6.1 Clustering Algorithm Suite

The pipeline implements six clustering algorithms that span three methodological families, providing a comprehensive evaluation of how different inductive biases interact with the structure of foundation model embeddings.

#### 6.1.1 K-Means

K-Means partitions the data into $K$ clusters by iteratively assigning each point to its nearest centroid and then recomputing centroids as cluster means. The algorithm minimizes the within-cluster sum of squared distances:

$$\mathcal{J} = \sum_{k=1}^{K} \sum_{\mathbf{x}_i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|_2^2$$

where $C_k$ is the set of points assigned to cluster $k$ and $\boldsymbol{\mu}_k = \frac{1}{|C_k|}\sum_{\mathbf{x}_i \in C_k} \mathbf{x}_i$ is the cluster centroid. Because the optimization is non-convex and sensitive to initialization, the pipeline runs the algorithm with 50 independent random restarts ($n_\text{init} = 50$) and selects the solution with the lowest objective value. Each run iterates for at most 300 steps. K-Means operates on PCA-reduced features (Path A).

#### 6.1.2 Spectral Clustering

Spectral Clustering constructs a nearest-neighbor affinity graph where each node is an image and edges connect the 15 nearest neighbors. It then computes the graph Laplacian $\mathbf{L} = \mathbf{D} - \mathbf{W}$, where $\mathbf{W}$ is the weighted adjacency matrix and $\mathbf{D}$ is the degree matrix, and finds the $K$ eigenvectors of $\mathbf{L}$ corresponding to the smallest eigenvalues. These eigenvectors form a spectral embedding of the graph, and K-Means is applied to this embedding to obtain the final cluster assignments. Spectral Clustering captures non-convex cluster shapes that K-Means cannot and is particularly effective when species form elongated or irregularly shaped regions in embedding space. It operates on PCA-reduced features (Path A) with the nearest-neighbors affinity mode.

#### 6.1.3 Agglomerative Clustering

Agglomerative (hierarchical) clustering proceeds bottom-up, starting with each image as its own cluster and repeatedly merging the pair of clusters that minimizes the Ward linkage criterion. Ward linkage minimizes the increase in total within-cluster variance at each merge step:

$$\Delta(C_a, C_b) = \frac{|C_a| \cdot |C_b|}{|C_a| + |C_b|} \|\boldsymbol{\mu}_a - \boldsymbol{\mu}_b\|_2^2$$

This criterion tends to produce compact, roughly equally sized clusters, which aligns well with the expectation that species classes in curated biological datasets have similar sample sizes. It operates on PCA-reduced features (Path A).

#### 6.1.4 Gaussian Mixture Model

The Gaussian Mixture Model (GMM) represents the data as a mixture of $K$ Gaussian distributions, each parameterized by a mean vector $\boldsymbol{\mu}_k$ and a covariance matrix $\boldsymbol{\Sigma}_k$:

$$p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \; \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

where $\pi_k$ are the mixing weights satisfying $\sum_k \pi_k = 1$. The parameters are estimated by the Expectation-Maximization algorithm. The pipeline uses diagonal covariance matrices ($\boldsymbol{\Sigma}_k$ is diagonal) to reduce the number of parameters and avoid singular matrices in moderate-dimensional spaces, with a regularization term of $10^{-6}$ added to the diagonal. Although GMM produces soft probabilistic assignments (posterior probabilities $\gamma_{ik} = p(k \mid \mathbf{x}_i)$), the pipeline uses hard assignments via argmax for compatibility with the evaluation metrics. It operates on PCA-reduced features (Path A).

#### 6.1.5 DBSCAN

Density-Based Spatial Clustering of Applications with Noise (DBSCAN) identifies clusters as connected regions of high density separated by regions of low density. A point is classified as a core point if at least $\text{min\_samples}$ other points lie within a radius of $\varepsilon$ in the feature space. Clusters are formed by connecting overlapping neighborhoods of core points, and points that are not reachable from any core point are labeled as noise (cluster ID = -1).

The critical parameter $\varepsilon$ is determined through k-distance analysis: for each point, the distance to its $k$-th nearest neighbor is computed, and these distances are sorted in ascending order. The resulting k-distance curve typically shows an "elbow" at the transition between intra-cluster and inter-cluster distances, and the $\varepsilon$ value at this elbow is used as the neighborhood radius. The pipeline uses Savitzky-Golay smoothing on the k-distance curve to robustly locate the elbow. DBSCAN operates on UMAP-reduced features (Path B).

#### 6.1.6 HDBSCAN

Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN) extends DBSCAN by eliminating the need for a global $\varepsilon$ parameter. Instead of using a fixed density threshold, HDBSCAN builds a hierarchy of clusterings across all density levels and extracts the most persistent clusters using the Excess of Mass (EOM) criterion. This means that different regions of the embedding space can have different effective density thresholds, which is valuable when species vary in their within-class visual diversity.

The primary parameter is $\text{min\_cluster\_size}$, which specifies the minimum number of points required to form a cluster. The pipeline uses a default value of 5, which allows the algorithm to detect small species groups while avoiding the formation of spurious clusters from pairs of similar images. HDBSCAN also produces two outputs beyond cluster assignments that are used in the uncertainty quantification stage:

- **Membership probabilities** $p_i \in [0, 1]$: For each non-noise point, the probability indicates how strongly the point belongs to its assigned cluster. Core points deep within a cluster receive probabilities close to 1, while points near cluster boundaries receive lower values.
- **Outlier scores** $o_i \in [0, 1]$: A measure of how outlying each point is with respect to the overall density structure, where 0 indicates a typical point and 1 indicates a strong outlier.

HDBSCAN operates on UMAP-reduced features (Path B) and is the default clustering method for the unlabeled specimen identification task.

### 6.2 Noise Point Handling

Both DBSCAN and HDBSCAN can assign points to a special noise class (cluster ID = -1), indicating that these points do not belong to any cluster with sufficient density support. Noise points are excluded from cluster-level evaluation metrics (ARI, NMI, Hungarian accuracy) to avoid penalizing the algorithm for correctly identifying ambiguous or outlying specimens. However, noise points are still annotated in the text matching stage (using their individual nearest-text-embedding match) and are included in the interactive review interface for expert assessment.


## 7. Evaluation Metrics

When ground-truth species labels are available, the pipeline computes three complementary evaluation metrics to assess clustering quality from different perspectives.

### 7.1 Hungarian Accuracy

Because unsupervised clustering assigns arbitrary integer labels to clusters, a direct comparison with ground-truth labels requires first finding the optimal mapping between cluster IDs and class IDs. The Hungarian algorithm solves this assignment problem by constructing a co-occurrence matrix $\mathbf{M} \in \mathbb{R}^{n \times n}$ where $M_{ij}$ counts the number of images assigned to cluster $i$ that belong to ground-truth class $j$, and then finding the one-to-one assignment $\sigma$ that maximizes the total number of correctly matched images:

$$\text{Accuracy} = \frac{1}{N} \max_{\sigma} \sum_{i=1}^{n} M_{i,\sigma(i)}$$

This optimization is solved in $O(n^3)$ time using the linear sum assignment algorithm applied to the negated co-occurrence matrix. Hungarian accuracy answers the question: "If we could assign the best possible species name to each cluster, what fraction of images would be correctly identified?"

### 7.2 Adjusted Rand Index

The Adjusted Rand Index (ARI) measures the agreement between two partitions of a dataset, adjusted for the expected agreement under random chance. For a clustering $\mathcal{C}$ and a ground-truth partition $\mathcal{T}$, the ARI is defined as:

$$\text{ARI} = \frac{\text{RI} - \mathbb{E}[\text{RI}]}{\max(\text{RI}) - \mathbb{E}[\text{RI}]}$$

where RI is the Rand Index, which counts the fraction of point pairs that are either in the same cluster and same class or in different clusters and different classes. The adjustment for chance ensures that a random clustering receives an expected ARI of 0, a perfect clustering receives 1, and clusterings worse than random can receive negative values. ARI is invariant to cluster relabeling and does not require the number of clusters to match the number of classes.

### 7.3 Normalized Mutual Information

Normalized Mutual Information (NMI) quantifies the amount of information shared between the clustering partition and the ground-truth partition:

$$\text{NMI}(\mathcal{C}, \mathcal{T}) = \frac{2 \cdot I(\mathcal{C}; \mathcal{T})}{H(\mathcal{C}) + H(\mathcal{T})}$$

where $I(\mathcal{C}; \mathcal{T})$ is the mutual information between the two partitions and $H(\cdot)$ denotes entropy. NMI ranges from 0 (no shared information) to 1 (perfect agreement) and provides a complementary perspective to ARI by emphasizing information-theoretic properties of the partition rather than pairwise agreement.

### 7.4 Per-Cluster Purity and Per-Class Accuracy

Beyond aggregate metrics, the pipeline computes per-cluster purity, defined as the fraction of images in each cluster that belong to the dominant ground-truth class:

$$\text{Purity}_k = \frac{\max_j |C_k \cap T_j|}{|C_k|}$$

and per-class accuracy, defined as the fraction of images from each ground-truth class that are correctly assigned under the Hungarian mapping. These fine-grained metrics identify specific species or clusters that are problematic, enabling targeted analysis of failure cases.


## 8. Zero-Shot Text Matching Annotation

### 8.1 Cluster Centroid Annotation

Once clusters have been formed, each cluster must receive a species-level name. The pipeline accomplishes this through zero-shot text matching in the original 768-dimensional embedding space, where the contrastive training of BioCLIP-2 has established a meaningful alignment between visual and textual representations.

For each cluster $C_k$ containing images with embeddings $\{\mathbf{z}_i\}_{i \in C_k}$, the cluster centroid is computed as the arithmetic mean of the member embeddings:

$$\boldsymbol{\mu}_k = \frac{1}{|C_k|} \sum_{i \in C_k} \mathbf{z}_i$$

The centroid is then compared against all $S$ candidate species text centroids $\{\hat{\bar{\mathbf{e}}}_1, \ldots, \hat{\bar{\mathbf{e}}}_S\}$ using cosine similarity:

$$\text{sim}(k, s) = \frac{\boldsymbol{\mu}_k \cdot \hat{\bar{\mathbf{e}}}_s}{\|\boldsymbol{\mu}_k\| \; \|\hat{\bar{\mathbf{e}}}_s\|}$$

Because both image embeddings and text centroids are L2-normalized, this reduces to a simple dot product. The species with the highest similarity score is assigned as the cluster's annotation:

$$s_k^{(1)} = \arg\max_s \; \text{sim}(k, s)$$

The runner-up species $s_k^{(2)}$ (the species with the second-highest similarity) is also recorded, and the margin between the two provides a direct measure of annotation confidence:

$$m_k = \text{sim}(k, s_k^{(1)}) - \text{sim}(k, s_k^{(2)})$$

A large margin indicates that the top candidate is clearly distinguished from alternatives, while a small margin suggests that two or more species are visually similar and the annotation may be unreliable.

### 8.2 Per-Image Voting

In parallel with the centroid-level annotation, each image independently votes for its nearest species text embedding:

$$v_i = \arg\max_s \; \text{sim}(\mathbf{z}_i, \hat{\bar{\mathbf{e}}}_s)$$

These per-image votes are not used directly for annotation (the centroid-based annotation is more stable because averaging across images reduces noise), but they provide the raw material for the cluster agreement uncertainty signal described in Section 9.3. They also serve as a secondary check: if the majority of images in a cluster vote for a different species than the centroid annotation, this disagreement is a strong indicator of annotation error.


## 9. Uncertainty Quantification

The annotation of each image is accompanied by a composite uncertainty score that combines four independent signals. Each signal captures a different mode of potential error, and their combination provides a robust, multi-dimensional assessment of annotation reliability. The signals are designed to be complementary: a correct annotation will typically have low values on all four signals, while different types of errors produce elevated values on different subsets of signals.

### 9.1 Signal 1: Semantic Uncertainty (Text Margin)

The semantic uncertainty measures how distinguishable the top-ranked species is from the runner-up in the text matching space. For each image $i$ with a full cosine similarity vector $\mathbf{s}_i \in \mathbb{R}^S$ over all candidate species, the margin is the gap between the top two scores:

$$u_i^{\text{sem}} = 1 - \left(s_i^{(1)} - s_i^{(2)}\right)$$

where $s_i^{(1)}$ and $s_i^{(2)}$ are the highest and second-highest cosine similarity scores, respectively. A value close to 0 indicates a large margin (strong distinction between the top two candidates), while a value close to 1 indicates that the top two species received nearly identical similarity scores and the annotation could easily swing either way. This signal is particularly effective at flagging confusion between visually similar congeners (species within the same genus), where the absolute similarity scores may both be high but the margin between them is small.

### 9.2 Signal 2: Spatial Uncertainty (Membership Probability)

The spatial uncertainty measures how confidently each image belongs to its assigned cluster. When HDBSCAN is used as the clustering method, this signal directly leverages the membership probabilities that HDBSCAN computes as part of its density-based cluster extraction:

$$u_i^{\text{spa}} = 1 - p_i$$

where $p_i$ is the HDBSCAN membership probability for image $i$. Images that lie deep within a dense core of their cluster receive $p_i$ close to 1 (low uncertainty), while images near cluster boundaries or in transitional density regions receive lower probabilities (high uncertainty). Noise points (cluster ID = -1) are assigned the maximum spatial uncertainty of 1.0.

When a clustering method other than HDBSCAN is used and membership probabilities are not available, the pipeline falls back to the per-sample silhouette score as a proxy for spatial uncertainty:

$$u_i^{\text{spa}} = \frac{1 - \text{sil}_i}{2}$$

where $\text{sil}_i \in [-1, 1]$ is the silhouette coefficient for image $i$, computed using cosine distance. The silhouette score measures how much more similar an image is to members of its own cluster than to members of the nearest neighboring cluster, and the linear transformation maps it to the $[0, 1]$ range.

### 9.3 Signal 3: Cluster Agreement

The cluster agreement signal measures the internal consistency of species-level voting within each cluster. For cluster $C_k$, the majority vote fraction is computed as the proportion of images whose individual vote matches the most popular species within the cluster:

$$f_k = \frac{|\{i \in C_k : v_i = v_k^*\}|}{|C_k|}$$

where $v_k^*$ is the species that received the most votes within cluster $C_k$. The base agreement uncertainty for all images in the cluster is:

$$u_i^{\text{agr}} = 1 - f_k$$

Additionally, images whose individual vote disagrees with the cluster majority receive an extra penalty:

$$u_i^{\text{agr}} = \min\left(1.0, \; (1 - f_k) + 0.2\right) \quad \text{if } v_i \neq v_k^*$$

This penalty reflects the fact that a dissenting image is more likely to be either misplaced in the wrong cluster or a genuinely ambiguous specimen. Noise points receive the maximum agreement uncertainty of 1.0.

### 9.4 Signal 4: Taxonomic Consistency

The taxonomic consistency signal detects clusters that mix species from different taxonomic families. Under normal conditions, all images in a well-formed cluster should belong to the same species and therefore the same family. If the per-image votes within a cluster span multiple families, this is a strong indicator that the cluster contains a heterogeneous mix of species, even if the images are visually similar.

For each cluster $C_k$, the set of unique families $\mathcal{F}_k$ is compiled from the taxonomic lookup of each image's individual species vote. The uncertainty is scaled by the number of distinct families:

$$u_i^{\text{tax}} = \min\left(1.0, \; \frac{|\mathcal{F}_k| - 1}{3}\right)$$

A taxonomically pure cluster ($|\mathcal{F}_k| = 1$) receives zero uncertainty. A cluster mixing two families receives $\frac{1}{3} \approx 0.33$, three families receives $\frac{2}{3} \approx 0.67$, and four or more families is capped at 1.0. This signal requires a taxonomy table with family-level information and returns zero when such information is unavailable.

### 9.5 Combined Uncertainty and Confidence Tiers

The four signals are combined into a single composite uncertainty score through a weighted average:

$$u_i^{\text{combined}} = \frac{w_{\text{sem}} \cdot u_i^{\text{sem}} + w_{\text{spa}} \cdot u_i^{\text{spa}} + w_{\text{agr}} \cdot u_i^{\text{agr}} + w_{\text{tax}} \cdot u_i^{\text{tax}}}{w_{\text{sem}} + w_{\text{spa}} + w_{\text{agr}} + w_{\text{tax}}}$$

The default weights are:

| Signal | Weight | Rationale |
|--------|--------|-----------|
| Semantic ($w_{\text{sem}}$) | 0.35 | The most informative single signal for annotation correctness |
| Spatial ($w_{\text{spa}}$) | 0.25 | Captures cluster boundary effects |
| Agreement ($w_{\text{agr}}$) | 0.25 | Detects internal cluster inconsistency |
| Taxonomic ($w_{\text{tax}}$) | 0.15 | Requires external taxonomy data, which may not always be available |

The combined uncertainty is then compared against two absolute thresholds to assign each image to one of three confidence tiers:

$$\text{tier}_i = \begin{cases} \text{auto\_accept} & \text{if } u_i^{\text{combined}} < \tau_{\text{review}} \\ \text{review} & \text{if } \tau_{\text{review}} \leq u_i^{\text{combined}} < \tau_{\text{human}} \\ \text{human\_required} & \text{if } u_i^{\text{combined}} \geq \tau_{\text{human}} \end{cases}$$

The default thresholds are $\tau_{\text{review}} = 0.55$ and $\tau_{\text{human}} = 0.70$.

The use of absolute thresholds rather than percentile-based thresholds is a deliberate design choice. Percentile-based thresholds (for example, "flag the top 5% most uncertain annotations") guarantee that a fixed fraction of images is always flagged for review, regardless of the actual uncertainty distribution. This masks real differences between methods: a high-quality clustering where all annotations are confident would still flag 5% of images, while a poor clustering with widespread confusion would flag the same 5%. Absolute thresholds allow the data to speak for itself. A strong clustering method can achieve zero flags, while a weak one may flag hundreds, providing a more honest and informative assessment of annotation reliability.


## 10. Interactive Expert Review

### 10.1 HTML Report Interface

The pipeline generates a self-contained HTML report that serves as the primary interface for expert review. Each cluster is presented as a visual card containing the specimen images from that cluster, reference images for the predicted species and the runner-up species, dual confidence meters (one for clustering confidence, one for annotation confidence), and per-image annotation controls.

The clustering confidence is computed from the spatial and agreement signals:

$$\text{conf}_{\text{cluster}} = 1 - (0.5 \cdot u^{\text{spa}} + 0.5 \cdot u^{\text{agr}})$$

This answers the question: "Do these images belong together as a group?" The annotation confidence is computed from the semantic and taxonomic signals:

$$\text{conf}_{\text{annot}} = 1 - (0.6 \cdot u^{\text{sem}} + 0.4 \cdot u^{\text{tax}})$$

This answers the distinct question: "Is the species name assigned to this group correct?" Separating these two aspects of confidence is important because a cluster can be tightly coherent (high clustering confidence) yet receive the wrong name (low annotation confidence), or conversely, the name can be correct but the cluster may contain a few misplaced images (high annotation confidence, moderate clustering confidence).

### 10.2 Expert Decision Workflow

The review interface supports three levels of expert intervention:

**Cluster-level verdicts.** The expert can mark an entire cluster as "All Correct" (confirming the AI annotation), "Wrong Name" (the cluster is coherent but misidentified, with a text field to enter the correct species), or "Species Not Listed" (the true species is not in the candidate list).

**Per-image controls.** Each image within a cluster has three buttons: a checkmark to confirm that the image is correctly placed, a cross to flag it as incorrectly placed, and an arrow to move it to a different species. The move action opens a modal dialog with an autocomplete search over all cluster species and candidate species, plus an option to mark the image as noise.

**Undo and redo.** All actions are recorded in a history stack and can be undone with Ctrl+Z or redone with Ctrl+Y, allowing the expert to explore alternative arrangements without risk.

### 10.3 Persistence and Finalization

All expert decisions are automatically saved to the browser's localStorage, keyed by the report filename. This means the expert can close the browser, return hours or days later, and find all their progress intact. When the review is complete, the expert clicks the "Finalize Review and Download" button, which iterates through every annotation record (embedded as JSON in the HTML at generation time), applies per-image decisions first (highest priority) and then cluster-level verdicts (for images without individual decisions), and produces a complete CSV file containing all images with their final annotations and provenance labels.

The provenance labels track the origin of each annotation:

- **AI_ZeroShot:** The original AI annotation, unreviewed.
- **Human_Confirmed:** The expert verified the AI annotation as correct.
- **Human_Expert:** The expert provided a corrected species name.
- **Flagged_Wrong:** The expert flagged the annotation as incorrect but did not provide a correction.
- **Human_Noise:** The expert determined that the image does not depict an identifiable specimen.


## 11. Post-Review Processing

### 11.1 Merge and Reconciliation

After the expert produces the finalized annotations CSV, the pipeline's merge stage reads this file, auto-detects its format (per-image, cluster-level, or review template), and applies the corrections to the master annotation DataFrame. For confirmed annotations, the provenance is updated and the uncertainty is set to zero. For corrected annotations, the species label is replaced, the provenance is set to "Human_Expert," and all four uncertainty signals are zeroed to reflect the expert's authority.

### 11.2 Dataset Organization

The merge stage also produces a structured output directory where images are organized into folders named after their final species assignment. Flagged images (those marked as wrong without a correction) are placed in a special folder, and images marked as noise are placed in a separate noise folder. This organized directory structure can serve directly as a training dataset for supervised classification or as a reference collection for further taxonomic work.

### 11.3 Summary Statistics

A summary report is generated that includes the total number of images and unique species, a breakdown of annotation provenance (how many were AI-accepted, human-confirmed, human-corrected, and flagged), and a per-species list with image counts and correction status. When ground-truth labels are available (in local or CSV mode), the summary also reports AI accuracy on the expert-reviewed subset, providing a direct measure of how often the zero-shot annotation was correct before human intervention.


## 12. Reproducibility

All stochastic components of the pipeline are controlled by a single random seed that is propagated to Python's random module, NumPy, PyTorch (both CPU and CUDA), and all scikit-learn algorithms that accept a random state parameter. This ensures that given the same input data, configuration, and seed, the pipeline produces identical results across runs. The default seed is 42.

All hyperparameters are specified in a single YAML configuration file, which serves as a complete and portable record of the experimental setup. No parameters are hardcoded in the source code; every numerical constant is either drawn from the configuration or documented as a default value in the relevant function's signature. This design allows experiments to be reproduced by sharing the configuration file alongside the code.
