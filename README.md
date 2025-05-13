
**Collab Link:** [https://colab.research.google.com/drive/1oVitFuzstCvtR3T_WBK-1LzDasEAllpR?usp=sharing](https://colab.research.google.com/drive/1oVitFuzstCvtR3T_WBK-1LzDasEAllpR?usp=sharing)

## Assignment Overview

This assignment focuses on applying Latent Dirichlet Allocation (LDA) for topic modeling on a dataset of real and fake news articles. The tasks include fitting an LDA model, analyzing the generated topics, examining topic distributions in individual documents, using LDA vectors as features for a classification task, and clustering documents based on their topic vectors.

---

### 1. LDA Topic Modeling and Analysis (k=10 topics)

Documents were preprocessed by converting to lowercase and filtering out stopwords. An LDA model was fitted with `k=10` topics.

**How well do the topics represent real-world topics?**
The topics represent real-world topics well because: each topic captures distinct, interpretable themes, the topics show strong coherence (all coherence scores > 370), there is low word overlap between topics (1.1-2.3 words on average), and they reflect known characteristics of true news (focus on policy, international affairs) versus fake news (focus on sensation, social media).

**Topic Summaries (Top words, Usage in Fake/True News, Bias):**

*   **Topic 0:** (trump, clinton, said, republican, campaign, hillary, president, party, election)
    *   Usage Fake: 0.184, True: 0.085. Bias (Fake): 0.368
*   **Topic 1:** (trump, people, one, president, like, donald, twitter, news, white)
    *   Usage Fake: 0.386, True: 0.024. Bias (Fake): 0.881
*   **Topic 2:** (said, house, bill, would, senate, republicans, republican, may, president)
    *   Usage Fake: 0.033, True: 0.109. Bias (True): -0.538
*   **Topic 3:** (said, trump, court, president, house, state, former, clinton, department)
    *   Usage Fake: 0.102, True: 0.154. Bias (True): -0.202
*   **Topic 4:** (police, gun, said, shooting, old, media, man, officers, year)
    *   Usage Fake: 0.088, True: 0.023. Bias (Fake): 0.578
*   **Topic 5:** (said, party, government, state, election, would, political, coalition, reuters)
    *   Usage Fake: 0.029, True: 0.106. Bias (True): -0.569
*   **Topic 6:** (said, trump, united, president, states, north, korea, russia, iran)
    *   Usage Fake: 0.036, True: 0.188. Bias (True): -0.677
*   **Topic 7:** (said, china, chinese, national, taiwan, cuba, beijing, obama, president)
    *   Usage Fake: 0.026, True: 0.032. Bias (True): -0.111
*   **Topic 8:** (said, government, reuters, people, security, state, military, killed, police)
    *   Usage Fake: 0.030, True: 0.127. Bias (True): -0.613
*   **Topic 9:** (said, would, tax, new, percent, trump, year, million, states)
    *   Usage Fake: 0.086, True: 0.150. Bias (True): -0.271

**Topic Coherence and Distinctiveness:**
All topics had coherence scores above 370 (e.g., Topic 0: 677.732, Topic 4: 370.581), indicating meaningful topics. Average word overlap between topics was low (ranging from 1.1 to 2.3 words), suggesting distinct topics.

---

### 2. Topic Distributions in Selected Documents

5 real news and 5 fake news examples were randomly selected to examine their topic distributions.

*   **Which topics are prevalent in the real news documents?**
    In the true news documents, they contain diverse topic distributions compared to fake news. There is a focus on institutional/governmental topics (e.g., Topic 2: legislative, Topic 3: legal/governmental, Topic 5: political/governmental, Topic 6: international affairs, Topic 8: security/governmental, Topic 9: economic/tax). True news documents also often have multiple significant topics per document.
    *   *Example True Document 1:* Dominated by Topic 5 (0.704) and Topic 3 (0.238).
    *   *Example True Document 4:* Dominated by Topic 6 (0.688) and Topic 5 (0.206).

*   **Which topics are prevalent in the fake news documents?**
    In the fake news documents, political content (Topic 0: campaign/election, often Trump/Clinton focused) and social media content (Topic 1: Trump/Twitter/sensational) are most prevalent.
    *   *Example Fake Document 1:* Dominated by Topic 0 (0.802).
    *   *Example Fake Document 2:* Dominated by Topic 1 (0.558) and Topic 0 (0.345).

---

### 3. Logistic Regression for Fake News Detection using LDA Vectors

LDA vectors (topic distributions for each document) were used as features to train a Logistic Regression classifier to predict whether a document is real or fake news.

**Which topics are most useful in determining real vs. fake news (based on regression coefficients)?**
Social media-related topics (Topic 1: coeff -10.345) and sensational content topics (Topic 4: coeff -4.181) are strong predictors of **fake news** (negative coefficients). International affairs (Topic 6: coeff 3.499) and legislative content (Topic 2: coeff 3.252) strongly indicate **true news** (positive coefficients).

**Top Coefficients and Interpretations:**
*   **Topic 1 (-10.345, FAKE):** (trump, people, one, president, like, donald, twitter, news, white) -> *Social media, Trump-centric*
*   **Topic 4 (-4.181, FAKE):** (police, gun, said, shooting, old, media, man, officers, year) -> *Sensational incidents, crime*
*   **Topic 0 (-0.753, FAKE):** (trump, clinton, said, republican, campaign, hillary, president, party, election) -> *Election/Campaign focus*

*   **Topic 6 (3.499, TRUE):** (said, trump, united, president, states, north, korea, russia, iran) -> *International affairs, foreign policy*
*   **Topic 2 (3.252, TRUE):** (said, house, bill, would, senate, republicans, republican, may, president) -> *Legislative, government process*
*   **Topic 8 (3.236, TRUE):** (said, government, reuters, people, security, state, military, killed, police) -> *Government, security, official reports*
*   And others like Topic 5, 3, 9, 7 also indicating TRUE news with smaller positive coefficients.

---

### 4. KMeans Clustering of Fake News Documents using LDA Vectors (K=10)

Fake news documents were clustered using KMeans based on their LDA topic vectors with K=10 clusters. 5 documents were selected from each cluster for analysis.

**Do the clusters correspond to anything?**
Yes, the clusters generally correspond to dominant underlying themes or combinations of topics within the fake news articles.

**Cluster Interpretations (based on selected fake news examples and their dominant topics):**

*   **Cluster 0:** Dominated by Topic 0 (campaign/election). Heavy focus on Trump, Clinton, and election coverage.
    *   *Example Docs:* Titles about Trump impressions, analyst poll on Hillary, Sanders open to being Clinton's VP.
*   **Cluster 1:** Dominated by Topic 1 (social media/Trump). Very high Topic 1 scores (0.650-0.984). Often critical/negative coverage of Trump.
    *   *Example Docs:* Report on Trump team working with Russians, Sesame Street on Trump, National Security expert on dangers of Trump.
*   **Cluster 2:** High Topic 3 scores (legal/court). Focus on investigations, legal cases, many related to Russia investigation.
    *   *Example Docs:* Firing of National Security Aide, Trump as illegitimate president due to Russia, Mueller hiring biased lawyers.
*   **Cluster 3:** Mix of Topics 8 (security), 4 (incidents), and 6 (international). Focus on national security, military, and international threats.
    *   *Example Docs:* McCain's health, security lapse at airport, Israelis encouraged to carry arms.
*   **Cluster 4:** High Topic 9 scores (economic/tax). About welfare, education funding, economics.
    *   *Example Docs:* Target raising minimum wage, African refugee population stressing welfare, Trump's taxes vs. Sanders'.
*   **Cluster 5:** Dominated by Topic 4 (police/incidents). Sensational crime stories or controversial incidents.
    *   *Example Docs:* School workers fired over transgender student, cops given permission to murder dogs, transgender target suing good samaritan.
*   **Cluster 6:** (Analysis not fully detailed for all 5 docs, but based on titles) Appears to be a mix, often involving political commentary or outrage.
    *   *Example Docs:* Republicans wanting Trump out, Trump rage at debate results.
*   **Cluster 7:** (Analysis not fully detailed) Appears to be diverse, possibly less coherent or mixing themes.
    *   *Example Docs:* Confused protesters at Trump fundraiser, bizarre race rant against Michelle Obama.
*   **Cluster 8:** (Analysis not fully detailed) Seems to involve political commentary and criticism.
    *   *Example Docs:* Christians and abortion, EU refugee aid, Wikileaks on James Clapper.
*   **Cluster 9:** (Analysis not fully detailed) Often involves foreign affairs or Trump's international dealings/statements.
    *   *Example Docs:* Trump supporter schooled on Russia, Trump's "Armada" lie about North Korea, Trump promises war on ISIS.

---

*   **Which resources did you use?**
    *   **Data source:** [Kaggle - Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
    *   **Course code (Colab links provided in original document):**
        *   [https://colab.research.google.com/drive/1q8Cf9QUwd45aWkOyPVjljOVb2N4xdUj5?usp=sharing#scrollTo=-V0J04mlOc5m](https://colab.research.google.com/drive/1q8Cf9QUwd45aWkOyPVjljOVb2N4xdUj5?usp=sharing#scrollTo=-V0J04mlOc5m)
        *   [https://colab.research.google.com/drive/1paraIRBL87aYdG9dtR6x2fs0P4QBEmHe?usp=sharing#scrollTo=Wi6X76BoaEN9](https://colab.research.google.com/drive/1paraIRBL87aYdG9dtR6x2fs0P4QBEmHe?usp=sharing#scrollTo=Wi6X76BoaEN9)
    *   **Scikit-learn Documentation:**
        *   LatentDirichletAllocation: [sklearn.decomposition.LatentDirichletAllocation](https://scikitlearn.org/1.5/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html)
        *   CountVectorizer: [sklearn.feature_extraction.text.CountVectorizer](https://scikitlearn.org/1.5/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)

*   **A few sentences about:**
    *   **What was the most difficult part of the assignment?**
        Understanding the logic of topic modeling: Raw Text -> CountVectorizer -> Word Counts -> LDA -> Topic Distributions.
        *   `# CountVectorizer: Converts text to word counts`
        *   `vec = CountVectorizer(stop_words=['the', 'a', ...])`
        *   `X = vec.fit_transform(df['document'])`
        *   `# LatentDirichletAllocation: Finds topics in word counts`
        *   `lda = LatentDirichletAllocation(n_components=10)`
        *   `doc_topics = lda.fit_transform(X)`
    *   **What was the most rewarding part of the assignment?**
        Practicing using clustering (KMeans) and LDA models together to analyze text data.
    *   **What did you learn doing the assignment?**
        How to handle situations when run time is too long (e.g., limit the number of documents by random selections for initial exploration or model tuning).

