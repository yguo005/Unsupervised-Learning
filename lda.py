from matplotlib import pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
nltk.download('stopwords')
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import re

def preprocess_text(text):
    # Convert to lowercase and split on non-word characters
    words = re.split(r'\W+', str(text).lower())
    # Filter out empty strings, numbers, and short words
    words = [word for word in words if word and not word.isnumeric() and len(word) > 2]
    return ' '.join(words)

# Random sampling
MAX_DOCS = 5000 # Maximum number of documents to use from each category
fake_df = pd.read_csv('Fake.csv').sample(n=MAX_DOCS, random_state=42)
true_df = pd.read_csv('True.csv').sample(n=MAX_DOCS, random_state=42)

# Preprocess text before creating document matrix
fake_df['text'] = fake_df['text'].apply(preprocess_text)
true_df['text'] = true_df['text'].apply(preprocess_text)
all_text = pd.concat([fake_df['text'], true_df['text']])

df = pd.DataFrame({
    'document': all_text
})

print("Sample documents:")
print(df.head())

# Set up LDA parameters
topics = 10
vec = CountVectorizer(
    
    stop_words=['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
               'of', 'with', 'by'] + list(nltk.corpus.stopwords.words('english')),
    
)

# Transform text documents to numerical matrix
X = vec.fit_transform(df['document'])

# Initialize LDA model to find 10 topics
lda = LatentDirichletAllocation(n_components=topics, random_state=0)
# Transform documents into topic distributions
doc_topics = lda.fit_transform(X) #doc_topics is document-topic matrix where each row represents a document, each column represents a topic

print("\nEvaluating Topic Quality for True vs Fake News Dataset:")
print("=" * 70)

# 1. Topic Distribution Analysis by News Category
fake_topics = doc_topics[:len(fake_df)]
true_topics = doc_topics[len(fake_df):]

print("\n1. Topic Usage Patterns in True vs Fake News:")
print("-" * 50)
for topic_idx in range(topics):
    # Get top words for this topic
    top_words_idx = lda.components_[topic_idx].argsort()[:-10:-1]
    top_words = [vec.get_feature_names_out()[i] for i in top_words_idx]
    
    # Calculate usage statistics
    fake_usage = fake_topics[:, topic_idx].mean()
    true_usage = true_topics[:, topic_idx].mean()
    usage_bias = (fake_usage - true_usage) / (fake_usage + true_usage)
    
    print(f"\nTopic {topic_idx}:")
    print(f"Top words: {', '.join(top_words)}")
    print(f"Usage in Fake News: {fake_usage:.3f}")
    print(f"Usage in True News: {true_usage:.3f}")
    print(f"Bias (+ = fake, - = true): {usage_bias:.3f}")

# 2. Topic Quality Metrics
print("\n2. Topic Coherence and Distinctiveness:")
print("-" * 50)
for topic_idx in range(topics):
    # Calculate topic coherence
    topic_words = lda.components_[topic_idx]
    top_indices = topic_words.argsort()[:-10:-1]
    coherence = np.mean(topic_words[top_indices]) / np.mean(topic_words)
    
    # Calculate distinctiveness from other topics
    distinctiveness = []
    for other_idx in range(topics):
        if other_idx != topic_idx:
            other_words = set(vec.get_feature_names_out()[lda.components_[other_idx].argsort()[:-10:-1]])
            this_words = set(vec.get_feature_names_out()[top_indices])
            overlap = len(this_words.intersection(other_words))
            distinctiveness.append(overlap)
    
    print(f"\nTopic {topic_idx}:")
    print(f"Coherence score: {coherence:.3f}")
    print(f"Average word overlap: {np.mean(distinctiveness):.1f} words")

# 3. Category Separation Analysis
print("\n3. Category Separation Analysis:")
print("-" * 50)

# Calculate dominant topics
fake_dominant = np.argmax(fake_topics, axis=1)
true_dominant = np.argmax(true_topics, axis=1)

# Analyze topic dominance by category
for topic_idx in range(topics):
    fake_dom_count = np.sum(fake_dominant == topic_idx)
    true_dom_count = np.sum(true_dominant == topic_idx)
    total_docs = len(fake_dominant) + len(true_dominant)
    
    print(f"\nTopic {topic_idx} dominance:")
    print(f"Dominant in {fake_dom_count} fake news documents ({fake_dom_count/len(fake_dominant)*100:.1f}%)")
    print(f"Dominant in {true_dom_count} true news documents ({true_dom_count/len(true_dominant)*100:.1f}%)")



# Calculate normalized topic-word distributions
topic_words = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]

# Get vocabulary and document-term matrix
preprocessed_docs = vec.get_feature_names_out() # Get all unique words in vocabulary
doc_term_matrix = X.toarray() #Convert matrix X to array


# Randomly select 5 examples from each category after preprocessing
n_samples = 5
fake_indices = np.random.choice(len(fake_df), n_samples, replace=False)
true_indices = np.random.choice(len(true_df), n_samples, replace=False)

print("\nAnalysis of 5 Selected Documents from Each Category:")
print("\nFake News Examples:")
print("-" * 50)
for i, idx in enumerate(fake_indices):
    print(f"\nFake Document {i+1}:")
    # Get words that appear in this document (count > 0)
    doc_terms = [word for word, count in zip(preprocessed_docs, doc_term_matrix[idx]) if count > 0]
    # Show first 10 words
    print(f"Preprocessed terms: {' '.join(doc_terms[:10])}...") 
    print(f"Topic distribution:")
    for topic in range(topics):
        print(f"Topic {topic}: {doc_topics[idx][topic]:.3f}")
    print("-" * 50)

print("\nTrue News Examples:")
print("-" * 50)
for i, idx in enumerate(true_indices):
    idx_adjusted = idx + len(fake_df)  # Adjust index for true news documents because when concatenated the fake and true news datasets:Fake news came first (indices 0 to len(fake_df)-1)
    print(f"\nTrue Document {i+1}:")
    doc_terms = [word for word, count in zip(preprocessed_docs, doc_term_matrix[idx_adjusted]) if count > 0]
    print(f"Preprocessed terms: {' '.join(doc_terms[:10])}...")
    print(f"Topic distribution:")
    for topic in range(topics):
        print(f"Topic {topic}: {doc_topics[idx_adjusted][topic]:.3f}")
    print("-" * 50)

# Create labels (0 for fake, 1 for true)
labels = np.array([0] * len(fake_df) + [1] * len(true_df))

# Split the LDA vectors (doc_topics) into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    doc_topics,  # Using LDA vectors instead of raw text
    labels, # target variable
    test_size=0.25, # 25% of data for testing
)

# Train logistic regression model
lr_model = LogisticRegression(max_iter=500)
lr_model.fit(X_train, y_train)

# Evaluate model
test_accuracy = lr_model.score(X_test, y_test)

print("\nLogistic Regression Classification using LDA vectors:")
print(f"Testing accuracy: {test_accuracy:.3f}")

# Get and analyze coefficients
print("Analyze Coefficients to determine which topics are most important for predicting True news (positive) vs Fake news (negative)")
print("-" * 70)

# Get coefficients and sort by absolute value to find most influential topics
coef = lr_model.coef_[0]  # Get coefficients from the model
topic_importance = [(i, coef[i]) for i in range(len(coef))]
sorted_importance = sorted(topic_importance, key=lambda x: abs(x[1]), reverse=True)

# Print topic importance
for topic_idx, importance in sorted_importance:
    direction = "TRUE" if importance > 0 else "FAKE"
    print(f"Topic {topic_idx}: {importance:.3f} (indicates {direction} news)")
    # Print top words for this topic
    top_words_idx = lda.components_[topic_idx].argsort()[:-10:-1]  # Get top 10 words
    top_words = [vec.get_feature_names_out()[i] for i in top_words_idx]
    print(f"Top words: {', '.join(top_words)}")
    print()

# Get LDA vectors for fake news documents only
fake_doc_topics = doc_topics[:len(fake_df)]
# Apply KMeans clustering with k=10
k = 10
km = KMeans(n_clusters=k)
cluster_labels = km.fit_predict(fake_doc_topics)
cluster_labels

# Visualize the predicted classes
# Reduce dimensionality to 2D for visualization
# Since LDA vectors that have 10 dimensions (topics), need to reduce them to 2D first to visualize them
pca = PCA(n_components=2)
fake_doc_topics_2d = pca.fit_transform(fake_doc_topics)

# Visualize the clusters
colors = [(0,1,0), (0,0,1), (1,0,0), (0,1,1), (1,0,1), (1,1,0), (1,1,1), 
          (0.3,0.8,0.8), (0.5,0.5,0), (0.8,0.2,0.5)]  # 10 colors for 10 clusters

plt.style.use('fast')
plt.figure(figsize=(12, 8), dpi=100)
# Plot each cluster
for i in range(k):
    # Get points belonging to cluster i
    mask = cluster_labels == i
    plt.scatter(fake_doc_topics_2d[mask, 0],  # x coordinates of points in cluster i
               fake_doc_topics_2d[mask, 1],    # y coordinates of points in cluster i
               s=50, 
               color=colors[i],
               marker='s', 
               edgecolor='black',
               label=f'Cluster {i}',
               rasterized=True )

# Plot centroids
centroids_2d = pca.transform(km.cluster_centers_)
plt.scatter(centroids_2d[:, 0],     # x coordinates of centroids
           centroids_2d[:, 1],      # y coordinates of centroids
           s=250, 
           marker='*',
           color='red', 
           edgecolor='black',
           label='Centroids')

plt.title('Document Clusters based on Topic Distributions')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.legend(scatterpoints=1)
plt.grid()
plt.show()

# Measure cluster quality using within-cluster SSE
print('\nEvaluating Cluster Quality:')
print('Within-cluster SSE: %.2f' % km.inertia_)

# Calculate within-cluster SSE for different k values (1 to 20)
wc_SSE = [KMeans(n_clusters=k).fit(fake_doc_topics).inertia_ 
          for k in range(1, 21)]  # Testing more k values since we used k=10

# Plot the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, 21), wc_SSE, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Within-cluster SSE')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()

print("\nSample FAKE News Documents from Each Cluster:")
print("=" * 80)

for i in range(k):
    print(f"\nCluster {i}:")
    print("-" * 50)
    
    # Get indices of FAKE news documents in this cluster
    cluster_indices = np.where(cluster_labels == i)[0]
    fake_indices = [idx for idx in cluster_indices if idx < len(fake_df)]  # Only fake news indices
    
    if not fake_indices:
        print("No fake news documents in this cluster")
        continue
    
    # Randomly select 5 fake news documents
    sample_indices = np.random.choice(fake_indices, 
                                    size=min(5, len(fake_indices)), 
                                    replace=False)
    
    # Print information for each selected document
    for idx in sample_indices:
        # Get document from fake_df
        title = fake_df.iloc[idx]['title']
        text = fake_df.iloc[idx]['text']
        
        print(f"\nFake Document {idx}")
        print(f"Title: {title}")
        print(f"Text preview: {text[:200]}...")
        
        # Show topic distribution for this document
        print("Top topics:")
        top_topics = sorted(enumerate(doc_topics[idx]), 
                          key=lambda x: x[1], 
                          reverse=True)[:3]
        for topic_idx, weight in top_topics:
            print(f"Topic {topic_idx}: {weight:.3f}")
        print("-" * 30)



