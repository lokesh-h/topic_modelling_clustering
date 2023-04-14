import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation

# Load the manual queries into a pandas dataframe
df = pd.read_excel(r'\Downloads\queries.xlsx')

# Preprocess the queries and item names using TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X1 = vectorizer.fit_transform(df['FIRST_OPEN_TEXT'])
X2 = vectorizer.fit_transform(df['ITEM_NAME'])

# Combine the preprocessed queries and item names into a single feature matrix
X = pd.concat([pd.DataFrame(X1.toarray()), pd.DataFrame(X2.toarray())], axis=1)

# Cluster the queries using K-means
kmeans = KMeans(n_clusters=10, random_state=42).fit(X)

# Get the most frequent words associated with each topic
lda = LatentDirichletAllocation(n_components=10, random_state=42)
lda.fit(X)
topic_words = []
if hasattr(vectorizer, 'get_feature_names'):
    feature_names = vectorizer.get_feature_names()
else:
    feature_names = list(vectorizer.vocabulary_.keys())
for topic_idx, topic in enumerate(lda.components_):
    top_word_indices = topic.argsort()[:-11:-1]
    topic_words.append([feature_names[i] for i in top_word_indices if i < len(feature_names)])

# Assign a descriptive name to each cluster based on the most frequent words associated with the corresponding topic
cluster_names = {}
for cluster_label in range(10):
    top_topic = lda.transform(X[kmeans.labels_ == cluster_label]).mean(axis=0).argmax()
    cluster_names[cluster_label] = ' '.join([w.capitalize() for w in topic_words[top_topic]])

# Add the cluster labels and names to the dataframe
df['cluster'] = kmeans.labels_
df['cluster_name'] = df['cluster'].map(cluster_names)

# Count the frequency of each query in each cluster
cluster_counts = df.groupby(['cluster', 'FIRST_OPEN_TEXT']).size().reset_index(name='count')

# Export the results to an Excel file
with pd.ExcelWriter(r'Downloads\clustered.xlsx') as writer:
    df.to_excel(writer, sheet_name='Queries', index=False)
    cluster_counts.to_excel(writer, sheet_name='Cluster Counts', index=False)
