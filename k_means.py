import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Load the manual queries into a pandas dataframe
df = pd.read_excel(r'queries.xlsx')

def preprocess_text(text):
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove stop words
    stop_words = set(['a', 'an', 'the', 'and', 'or', 'in', 'on', 'at', 'to', 'from'])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text 

vectorizer = TfidfVectorizer(preprocessor =preprocess_text)
X = vectorizer.fit_transform(df['FIRST_OPEN_TEXT'])

# Cluster the queries using K-means
kmeans = KMeans(n_clusters=20,n_init =50, random_state=42).fit(X)

# Add the cluster labels to the dataframe
df['cluster'] = kmeans.labels_

#count the frequency of each query in each cluster
cluster_counts =df.groupby(['cluster','FIRST_OPEN_TEXT']).size().reset_index(name ='count')
# Print the queries in each cluster
for i in range(10):
    print(f"Cluster {i}:")
    cluster_df =cluster_counts[cluster_counts['cluster']==i]
    if not cluster_df.empty:
        most_common_query =cluster_df.loc[cluster_df['count'].idxmax(),'FIRST_OPEN_TEXT']
        print(f"most common query:{most_common_query}")
        print(cluster_df.sort_values('count',ascending =False))
    else:
        print("No queries in this cluster")
    print()    
   # Save the results to a CSV file
with pd.ExcelWriter(r'clustered.xlsx') as writer:
    # Write the clustered queries to a worksheet
    df.to_excel(writer, sheet_name='clustered_queries', index=False)

    # Write the frequency counts to a worksheet for each cluster
    for i in range(20):
        cluster_df = cluster_counts[cluster_counts['cluster'] == i]
        if not cluster_df.empty:
            sheet_name = f"cluster_{i}"
            cluster_df.to_excel(writer, sheet_name=sheet_name, index=False)
