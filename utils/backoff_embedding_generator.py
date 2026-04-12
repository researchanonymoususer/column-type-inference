import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class BackedOffEmbedding:
    def __init__(self, embedding_model, logger, ngram_range=(3, 5), cache=None):
        self.embedding_model = embedding_model
        self.vocab = embedding_model.index_to_key
        self.oov_results = []
        self.cache = cache

        self.vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=ngram_range
        )

        # Pre-compute TF-IDF vectorizer on entire vocabulary
        self.vocab_tfidf = self.vectorizer.fit_transform(self.vocab)
        self.logger = logger

    def surface_similarity_batch(self, word):
        """
        Compute similarity between OOV word and all vocab words efficiently.
        """
        # Transform OOV word using fitted vectorizer
        word_tfidf = self.vectorizer.transform([word])

        similarities = cosine_similarity(word_tfidf, self.vocab_tfidf)[0]

        return similarities

    def get_embedding(self, word, top_k=5, verbose=False):
        """
        Generate backed-off embedding for OOV word.
        """
        # If word exists in vocabulary, return embeddings
        if self.embedding_model.has_index_for(word):
            return self.embedding_model[word]

        # Check cache for OOV word
        if self.cache is not None:
            cached = self.cache.get(word)
            if cached is not None:
                return cached

        # Else, perform Back-off Estimation for the OOV word
        # Compute all similarities at once
        similarities = self.surface_similarity_batch(word)

        # Get top-k indices (argpartition is O(n) vs O(n log n) for sort)
        if top_k < len(similarities):
            top_k_indices = np.argpartition(similarities, -top_k)[-top_k:]
            # Sort only the top-k
            top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])[::-1]]
        else:
            top_k_indices = np.argsort(similarities)[::-1][:top_k]

        top_scores = similarities[top_k_indices]

        # Normalize weights
        weights = top_scores / (top_scores.sum() + 1e-10)

        if verbose:
            best_candidates = [(self.vocab[idx], top_scores[i])
                               for i, idx in enumerate(top_k_indices)]
            self.logger.info(f"Best candidates: {best_candidates}")
            self.oov_results.append({
                "OOV": word,
                "Best_Candidates": best_candidates
            })

        # Vectorized embedding lookup and weighted sum
        # Get embeddings for top-k words
        top_words = [self.vocab[idx] for idx in top_k_indices]
        vectors = np.array([self.embedding_model[w] for w in top_words])

        # Weighted sum
        embedding = weights @ vectors

        # Store result in cache
        if self.cache is not None:
            self.cache.set(word, embedding)

        return embedding
