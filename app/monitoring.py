from prometheus_client import Counter, Histogram, Gauge

def setup_metrics():
    return {
        'rag_queries_total': Counter(
            'rag_queries_total',
            'Total RAG queries processed',
            ['collection', 'status']
        ),
        'request_duration': Histogram(
            'rag_request_duration_seconds',
            'RAG request processing time',
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        ),
        'cache_hits_total': Counter('rag_cache_hits_total', 'RAG cache hits'),
        'rag_errors_total': Counter('rag_errors_total', 'RAG processing errors'),
        'document_embeddings_total': Counter('document_embeddings_created_total', 'Document embeddings created'),
        'similarity_score': Histogram(
            'rag_similarity_scores',
            'RAG similarity scores distribution',
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
    }