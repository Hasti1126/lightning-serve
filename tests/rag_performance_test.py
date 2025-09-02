import requests
import time
import threading
import statistics
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

class RAGPerformanceTest:
    def __init__(self, base_url="http://localhost:8000", collection="default"):
        self.base_url = base_url
        self.collection = collection
        self.results = []
        
    def test_rag_query(self, query, collection=None):
        """Test single RAG query performance"""
        start_time = time.time()
        effective_collection = collection or self.collection
        
        try:
            response = requests.post(
                f"{self.base_url}/rag/query",
                data={
                    "query": query,
                    "collection_name": effective_collection,
                    "top_k": 3,
                    "include_context": True
                }
            )
            
            end_time = time.time()
            
            if response.status_code == 200:
                result_data = response.json()
                return {
                    "success": True,
                    "response_time": end_time - start_time,
                    "processing_time": result_data.get("processing_time", 0),
                    "similarity_score": result_data.get("similarity_score", 0),
                    "from_cache": result_data.get("from_cache", False),
                    "answer_length": len(result_data.get("answer", "")),
                    "context_docs": len(result_data.get("context_documents", []))
                }
            else:
                return {
                    "success": False,
                    "response_time": end_time - start_time,
                    "error": response.text
                }
                
        except Exception as e:
            return {
                "success": False,
                "response_time": time.time() - start_time,
                "error": str(e)
            }
    
    def concurrent_rag_test(self, num_queries=50, max_workers=10):
        """Test concurrent RAG queries"""
        print(f"🚀 Running concurrent RAG test: {num_queries} queries, {max_workers} workers")
        
        test_queries = [
            "What is the main topic of this document?",
            "Summarize the key findings",
            "What methodology is described?",
            "What are the conclusions?",
            "Find any statistical data",
            "What recommendations are made?",
            "Describe the visual elements",
            "What problems are identified?",
            "What solutions are proposed?",
            "What is the document structure?"
        ]
        
        # Repeat queries to reach target number
        queries_to_run = []
        for i in range(num_queries):
            query = test_queries[i % len(test_queries)]
            queries_to_run.append(f"{query} (Test {i+1})")
        
        start_time = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all queries
            future_to_query = {
                executor.submit(self.test_rag_query, query): query 
                for query in queries_to_run
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    result = future.result()
                    result["query"] = query
                    results.append(result)
                except Exception as e:
                    results.append({
                        "query": query,
                        "success": False,
                        "error": str(e),
                        "response_time": 0
                    })
        
        total_time = time.time() - start_time
        
        return self.analyze_rag_results(results, total_time)
    
    def analyze_rag_results(self, results, total_time):
        """Analyze RAG performance results"""
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        cached = [r for r in successful if r.get("from_cache")]
        
        if not successful:
            return {"error": "No successful queries"}
        
        response_times = [r["response_time"] for r in successful]
        processing_times = [r["processing_time"] for r in successful]
        similarity_scores = [r["similarity_score"] for r in successful]
        
        return {
            "summary": {
                "total_queries": len(results),
                "successful_queries": len(successful),
                "failed_queries": len(failed),
                "success_rate": len(successful) / len(results) * 100,
                "cache_hit_rate": len(cached) / len(successful) * 100 if successful else 0,
                "total_time": total_time,
                "queries_per_second": len(successful) / total_time,
                "concurrent_capability": "Excellent" if len(successful) > 40 else "Good"
            },
            "performance": {
                "avg_response_time": statistics.mean(response_times),
                "min_response_time": min(response_times),
                "max_response_time": max(response_times),
                "p95_response_time": statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else max(response_times),
                "avg_processing_time": statistics.mean(processing_times),
                "avg_similarity_score": statistics.mean(similarity_scores)
            },
            "quality_metrics": {
                "avg_answer_length": statistics.mean([r["answer_length"] for r in successful]),
                "avg_context_docs": statistics.mean([r["context_docs"] for r in successful]),
                "high_similarity_rate": len([r for r in successful if r["similarity_score"] > 0.7]) / len(successful) * 100
            }
        }

def main():
    print("🎯 Lightning-Serve RAG Performance Testing")
    print("=" * 60)

    parser = argparse.ArgumentParser(description="Run RAG performance tests")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--collection", default=None, help="Target collection name. If omitted, picks first available.")
    args = parser.parse_args()

    # Check if server is running
    try:
        health = requests.get(f"{args.base_url}/health", timeout=5)
        if health.status_code != 200:
            print("❌ Server not running! Start with: uvicorn app.main:app --reload")
            return
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return

    # Determine a valid collection to test
    try:
        cols_resp = requests.get(f"{args.base_url}/rag/collections", timeout=5)
        if cols_resp.status_code == 200:
            collections = cols_resp.json().get("collections", [])
        else:
            collections = []
    except Exception as e:
        print(f"❌ Failed to fetch collections: {e}")
        return

    chosen_collection = args.collection
    if not chosen_collection:
        if not collections:
            print("⚠️ No collections found. Upload documents first via /rag/upload-documents before running performance tests.")
            return
        chosen_collection = collections[0]["name"]

    print(f"✅ Using collection: {chosen_collection}\n")

    tester = RAGPerformanceTest(base_url=args.base_url, collection=chosen_collection)

    # Test scenarios
    scenarios = [
        {"queries": 20, "workers": 3, "name": "Light RAG Load Test"},
        {"queries": 40, "workers": 5, "name": "Medium RAG Load Test"},
        {"queries": 30, "workers": 10, "name": "High Concurrency RAG Test"}
    ]

    for scenario in scenarios:
        print(f"📊 {scenario['name']}")
        print("-" * 50)

        results = tester.concurrent_rag_test(
            num_queries=scenario['queries'],
            max_workers=scenario['workers']
        )

        summary = results['summary']
        performance = results['performance']
        quality = results['quality_metrics']

        # Key performance indicators
        print(f"🚀 Queries per second: {summary['queries_per_second']:.2f}")
        print(f"✅ Success rate: {summary['success_rate']:.1f}%")
        print(f"⚡ Avg response time: {performance['avg_response_time']:.3f}s")
        print(f"🎯 Avg similarity score: {performance['avg_similarity_score']:.3f}")
        print(f"💾 Cache hit rate: {summary['cache_hit_rate']:.1f}%")
        print(f"📄 Avg answer length: {quality['avg_answer_length']:.0f} chars")
        print(f"🔍 High relevance rate: {quality['high_similarity_rate']:.1f}%")
        print()

if __name__ == "__main__":
    main()