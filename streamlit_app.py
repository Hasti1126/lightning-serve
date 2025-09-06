import streamlit as st
import requests
import json
import time
from PIL import Image
import io
import os
if 'benchmark_running' not in st.session_state:
    st.session_state['benchmark_running'] = False

st.set_page_config(
    page_title="Lightning-Serve: Vision-Language RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stSidebar {
        background-color: #262730;
    }
    .stSelectbox > div > div {
        background-color: #262730;
        color: #fafafa;
    }
    .stTextInput > div > div > input {
        background-color: #262730;
        color: #fafafa;
    }
    .stTextArea > div > div > textarea {
        background-color: #262730;
        color: #fafafa;
    }
    .stSlider > div > div > div > div {
        background-color: #262730;
    }
    .stCheckbox > div > div {
        background-color: #262730;
    }
    .stFileUploader > div > div {
        background-color: #262730;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border: none;
    }
    .stButton > button:hover {
        background-color: #0d5a8a;
    }
    .stSuccess {
        background-color: #1e3a1e;
        border: 1px solid #4caf50;
    }
    .stError {
        background-color: #3a1e1e;
        border: 1px solid #f44336;
    }
    .stWarning {
        background-color: #3a3a1e;
        border: 1px solid #ff9800;
    }
    .stInfo {
        background-color: #1e2a3a;
        border: 1px solid #2196f3;
    }
    .metric-container {
        background-color: #262730;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #404040;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Lightning-Serve: Production Vision-Language RAG")
st.markdown("*Enterprise-grade document understanding with real-time processing*")

# API_BASE = "http://localhost:8000"


API_BASE = os.getenv("API_BASE", "http://localhost:8000")

# Sidebar: System Status
with st.sidebar:
    st.header("🔧 System Status")
    
    try:
        health = requests.get(f"{API_BASE}/health").json()
        
        if health["status"] == "healthy":
            st.success("✅ System Online")
            
            service_labels = {
                "vision_rag": "Vision-RAG Core",
                "redis_cache": "Redis Cache",
                "embeddings_api": "Embeddings (Cohere) API",
                "llm_api": "LLM (Gemini) API"
            }
            for service, status in health["services"].items():
                label = service_labels.get(service, service.replace('_', ' ').title())
                if status:
                    st.success(f"✅ {label}")
                else:
                    st.warning(f"⚠️ {label}")
        
        # Performance metrics
        st.subheader("📊 Performance")
        perf = health.get("performance", {})
        st.metric("Uptime", f"{perf.get('uptime', 0):.0f}s")
        st.metric("Total Queries", perf.get('total_rag_queries', 0))
        st.metric("Cache Hit Rate", perf.get('cache_hit_rate', '0%'))
        
    except:
        st.error("❌ API Server Offline")

# Main Interface
tab1, tab2, tab3, tab4 = st.tabs([
    "📄 Document Upload", 
    "🔍 RAG Query", 
    "⚡ Performance Test",
    "📈 Analytics"
])

with tab1:
    st.header("📄 Document Upload & Indexing")
    st.markdown("Upload PDFs and images to create searchable knowledge base")
    
    collection_name = st.text_input("Collection Name:", "my_documents")
    uploaded_files = st.file_uploader(
        "Upload Documents (PDFs, Images)",
        type=['pdf', 'png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("🚀 Process Documents"):
        with st.spinner("Processing documents..."):
            try:
                files = [
                    ('files', (file.name, file.read(), file.type))
                    for file in uploaded_files
                ]
                
                response = requests.post(
                    f"{API_BASE}/rag/upload-documents",
                    data={"collection_name": collection_name},
                    files=files
                )
                
                if response.status_code == 200:
                    result = response.json()
                    st.success("✅ Documents processed successfully!")
                    
                    # Display processing results
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Documents", result.get('documents_processed', 0))
                    with col2:
                        st.metric("Total Pages", result.get('total_pages', 0))
                    with col3:
                        st.metric("Embeddings", result.get('embeddings_created', 0))
                    with col4:
                        st.metric("Processing Time", f"{result.get('processing_time', 0):.2f}s")
                    
                    st.json(result)
                else:
                    st.error(f"Upload failed: {response.text}")
                    
            except Exception as e:
                st.error(f"Error: {e}")

with tab2:
    st.header("🔍 Vision-Language RAG Query")
    st.markdown("Ask questions about your uploaded documents")
    
    # Collection selector
    try:
        collections_response = requests.get(f"{API_BASE}/rag/collections")
        if collections_response.status_code == 200:
            collections = collections_response.json().get("collections", [])
            if collections:
                collection_names = [c["name"] for c in collections]
                selected_collection = st.selectbox("Select Collection:", collection_names)
                
                # Show collection info
                selected_info = next(c for c in collections if c["name"] == selected_collection)
                st.info(f"📚 {selected_info['document_count']} documents, {selected_info['total_pages']} pages")
            else:
                st.warning("No collections found. Upload documents first!")
                selected_collection = "default"
        else:
            selected_collection = st.text_input("Collection Name:", "default")
    except:
        selected_collection = st.text_input("Collection Name:", "default")
    
    # Query interface
    user_query = st.text_area(
        "Ask a question about your documents:",
        "What is the main topic discussed in this document?"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        top_k = st.slider("Number of relevant documents to retrieve:", 1, 5, 1)
    with col2:
        include_context = st.checkbox("Include context", True)
    
    if user_query and st.button("🔍 Query Documents"):
        with st.spinner("Searching and generating answer..."):
            try:
                response = requests.post(
                    f"{API_BASE}/rag/query",
                    data={
                        "query": user_query,
                        "collection_name": selected_collection,
                        "top_k": top_k,
                        "include_context": include_context
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.success("✅ Query processed successfully!")
                    
                    # Display answer prominently
                    st.subheader("📝 Answer")
                    st.write(result.get("answer", "No answer generated"))
                    
                    # Performance metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Similarity", f"{result.get('similarity_score', 0):.3f}")
                    with col2:
                        st.metric("Processing Time", f"{result.get('processing_time', 0):.3f}s")
                    with col3:
                        st.metric("From Cache", "Yes" if result.get('from_cache') else "No")
                    with col4:
                        st.metric("Source Doc", result.get('source_document', 'N/A'))
                    
                    # Context documents
                    if result.get('context_documents'):
                        st.subheader("📄 Retrieved Documents")
                        for i, doc in enumerate(result['context_documents']):
                            st.write(f"{i+1}. **{doc['document']}** (similarity: {doc['similarity']:.3f})")
                    
                    # Full result JSON
                    with st.expander("🔍 Full Response Details"):
                        st.json(result)
                        
                else:
                    st.error(f"Query failed: {response.text}")
                    
            except Exception as e:
                st.error(f"Error: {e}")

with tab3:
    st.header("⚡ Performance Benchmarking")
    st.markdown("Test system performance and scalability")

    col1, col2 = st.columns(2)

    # --------------------------
    # 🚀 RAG Benchmark (LEFT)
    # --------------------------
    with col1:
        st.subheader("🚀 RAG Benchmark")

        # Button to trigger benchmark
        run_benchmark = st.button(
            "Run RAG Performance Test", 
            disabled=st.session_state['benchmark_running']
        )

        # Container where results will always be rendered
        benchmark_container = st.container()

        if run_benchmark:
            st.session_state['benchmark_running'] = True
            with st.spinner("Running comprehensive benchmark..."):
                try:
                    response = requests.get(f"{API_BASE}/rag/benchmark")
                    result = response.json()

                    with benchmark_container:
                        st.success("✅ Benchmark Complete")

                        summary = result.get("benchmark_summary", {})

                        # Key metrics
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("⚡ Queries/Second", f"{summary.get('queries_per_second', 0):.2f}")
                        with col_b:
                            st.metric("✅ Success Rate", f"{summary.get('success_rate', 0):.1f}%")
                            st.progress(int(summary.get("success_rate", 0)))
                        with col_c:
                            st.metric("⏱ Avg Query Time", f"{summary.get('avg_query_time', 0):.3f}s")

                        # Detailed results
                        st.subheader("📊 Detailed Results")
                        for i, result_item in enumerate(result.get("detailed_results", []), start=1):
                            with st.expander(f"Query {i}: {result_item['query']}"):
                                if result_item.get("success"):
                                    st.success(f"✅ Success in {result_item['response_time']:.3f}s")
                                else:
                                    st.error(f"❌ Failed: {result_item.get('error', 'Unknown error')}")

                        # System snapshot after benchmark
                        try:
                            stats_response = requests.get(f"{API_BASE}/stats")
                            stats = stats_response.json()
                            st.subheader("🖥️ System Snapshot After Benchmark")

                            c1, c2, c3 = st.columns(3)
                            with c1:
                                st.metric("🖥 CPU Usage", f"{stats['system']['cpu_percent']:.1f}%")
                            with c2:
                                st.metric("💾 Memory Usage", f"{stats['system']['memory_percent']:.1f}%")
                            with c3:
                                cache_status = "✅ Connected" if stats['cache']['connected'] else "⚠️ Disconnected"
                                st.metric("📦 Cache", cache_status)
                        except Exception:
                            st.warning("Could not fetch system snapshot.")
                except Exception as e:
                    with benchmark_container:
                        st.error(f"Benchmark failed: {e}")
                finally:
                    st.session_state['benchmark_running'] = False

    # --------------------------
    # 📊 System Load Test (RIGHT)
    # --------------------------
    # with col2:
    #     st.subheader("📊 System Load Test")

    #     num_queries = st.slider("Number of test queries:", 1,10,5)
    #     concurrent_queries = st.slider("Concurrent queries:", 1, 10,2)

    #     if st.button("Run Load Test"):
    #         st.info(f"Running {num_queries} queries with {concurrent_queries} workers...")

    #         test_queries = [
    #             f"Test query {i}: What information is shown in this document?"
    #             for i in range(num_queries)
    #         ]

    #         # Progress bar + live text
    #         progress = st.progress(0)
    #         status_text = st.empty()

    #         def run_query(query):
    #             try:
    #                 query_start = time.time()
    #                 r = requests.post(
    #                     f"{API_BASE}/rag/query",
    #                     data={
    #                         "query": query,
    #                         "collection_name": "default",
    #                         "top_k": 1,
    #                         "include_context": "true"
    #                     },
    #                     timeout=60
    #                 )
    #                 dt = time.time() - query_start
    #                 if r.status_code == 200:
    #                     return True, dt
    #                 return False, 0
    #             except Exception:
    #                 st.error(f"Request failed: {e}")


    #         start_time = time.time()
    #         success_count, total_response_time = 0, 0

    #         import concurrent.futures
    #         with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_queries) as ex:
    #             futures = [ex.submit(run_query, q) for q in test_queries]
    #             for i, fut in enumerate(concurrent.futures.as_completed(futures)):
    #                 ok, dt = fut.result()
    #                 if ok:
    #                     success_count += 1
    #                     total_response_time += dt

    #                 # update progress
    #                 progress.progress((i + 1) / num_queries)
    #                 status_text.text(f"Completed {i+1}/{num_queries} queries...")

    #         total_time = time.time() - start_time

    #         st.success("✅ Load test complete!")

    #         col_x, col_y, col_z = st.columns(3)
    #         with col_x:
    #             st.metric("Success Rate", f"{(success_count/num_queries*100):.1f}%")
    #         with col_y:
    #             st.metric("Queries/Second", f"{success_count/total_time:.2f}")
    #         with col_z:
    #             st.metric("Avg Response", f"{total_response_time/max(success_count,1):.3f}s")


with tab4:
    st.header("📈 System Analytics")
    
    if st.button("🔄 Refresh Analytics"):
        try:
            stats_response = requests.get(f"{API_BASE}/stats")
            stats = stats_response.json()
            
            # System health
            st.subheader("💻 System Health")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("CPU Usage", f"{stats['system']['cpu_percent']:.1f}%")
            with col2:
                st.metric("Memory Usage", f"{stats['system']['memory_percent']:.1f}%")
            with col3:
                st.metric("Disk Usage", f"{stats['system']['disk_usage']:.1f}%")
            
            # RAG performance
            st.subheader("🎯 RAG Performance")
            rag_stats = stats.get('rag_service', {})
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Collections", rag_stats.get('collections', 0))
            with col2:
                st.metric("Total Documents", rag_stats.get('total_documents', 0))
            with col3:
                st.metric("Cache Hit Rate", f"{rag_stats.get('performance', {}).get('cache_hits', 0)/max(rag_stats.get('performance', {}).get('total_queries', 1), 1)*100:.1f}%")
            with col4:
                st.metric("Avg Query Time", f"{rag_stats.get('performance', {}).get('avg_query_time', 0):.3f}s")
            
            # Full stats
            with st.expander("🔍 Complete System Stats"):
                st.json(stats)
                
        except Exception as e:
            st.error(f"Failed to fetch stats: {e}")

# Footer
st.markdown("---")
st.markdown("""
### 🚀 **System Highlights:**
- **Real-time document understanding** with sub-second response times
- **Distributed caching** for 3x performance improvement  
- **Production monitoring** with Prometheus metrics
- **Scalable architecture** ready for enterprise deployment
- **Multimodal processing** combining vision and language understanding

*Built to demonstrate production-ready AI infrastructure capabilities*
""")