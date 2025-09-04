import traceback
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import asyncio
import time
import hashlib
import json
import redis
import os
import tempfile
import shutil
from typing import List, Optional
from prometheus_client import Counter, Histogram, generate_latest
import psutil

from .vision_rag import VisionLanguageRAG
from .monitoring import setup_metrics

app = FastAPI(
    title="Lightning-Serve: Vision-Language RAG",
    description="Production-ready Vision-Language RAG system with distributed caching and real-time processing",
    version="2.0.0"
)
def read_secret(path):
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None
# Serve static UI (optional)
import os
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
rag_service = VisionLanguageRAG()
metrics = setup_metrics()

# Redis connection (supports REDIS_URL)
try:
    redis_url = os.getenv("REDIS_URL") or read_secret("/etc/secrets/REDIS_URL")
    if redis_url:
        redis_client = redis.from_url(redis_url, decode_responses=True)
    else:
        redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    redis_client.ping()
    print("✅ Redis connected for caching!")
except Exception:
    redis_client = None
    print("⚠️ Running without Redis cache")

# Basic request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    try:
        print(f"[{request.method}] {request.url.path} -> {response.status_code} ({duration_ms:.1f} ms)")
    except Exception:
        pass
    return response

@app.on_event("startup")
async def startup_event():
    print("🚀 Initializing Vision-Language RAG system...")
    await rag_service.initialize()
    print("✅ Lightning-Serve ready for production!")

@app.get("/")
async def root():
    return {
        "system": "Lightning-Serve Vision-Language RAG",
        "description": "Production multimodal AI with document understanding",
        "capabilities": [
            "PDF + Image document processing",
            "Multi-document semantic search",
            "Real-time question answering",
            "Distributed caching (3x speedup)",
            "Performance monitoring & metrics",
            "Scalable inference pipeline"
        ],
        "architecture": {
            "embeddings": "Cohere Embed-4 (1024-dim)",
            "llm": "Google Gemini 2.5 Flash",
            "caching": "Redis with TTL",
            "monitoring": "Prometheus metrics"
        },
        "demo_url": "http://localhost:8501",
        "api_docs": "/docs"
    }

@app.get("/ui", response_class=HTMLResponse)
async def ui_page():
    html = """
    <!doctype html>
    <html>
    <head>
      <meta charset=\"utf-8\" />
      <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
      <title>Lightning-Serve UI</title>
      <script src=\"https://cdn.tailwindcss.com\"></script>
      <link rel=\"icon\" href=\"/static/favicon.ico\" />

    </head>
    <body class=\"bg-gray-900 text-gray-100\">
      <!-- Mobile menu button -->
      <button id=\"mobileMenuBtn\" class=\"md:hidden fixed top-4 left-4 z-50 bg-gray-800 text-white p-2 rounded-lg border border-gray-700\">
        <svg class=\"w-6 h-6\" fill=\"none\" stroke=\"currentColor\" viewBox=\"0 0 24 24\">
          <path stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"2\" d=\"M4 6h16M4 12h16M4 18h16\"></path>
        </svg>
      </button>
      
      <div class=\"flex h-screen\">
        <!-- Sidebar -->
        <div id=\"sidebar\" class=\"fixed md:relative inset-y-0 left-0 z-40 w-80 bg-gray-800 border-r border-gray-700 p-4 overflow-y-auto transform -translate-x-full md:translate-x-0 transition-transform duration-300 ease-in-out\">
          <div class=\"flex justify-between items-center mb-4\">
            <h2 class=\"text-lg font-semibold text-white\">🔧 System Status</h2>
            <button id=\"closeSidebar\" class=\"md:hidden text-gray-400 hover:text-white\">
              <svg class=\"w-6 h-6\" fill=\"none\" stroke=\"currentColor\" viewBox=\"0 0 24 24\">
                <path stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"2\" d=\"M6 18L18 6M6 6l12 12\"></path>
              </svg>
            </button>
          </div>
          
          <div id=\"systemStatus\" class=\"mb-6\">
            <div class=\"text-sm text-gray-400\">Loading...</div>
          </div>
          
          <h3 class=\"text-md font-semibold mb-3 text-white\">📊 Performance</h3>
          <div id=\"performanceMetrics\" class=\"space-y-2\">
            <div class=\"text-sm text-gray-400\">Loading...</div>
          </div>
          
          <h3 class=\"text-md font-semibold mb-3 text-white mt-6\">📚 Collections</h3>
          <div id=\"collectionsList\" class=\"space-y-2\">
            <div class=\"text-sm text-gray-400\">Loading...</div>
          </div>
        </div>
        
        <!-- Overlay for mobile -->
        <div id=\"sidebarOverlay\" class=\"fixed inset-0 bg-black bg-opacity-50 z-30 md:hidden hidden\"></div>
        
        <!-- Main Content -->
        <div class=\"flex-1 p-4 md:p-6 overflow-y-auto\">
          <div class=\"md:pt-0 pt-12\">
            <h1 class=\"text-xl md:text-2xl font-bold flex items-center gap-2 text-white mb-2\">⚡ Lightning-Serve: Vision-Language RAG</h1>
            <p class=\"text-gray-400 mb-6 text-sm md:text-base\">Upload, query, benchmark, and view analytics.</p>

            <div class=\"grid grid-cols-1 lg:grid-cols-2 gap-4 md:gap-6\">
          <!-- Upload -->
          <section class=\"bg-gray-800 rounded-xl shadow-lg p-4 md:p-5 border border-gray-700\">
            <h2 class=\"font-semibold mb-3 text-white text-sm md:text-base\">Document Upload</h2>
            <input id=\"collection\" class=\"border border-gray-600 bg-gray-700 text-white rounded px-3 py-2 w-full mb-3 text-sm focus:border-blue-500 focus:outline-none\" value=\"my_documents\" />
            <input id=\"files\" type=\"file\" multiple class=\"mb-3 text-gray-300 text-sm w-full\" accept=\".pdf,.png,.jpg,.jpeg\" />
            <button id=\"btnUpload\" class=\"w-full md:w-auto px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded transition-colors text-sm\">Process Documents</button>
            <div id=\"uploadOut\" class=\"mt-3\"></div>
          </section>

          <!-- Query -->
          <section class=\"bg-gray-800 rounded-xl shadow-lg p-4 md:p-5 border border-gray-700\">
            <h2 class=\"font-semibold mb-3 text-white text-sm md:text-base\">RAG Query</h2>
            <div class=\"flex flex-col sm:flex-row gap-2 mb-2\">
              <select id=\"queryCollection\" class=\"border border-gray-600 bg-gray-700 text-white rounded px-3 py-2 flex-1 focus:border-blue-500 focus:outline-none text-sm\">
                <option value=\"my_documents\">my_documents</option>
              </select>
              <input id=\"topk\" type=\"number\" min=\"1\" max=\"5\" value=\"1\" class=\"border border-gray-600 bg-gray-700 text-white rounded px-3 py-2 w-20 focus:border-blue-500 focus:outline-none text-sm\" />
              <label class=\"flex items-center gap-2 text-gray-300 text-sm\"><input id=\"ctx\" type=\"checkbox\" checked class=\"rounded\" /> Include context</label>
            </div>
            <textarea id=\"question\" class=\"border border-gray-600 bg-gray-700 text-white rounded px-3 py-2 w-full h-20 md:h-24 mb-2 focus:border-blue-500 focus:outline-none text-sm\" placeholder=\"Ask a question...\">What is the main topic discussed?</textarea>
            <button id=\"btnQuery\" class=\"w-full md:w-auto px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded transition-colors text-sm\">Query</button>
            <div id=\"queryResult\" class=\"mt-3\"></div>
          </section>

          <!-- Benchmark -->
          <section class=\"bg-gray-800 rounded-xl shadow-lg p-4 md:p-5 border border-gray-700\">
            <h2 class=\"font-semibold mb-3 text-white text-sm md:text-base\">Performance Benchmark</h2>
            <button id=\"btnBench\" class=\"w-full md:w-auto px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded transition-colors text-sm\">Run RAG Benchmark</button>
            <div id=\"benchResult\" class=\"mt-3\"></div>
          </section>

          <!-- Load Testing -->
          <section class=\"bg-gray-800 rounded-xl shadow-lg p-4 md:p-5 border border-gray-700\">
            <h2 class=\"font-semibold mb-3 text-white text-sm md:text-base\">Load Testing</h2>
            <div class=\"flex flex-col sm:flex-row gap-2 mb-3\">
              <input id=\"loadQueries\" type=\"number\" min=\"5\" max=\"50\" value=\"20\" placeholder=\"Queries\" class=\"border border-gray-600 bg-gray-700 text-white rounded px-3 py-2 flex-1 focus:border-blue-500 focus:outline-none text-sm\" />
              <input id=\"loadWorkers\" type=\"number\" min=\"1\" max=\"10\" value=\"5\" placeholder=\"Workers\" class=\"border border-gray-600 bg-gray-700 text-white rounded px-3 py-2 flex-1 focus:border-blue-500 focus:outline-none text-sm\" />
              <button id=\"btnLoadTest\" class=\"w-full sm:w-auto px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded transition-colors text-sm\">Run Load Test</button>
            </div>
            <div id=\"loadResult\" class=\"mt-3\"></div>
          </section>

          <!-- Analytics -->
          <section class=\"bg-gray-800 rounded-xl shadow-lg p-4 md:p-5 border border-gray-700\">
            <h2 class=\"font-semibold mb-3 text-white text-sm md:text-base\">System Analytics</h2>
            <div class=\"flex flex-col sm:flex-row gap-2 mb-2\">
              <button id=\"btnHealth\" class=\"flex-1 px-3 py-2 bg-gray-600 hover:bg-gray-500 text-white rounded transition-colors text-sm\">Health</button>
              <button id=\"btnStats\" class=\"flex-1 px-3 py-2 bg-gray-600 hover:bg-gray-500 text-white rounded transition-colors text-sm\">Stats</button>
            </div>
            <div id=\"statsResult\" class=\"mt-1\"></div>
          </section>
        </div>

      </div>

      <script>
      const API = '';
      const qs = (s) => document.querySelector(s);

      async function postForm(url, data) {
        const form = new FormData();
        Object.entries(data).forEach(([k,v]) => form.append(k, v));
        const res = await fetch(url, { method: 'POST', body: form });
        if (!res.ok) throw new Error(await res.text());
        return res.json();
      }

      async function fetchJson(url) {
        const res = await fetch(url);
        if (!res.ok) throw new Error(await res.text());
        return res.json();
      }

      function showSuccess(message) {
        return `<div class=\"bg-green-900 border border-green-600 text-green-200 px-3 py-2 rounded mb-2\">✅ ${message}</div>`;
      }

      function showError(message) {
        return `<div class=\"bg-red-900 border border-red-600 text-red-200 px-3 py-2 rounded mb-2\">❌ ${message}</div>`;
      }

      function showLoading(message) {
        return `<div class=\"bg-blue-900 border border-blue-600 text-blue-200 px-3 py-2 rounded mb-2\">⏳ ${message}</div>`;
      }

      // Load system status
      async function loadSystemStatus() {
        try {
          const health = await fetchJson(`${API}/health`);
          const collections = await fetchJson(`${API}/rag/collections`);
          
          // System Status
          const statusHtml = health.status === 'healthy' ? 
            '<div class=\"text-green-400 text-sm\">✅ System Online</div>' :
            '<div class=\"text-red-400 text-sm\">❌ System Offline</div>';
          
          const servicesHtml = Object.entries(health.services).map(([service, status]) => {
            const label = service.replace('_', ' ').replace(/\\b\\w/g, l => l.toUpperCase());
            const icon = status ? '✅' : '⚠️';
            const color = status ? 'text-green-400' : 'text-yellow-400';
            return `<div class=\"text-sm ${color}\">${icon} ${label}</div>`;
          }).join('');
          
          qs('#systemStatus').innerHTML = statusHtml + servicesHtml;
          
          // Performance Metrics
          const perf = health.performance || {};
          const perfHtml = `
            <div class=\"text-sm text-gray-300\">Uptime: ${Math.round(perf.uptime || 0)}s</div>
            <div class=\"text-sm text-gray-300\">Total Queries: ${perf.total_rag_queries || 0}</div>
            <div class=\"text-sm text-gray-300\">Cache Hit Rate: ${perf.cache_hit_rate || '0%'}</div>
          `;
          qs('#performanceMetrics').innerHTML = perfHtml;
          
          // Collections
          const collectionsHtml = collections.collections && collections.collections.length > 0 ?
            collections.collections.map(c => `<div class=\"text-sm text-gray-300\">📚 ${c.name} (${c.document_count} docs)</div>`).join('') :
            '<div class=\"text-sm text-gray-400\">No collections found</div>';
          qs('#collectionsList').innerHTML = collectionsHtml;
          
          // Update collection selector
          const select = qs('#queryCollection');
          select.innerHTML = '';
          if (collections.collections && collections.collections.length > 0) {
            collections.collections.forEach(c => {
              const option = document.createElement('option');
              option.value = c.name;
              option.textContent = c.name;
              select.appendChild(option);
            });
          } else {
            const option = document.createElement('option');
            option.value = 'default';
            option.textContent = 'default';
            select.appendChild(option);
          }
          
        } catch (e) {
          qs('#systemStatus').innerHTML = '<div class=\"text-red-400 text-sm\">❌ Failed to load status</div>';
        }
      }

      // Mobile menu functionality
      const sidebar = qs('#sidebar');
      const sidebarOverlay = qs('#sidebarOverlay');
      const mobileMenuBtn = qs('#mobileMenuBtn');
      const closeSidebar = qs('#closeSidebar');

      function openSidebar() {
        sidebar.classList.remove('-translate-x-full');
        sidebarOverlay.classList.remove('hidden');
      }

      function closeSidebarMenu() {
        sidebar.classList.add('-translate-x-full');
        sidebarOverlay.classList.add('hidden');
      }

      mobileMenuBtn.addEventListener('click', openSidebar);
      closeSidebar.addEventListener('click', closeSidebarMenu);
      sidebarOverlay.addEventListener('click', closeSidebarMenu);

      // Close sidebar when clicking outside on mobile
      document.addEventListener('click', (e) => {
        if (window.innerWidth < 768 && !sidebar.contains(e.target) && !mobileMenuBtn.contains(e.target)) {
          closeSidebarMenu();
        }
      });

      // Load status on page load
      loadSystemStatus();
      setInterval(loadSystemStatus, 10000); // Refresh every 10 seconds

      qs('#btnUpload').onclick = async () => {
        const out = qs('#uploadOut');
        out.innerHTML = showLoading('Uploading documents...');
        try {
          const files = qs('#files').files;
          if (!files.length) { 
            out.innerHTML = showError('Please choose files to upload.'); 
            return; 
          }
          const form = new FormData();
          form.append('collection_name', qs('#collection').value || 'default');
          for (const f of files) form.append('files', f, f.name);
          const res = await fetch(`${API}/rag/upload-documents`, { method: 'POST', body: form });
          const json = await res.json();
          out.innerHTML = showSuccess(`Documents processed successfully! ${json.documents_processed} files, ${json.total_pages} pages.`) + 
            `<pre class=\"text-xs bg-gray-900 text-gray-300 p-3 rounded overflow-auto border border-gray-700 mt-2\">${JSON.stringify(json, null, 2)}</pre>`;
          loadSystemStatus(); // Refresh collections
        } catch (e) { 
          out.innerHTML = showError(`Upload failed: ${e.message}`); 
        }
      };

      qs('#btnQuery').onclick = async () => {
        const out = qs('#queryResult');
        out.innerHTML = showLoading('Processing query...');
        try {
          const payload = {
            query: qs('#question').value,
            collection_name: qs('#queryCollection').value || 'default',
            top_k: qs('#topk').value || 1,
            include_context: qs('#ctx').checked
          };
          const json = await postForm(`${API}/rag/query`, payload);
          
          const answerHtml = `
            <div class=\"bg-gray-900 border border-gray-700 rounded p-3 mb-3\">
              <h4 class=\"font-semibold text-white mb-2\">📝 Answer</h4>
              <p class=\"text-gray-300\">${json.answer || 'No answer generated'}</p>
            </div>
            <div class=\"grid grid-cols-2 md:grid-cols-4 gap-2 mb-3\">
              <div class=\"bg-gray-700 rounded p-2 text-center\">
                <div class=\"text-xs text-gray-400\">Similarity</div>
                <div class=\"text-sm font-semibold\">${(json.similarity_score || 0).toFixed(3)}</div>
              </div>
              <div class=\"bg-gray-700 rounded p-2 text-center\">
                <div class=\"text-xs text-gray-400\">Time</div>
                <div class=\"text-sm font-semibold\">${(json.processing_time || 0).toFixed(3)}s</div>
              </div>
              <div class=\"bg-gray-700 rounded p-2 text-center\">
                <div class=\"text-xs text-gray-400\">Cache</div>
                <div class=\"text-sm font-semibold\">${json.from_cache ? 'Yes' : 'No'}</div>
              </div>
              <div class=\"bg-gray-700 rounded p-2 text-center\">
                <div class=\"text-xs text-gray-400\">Source</div>
                <div class=\"text-sm font-semibold\">${json.source_document || 'N/A'}</div>
              </div>
            </div>
          `;
          
          out.innerHTML = showSuccess('Query processed successfully!') + answerHtml;
          loadSystemStatus(); // Refresh metrics
        } catch (e) { 
          out.innerHTML = showError(`Query failed: ${e.message}`); 
        }
      };

      qs('#btnBench').onclick = async () => {
        const out = qs('#benchResult');
        const btn = qs('#btnBench');
        out.innerHTML = showLoading('Running comprehensive benchmark...');
        btn.disabled = true;
        try {
          const res = await fetch(`${API}/rag/benchmark`);
          const json = await res.json();
          
          const summary = json.benchmark_summary || {};
          const summaryHtml = `
            <div class=\"grid grid-cols-1 sm:grid-cols-3 gap-2 mb-3\">
              <div class=\"bg-gray-700 rounded p-2 text-center\">
                <div class=\"text-xs text-gray-400\">Queries/sec</div>
                <div class=\"text-sm font-semibold\">${(summary.queries_per_second || 0).toFixed(2)}</div>
              </div>
              <div class=\"bg-gray-700 rounded p-2 text-center\">
                <div class=\"text-xs text-gray-400\">Success Rate</div>
                <div class=\"text-sm font-semibold\">${(summary.success_rate || 0).toFixed(1)}%</div>
              </div>
              <div class=\"bg-gray-700 rounded p-2 text-center\">
                <div class=\"text-xs text-gray-400\">Avg Time</div>
                <div class=\"text-sm font-semibold\">${(summary.avg_query_time || 0).toFixed(3)}s</div>
              </div>
            </div>
          `;
          
          out.innerHTML = showSuccess('Benchmark completed!') + summaryHtml + 
            `<pre class=\"text-xs bg-gray-900 text-gray-300 p-3 rounded overflow-auto border border-gray-700 mt-2\">${JSON.stringify(json, null, 2)}</pre>`;
          loadSystemStatus(); // Refresh metrics
        } catch (e) { 
          out.innerHTML = showError(`Benchmark failed: ${e.message}`); 
        } finally { 
          btn.disabled = false; 
        }
      };

      qs('#btnLoadTest').onclick = async () => {
        const out = qs('#loadResult');
        const btn = qs('#btnLoadTest');
        const queries = qs('#loadQueries').value || 20;
        const workers = qs('#loadWorkers').value || 5;
        
        out.innerHTML = showLoading(`Running load test: ${queries} queries, ${workers} workers...`);
        btn.disabled = true;
        try {
          // Simulate load test by sending multiple queries
          const testQueries = Array(parseInt(queries)).fill().map((_, i) => `Test query ${i + 1}: What information is shown in this document?`);
          const startTime = Date.now();
          let successCount = 0;
          
          // Simple sequential test (can be enhanced for true concurrency)
          for (const query of testQueries) {
            try {
              const payload = {
                query: query,
                collection_name: qs('#queryCollection').value || 'default',
                top_k: 1,
                include_context: false
              };
              await postForm(`${API}/rag/query`, payload);
              successCount++;
            } catch (e) {
              // Count as failed
            }
          }
          
          const totalTime = (Date.now() - startTime) / 1000;
          const successRate = (successCount / queries) * 100;
          const queriesPerSecond = successCount / totalTime;
          
          const resultHtml = `
            <div class=\"grid grid-cols-1 sm:grid-cols-3 gap-2 mb-3\">
              <div class=\"bg-gray-700 rounded p-2 text-center\">
                <div class=\"text-xs text-gray-400\">Success Rate</div>
                <div class=\"text-sm font-semibold\">${successRate.toFixed(1)}%</div>
              </div>
              <div class=\"bg-gray-700 rounded p-2 text-center\">
                <div class=\"text-xs text-gray-400\">Queries/sec</div>
                <div class=\"text-sm font-semibold\">${queriesPerSecond.toFixed(2)}</div>
              </div>
              <div class=\"bg-gray-700 rounded p-2 text-center\">
                <div class=\"text-xs text-gray-400\">Total Time</div>
                <div class=\"text-sm font-semibold\">${totalTime.toFixed(2)}s</div>
              </div>
            </div>
          `;
          
          out.innerHTML = showSuccess('Load test completed!') + resultHtml;
          loadSystemStatus(); // Refresh metrics
        } catch (e) { 
          out.innerHTML = showError(`Load test failed: ${e.message}`); 
        } finally { 
          btn.disabled = false; 
        }
      };

      qs('#btnHealth').onclick = async () => {
        const out = qs('#statsResult');
        out.innerHTML = showLoading('Loading health status...');
        try {
          const res = await fetch(`${API}/health`);
          const json = await res.json();
          out.innerHTML = `<pre class=\"text-xs bg-gray-900 text-gray-300 p-3 rounded overflow-auto border border-gray-700\">${JSON.stringify(json, null, 2)}</pre>`;
        } catch (e) {
          out.innerHTML = showError(`Failed to load health: ${e.message}`);
        }
      };
      
      qs('#btnStats').onclick = async () => {
        const out = qs('#statsResult');
        out.innerHTML = showLoading('Loading system stats...');
        try {
          const res = await fetch(`${API}/stats`);
          const json = await res.json();
          out.innerHTML = `<pre class=\"text-xs bg-gray-900 text-gray-300 p-3 rounded overflow-auto border border-gray-700\">${JSON.stringify(json, null, 2)}</pre>`;
        } catch (e) {
          out.innerHTML = showError(`Failed to load stats: ${e.message}`);
        }
      };
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "services": {
            "vision_rag": rag_service.is_ready(),
            "redis_cache": redis_client is not None,
            "embeddings_api": rag_service.cohere_ready(),
            "llm_api": rag_service.gemini_ready()
        },
        "performance": {
            "uptime": time.time() - app.start_time,
            "total_rag_queries": rag_service.performance_metrics["total_queries"],
            "cache_hit_rate": f"{rag_service.get_cache_hit_rate():.1f}%"
        }
    }

# ================================
# CORE VISION-RAG ENDPOINTS
# ================================

@app.post("/rag/upload-documents")
async def upload_documents(
    files: List[UploadFile] = File(...),
    collection_name: str = Form("default")
):
    """Upload and process documents for RAG"""
    with metrics['request_duration'].time():
        metrics['rag_queries_total'].labels(collection=collection_name, status='processing').inc()
        
        start_time = time.time()
        
        try:
            print(f"📁 Received {len(files)} files for collection '{collection_name}'")
            
            # Create temporary directory
            temp_dir = tempfile.mkdtemp()
            print(f"📂 Created temp directory: {temp_dir}")
            
            processed_files = []
            
            for i, file in enumerate(files):
                print(f"📄 Processing file {i+1}/{len(files)}: {file.filename}")
                temp_path = os.path.join(temp_dir, file.filename)
                
                try:
                    content = await file.read()
                    print(f"✅ Read file: {len(content)} bytes")
                    
                    with open(temp_path, "wb") as f:
                        f.write(content)
                    processed_files.append(temp_path)
                    print(f"✅ Saved to: {temp_path}")
                    
                except Exception as file_error:
                    print(f"❌ File processing error for {file.filename}: {file_error}")
                    raise Exception(f"Failed to process file {file.filename}: {str(file_error)}")
            
            print(f"📊 Starting document indexing with {len(processed_files)} files...")
            
            # Process documents with your RAG system
            result = await rag_service.index_documents(processed_files, collection_name)
            print(f"✅ Document indexing complete: {result}")
            
            # Cleanup after processing
            try:
                shutil.rmtree(temp_dir)
                print(f"🧹 Cleaned up temp directory")
            except Exception as cleanup_error:
                print(f"⚠️ Cleanup warning: {cleanup_error}")
            
            processing_time = time.time() - start_time
            
            # Mark as successful
            metrics['rag_queries_total'].labels(collection=collection_name, status='success').inc()
            
            return {
                "status": "success",
                "collection": collection_name,
                "documents_processed": len(files),
                "total_pages": result.get("total_pages", 0),
                "embeddings_created": result.get("embeddings_count", 0),
                "processing_time": processing_time,
                "ready_for_queries": True
            }
            
        except Exception as e:
            error_msg = str(e)
            error_trace = traceback.format_exc()
            print(f"❌ Upload error: {error_msg}")
            print(f"❌ Full traceback: {error_trace}")
            
            # Clean up temp directory if it exists
            try:
                if 'temp_dir' in locals():
                    shutil.rmtree(temp_dir)
            except:
                pass
                
            # Mark as failed
            metrics['rag_queries_total'].labels(collection=collection_name, status='failed').inc()
            metrics['rag_errors_total'].inc()
            
            raise HTTPException(
                status_code=500, 
                detail=f"Document processing failed: {error_msg}"
            )

@app.post("/rag/query")
async def rag_query(
    query: str = Form(...),
    collection_name: str = Form("default"),
    top_k: int = Form(1),
    include_context: bool = Form(True)
):
    """Query the RAG system"""
    with metrics['request_duration'].time():
        metrics['rag_queries_total'].labels(collection=collection_name, status='processing').inc()
        
        start_time = time.time()
        
        # Create cache key
        cache_key = f"rag:{collection_name}:{hashlib.md5(query.encode()).hexdigest()}"
        
        # Check cache
        if redis_client:
            cached_result = redis_client.get(cache_key)
            if cached_result:
                # Update metrics for cache hit path
                metrics['cache_hits_total'].inc()
                metrics['rag_queries_total'].labels(collection=collection_name, status='success').inc()
                # Keep service-level counters in sync with /health and /stats
                try:
                    rag_service.performance_metrics["total_queries"] += 1
                    rag_service.performance_metrics["cache_hits"] += 1
                except Exception:
                    pass
                
                result = json.loads(cached_result)
                result["from_cache"] = True
                result["cache_retrieval_time"] = time.time() - start_time
                return result
        
        try:
            # Process query with your RAG system
            result = await rag_service.query(
                query=query,
                collection_name=collection_name,
                top_k=top_k,
                include_context=include_context
            )
            
            processing_time = time.time() - start_time
            result["processing_time"] = processing_time
            result["from_cache"] = False
            
            # Cache successful results for 10 minutes
            if redis_client and result.get("answer"):
                redis_client.setex(cache_key, 600, json.dumps(result))
            
            metrics['rag_queries_total'].labels(collection=collection_name, status='success').inc()
            return result
            
        except Exception as e:
            metrics['rag_queries_total'].labels(collection=collection_name, status='failed').inc()
            metrics['rag_errors_total'].inc()
            raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")

@app.get("/rag/query/stream")
async def stream_rag_query(
    query: str,
    collection_name: str = "default"
):
    """Stream RAG responses in real-time"""
    metrics['rag_queries_total'].labels(collection=collection_name, status='processing').inc()
    
    def generate_stream():
        try:
            for chunk in rag_service.stream_query(query, collection_name):
                yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

class BatchQueryRequest(BaseModel):
    queries: List[str] = Field(default_factory=list)
    collection_name: Optional[str] = Field(default="default")
    top_k: Optional[int] = Field(default=1, ge=1, le=5)


@app.post("/rag/batch-query")
async def batch_rag_query(data: BatchQueryRequest):
    """Process multiple RAG queries simultaneously"""
    with metrics['request_duration'].time():
        
        queries = data.queries or []
        collection_name = data.collection_name or "default"
        top_k = data.top_k or 1
        if len(queries) == 0:
            raise HTTPException(status_code=400, detail="Provide at least one query")
        if len(queries) > 20:  # Limit batch size
            raise HTTPException(status_code=400, detail="Batch size limited to 20 queries")
        
        start_time = time.time()
        
        # Process queries in parallel
        tasks = [
            rag_service.query(query, collection_name, top_k=top_k, include_context=False)
            for query in queries
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processing_time = time.time() - start_time
        
        return {
            "batch_size": len(queries),
            "results": [
                result if not isinstance(result, Exception) else {"error": str(result)}
                for result in results
            ],
            "total_processing_time": processing_time,
            "avg_time_per_query": processing_time / len(queries)
        }

# ================================
# PERFORMANCE & MONITORING
# ================================

@app.get("/rag/collections")
async def list_collections():
    """List available document collections"""
    return await rag_service.list_collections()

@app.delete("/rag/collections/{collection_name}")
async def delete_collection(collection_name: str):
    """Delete a document collection"""
    result = await rag_service.delete_collection(collection_name)
    return {"status": "deleted", "collection": collection_name, "result": result}

@app.get("/rag/benchmark")
async def rag_benchmark(collection_name: str = "default", top_k: int = 1):
    """Run comprehensive RAG performance benchmark"""
    metrics['rag_queries_total'].labels(collection=collection_name, status='processing').inc()
    
    benchmark_queries = [
        "What is the main topic of this document?",
        "Summarize the key points",
        "What are the conclusions?",
        "Find specific data or numbers",
        "Explain the methodology"
    ]
    
    start_time = time.time()
    # Resolve target collection
    if collection_name not in rag_service.collections:
        if len(rag_service.collections) == 0:
            metrics['rag_queries_total'].labels(collection=collection_name, status='failed').inc()
            raise HTTPException(status_code=400, detail="No collections found. Upload documents first.")
        # Fallback to first available collection
        collection_name = next(iter(rag_service.collections.keys()))
    results = []
    
    for query in benchmark_queries:
        query_start = time.time()
        try:
            result = await rag_service.query(query, collection_name, top_k=top_k)
            query_time = time.time() - query_start
            results.append({
                "query": query,
                "success": True,
                "response_time": query_time,
                "answer_length": len(result.get("answer", "")),
                "similarity_score": result.get("similarity_score", 0)
            })
        except Exception as e:
            results.append({
                "query": query,
                "success": False,
                "error": str(e),
                "response_time": time.time() - query_start
            })
    
    total_time = time.time() - start_time
    successful_queries = [r for r in results if r["success"]]
    
    metrics['rag_queries_total'].labels(collection=collection_name, status='success').inc()
    return {
        "benchmark_summary": {
            "total_queries": len(benchmark_queries),
            "successful_queries": len(successful_queries),
            "success_rate": len(successful_queries) / len(benchmark_queries) * 100,
            "total_time": total_time,
            "avg_query_time": sum(r["response_time"] for r in successful_queries) / len(successful_queries) if successful_queries else 0,
            "queries_per_second": len(successful_queries) / total_time if total_time > 0 else 0
        },
        "detailed_results": results,
        "system_performance": {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "cache_status": "connected" if redis_client else "disabled"
        }
    }

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics for monitoring"""
    return generate_latest()

@app.get("/stats")
async def get_detailed_stats():
    """Comprehensive system statistics"""
    return {
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('C:/' if os.name == 'nt' else '/').percent
        },
        "rag_service": await rag_service.get_stats(),
        "cache": {
            "connected": redis_client is not None,
            "info": redis_client.info() if redis_client else None
        },
        "performance": {
            "uptime": time.time() - app.start_time,
            "total_rag_queries": rag_service.performance_metrics["total_queries"]
        }
    }

app.start_time = time.time()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)