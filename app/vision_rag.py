import os
import io
import base64
import numpy as np
import PIL
import fitz  # PyMuPDF
import tqdm
from google import genai
import cohere
from dotenv import load_dotenv
import asyncio
import tempfile
import time
import hashlib
import json
import shutil
from typing import List, Dict, Tuple, Optional

load_dotenv()

class VisionLanguageRAG:
    def __init__(self):
        self.gemini_api_key = os.getenv("GOOGLE_API_KEY")
        self.cohere_api_key = os.getenv("COHERE_API_KEY")
        self.client = None
        self.co = None
        self.ready = False
        
        # In-memory document store (production would use vector DB)
        self.collections = {}
        self.query_cache = {}
        self.performance_metrics = {
            "total_queries": 0,
            "cache_hits": 0,
            "avg_query_time": 0,
            "total_documents": 0
        }
        
    async def initialize(self):
        """Initialize API clients"""
        try:
            if self.gemini_api_key:
                self.client = genai.Client(api_key=self.gemini_api_key)
                print("✅ Gemini client initialized")
            else:
                print("⚠️ No Gemini API key found")
                
            if self.cohere_api_key:
                self.co = cohere.ClientV2(api_key=self.cohere_api_key)
                print("✅ Cohere client initialized")
            else:
                print("⚠️ No Cohere API key found")
            
            self.ready = bool(self.client and self.co)
            
            if not self.ready:
                print("⚠️ Running in demo mode without API keys")
                
        except Exception as e:
            print(f"❌ RAG initialization failed: {e}")
            self.ready = False
    
    def is_ready(self):
        return self.ready
    
    def cohere_ready(self):
        return bool(self.co)
    
    def gemini_ready(self):
        return bool(self.client)
    
    # ================================
    # YOUR CORE FUNCTIONS (Enhanced)
    # ================================
    
    def resize_image(self, pil_image, max_pixels=1568*1568):
        """Optimize image size for processing"""
        org_width, org_height = pil_image.size
        if org_width * org_height > max_pixels:
            scale_factor = (max_pixels / (org_width * org_height)) ** 0.5
            new_width = int(org_width * scale_factor)
            new_height = int(org_height * scale_factor)
            pil_image.thumbnail((new_width, new_height))
    
    def base64_from_image(self, img_path):
        """Convert image to base64 for API"""
        pil_image = PIL.Image.open(img_path)
        img_format = pil_image.format if pil_image.format else "PNG"
        self.resize_image(pil_image)
        
        with io.BytesIO() as img_buffer:
            pil_image.save(img_buffer, format=img_format)
            img_buffer.seek(0)
            img_data = f"data:image/{img_format.lower()};base64,"+base64.b64encode(img_buffer.read()).decode("utf-8")
        return img_data
    
    def convert_pdf_to_images_pymupdf(self, pdf_path, output_folder, dpi=300):
        """Extract images from PDF pages"""
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        doc = fitz.open(pdf_path)
        paths = []
        
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=dpi)
            image_filename = os.path.join(output_folder, f'page_{i+1}.png')
            pix.save(image_filename)
            paths.append(image_filename)
            
        doc.close()
        return paths
    
    async def process_images(self, input_paths, collection_name="default"):
        """Create embeddings for image collection"""
        if not self.ready:
            # Demo mode - return mock embeddings
            return input_paths, np.random.rand(len(input_paths), 1024)
        
        def is_image_file(filename):
            IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
            return filename.lower().endswith(IMG_EXTS)
        
        # Collect all image paths
        img_paths = []
        for path in input_paths:
            if os.path.isfile(path) and is_image_file(path):
                img_paths.append(path)
            elif os.path.isdir(path):
                img_paths.extend([
                    os.path.join(path, f) for f in os.listdir(path) 
                    if is_image_file(f)
                ])
        
        print(f"📸 Processing {len(img_paths)} images...")
        embeddings = []
        
        for img_path in img_paths:
            try:
                api_input_document = {
                    "content": [
                        {"type": "image", "image": self.base64_from_image(img_path)},
                    ]
                }
                
                # Run in executor to avoid blocking
                loop = asyncio.get_event_loop()
                api_response = await loop.run_in_executor(
                    None,
                    lambda: self.co.embed(
                        model="embed-v4.0",
                        input_type="search_document",
                        embedding_types=["float"],
                        inputs=[api_input_document],
                    )
                )
                
                emb = np.asarray(api_response.embeddings.float[0])
                embeddings.append(emb)
                
            except Exception as e:
                print(f"⚠️ Failed to process {img_path}: {e}")
                continue
        
        return img_paths, np.vstack(embeddings) if embeddings else np.array([])
    
    async def search(self, question, img_paths, doc_embeddings, top_k=3):
        """Semantic search for relevant images"""
        if not self.ready:
            # Demo mode
            return [(img_paths[0], PIL.Image.open(img_paths[0]), 0.95)] if img_paths else []
        
        loop = asyncio.get_event_loop()
        api_response = await loop.run_in_executor(
            None,
            lambda: self.co.embed(
                model="embed-v4.0",
                input_type="search_query",
                embedding_types=["float"],
                texts=[question],
            )
        )
        
        query_emb = np.asarray(api_response.embeddings.float[0])
        cos_sim_scores = np.dot(query_emb, doc_embeddings.T)
        
        # Get top-k results
        top_indices = np.argsort(cos_sim_scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            img_path = img_paths[idx]
            image = PIL.Image.open(img_path)
            image.thumbnail((800, 800))  # Optimize size
            similarity = float(cos_sim_scores[idx])
            results.append((img_path, image, similarity))
        
        return results
    
    async def answer(self, question, img_path, context_images=None):
        """Generate answer using Gemini with context"""
        if not self.ready:
            return f"Demo answer for: {question} (based on {os.path.basename(img_path)})"
        
        # Enhanced prompt with context
        prompt_text = f"""You are an expert document analyst. Answer the question based on the provided image(s).

Question: {question}

Instructions:
- Provide a comprehensive, accurate answer
- Include specific details from the image
- If multiple images are provided, synthesize information across them
- Cite relevant visual elements that support your answer
- Be concise but thorough

Answer:"""

        prompt = [prompt_text, PIL.Image.open(img_path)]
        
        # Add context images if provided
        if context_images:
            for ctx_img_path, _, _ in context_images[:2]:  # Limit context
                prompt.append(PIL.Image.open(ctx_img_path))
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
        )
        
        return response.text
    
    # ================================
    # PRODUCTION RAG PIPELINE
    # ================================
    
    async def index_documents(self, file_paths: List[str], collection_name: str):
        """Index documents into collection"""
        start_time = time.time()
        
        # Create persistent storage directory for this collection
        collection_dir = os.path.join("collections", collection_name)
        os.makedirs(collection_dir, exist_ok=True)
        
        # Process PDFs to images and copy files to persistent storage
        all_image_paths = []
        total_pages = 0
        
        for file_path in file_paths:
            if file_path.lower().endswith('.pdf'):
                # Create subdirectory for PDF pages
                pdf_name = os.path.splitext(os.path.basename(file_path))[0]
                pdf_dir = os.path.join(collection_dir, pdf_name)
                os.makedirs(pdf_dir, exist_ok=True)
                
                pdf_images = self.convert_pdf_to_images_pymupdf(file_path, pdf_dir)
                all_image_paths.extend(pdf_images)
                total_pages += len(pdf_images)
            else:
                # Copy image file to collection directory
                filename = os.path.basename(file_path)
                persistent_path = os.path.join(collection_dir, filename)
                shutil.copy2(file_path, persistent_path)
                all_image_paths.append(persistent_path)
                total_pages += 1
        
        # Create embeddings
        img_paths, embeddings = await self.process_images(all_image_paths, collection_name)
        
        # Store in collection
        self.collections[collection_name] = {
            "image_paths": img_paths,
            "embeddings": embeddings,
            "created_at": time.time(),
            "document_count": len(file_paths),
            "total_pages": total_pages
        }
        
        self.performance_metrics["total_documents"] += len(file_paths)
        
        processing_time = time.time() - start_time
        
        return {
            "total_pages": total_pages,
            "embeddings_count": len(embeddings),
            "processing_time": processing_time,
            "collection_size": len(self.collections[collection_name]["image_paths"])
        }
    
    async def query(self, query: str, collection_name: str, top_k: int = 3, include_context: bool = True):
        """Main RAG query pipeline"""
        start_time = time.time()
        self.performance_metrics["total_queries"] += 1
        
        # Check if collection exists
        if collection_name not in self.collections:
            raise Exception(f"Collection '{collection_name}' not found. Upload documents first.")
        
        collection = self.collections[collection_name]
        img_paths = collection["image_paths"]
        embeddings = collection["embeddings"]
        
        if len(img_paths) == 0:
            raise Exception("No documents in collection")
        
        # Search for relevant images
        search_results = await self.search(query, img_paths, embeddings, top_k)
        
        if not search_results:
            raise Exception("No relevant documents found")
        
        # Get best match
        best_img_path, best_image, best_similarity = search_results[0]
        
        # Generate answer with context
        context_images = search_results[1:] if include_context else None
        answer = await self.answer(query, best_img_path, context_images)
        
        processing_time = time.time() - start_time
        
        # Update performance metrics
        self.performance_metrics["avg_query_time"] = (
            (self.performance_metrics["avg_query_time"] * (self.performance_metrics["total_queries"] - 1) + processing_time) 
            / self.performance_metrics["total_queries"]
        )
        
        return {
            "query": query,
            "answer": answer,
            "source_document": os.path.basename(best_img_path),
            "similarity_score": best_similarity,
            "context_documents": [
                {
                    "document": os.path.basename(path),
                    "similarity": float(sim)
                }
                for path, _, sim in search_results
            ],
            "collection": collection_name,
            "processing_time": processing_time,
            "system_info": {
                "embeddings_model": "Cohere Embed-4",
                "llm_model": "Gemini 2.5 Flash",
                "top_k_retrieved": len(search_results)
            }
        }
    
    def stream_query(self, query: str, collection_name: str):
        """Stream RAG response in chunks"""
        try:
            # This would implement streaming with your RAG system
            # For now, simulate streaming
            answer_parts = [
                "Analyzing documents...",
                "Finding relevant context...", 
                "Generating comprehensive answer...",
                "Final answer based on document analysis"
            ]
            
            for i, part in enumerate(answer_parts):
                yield {
                    "chunk_id": i,
                    "content": part,
                    "progress": (i + 1) / len(answer_parts),
                    "timestamp": time.time()
                }
                time.sleep(0.5)  # Simulate processing time
                
        except Exception as e:
            yield {"error": str(e)}
    
    async def list_collections(self):
        """List all document collections"""
        return {
            "collections": [
                {
                    "name": name,
                    "document_count": data["document_count"],
                    "total_pages": data["total_pages"],
                    "created_at": data["created_at"],
                    "size_mb": len(data["embeddings"]) * 1024 * 4 / (1024*1024) if len(data["embeddings"]) > 0 else 0
                }
                for name, data in self.collections.items()
            ]
        }
    
    async def delete_collection(self, collection_name: str):
        """Delete a document collection"""
        if collection_name in self.collections:
            del self.collections[collection_name]
            return True
        return False
    
    async def get_stats(self):
        """Get comprehensive RAG statistics"""
        return {
            "ready": self.ready,
            "collections": len(self.collections),
            "total_documents": self.performance_metrics["total_documents"],
            "performance": self.performance_metrics,
            "apis": {
                "gemini": bool(self.client),
                "cohere": bool(self.co)
            }
        }
    
    def get_cache_hit_rate(self):
        """Calculate cache hit rate"""
        total = self.performance_metrics["total_queries"]
        hits = self.performance_metrics["cache_hits"]
        return (hits / max(total, 1)) * 100