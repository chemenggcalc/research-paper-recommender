# app.py
import os
import asyncio
import aiohttp
import logging
import arxiv
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
from lxml import etree
from Bio import Entrez

# Initialize Flask application
app = Flask(__name__)
CORS(app, resources={r"/search": {"origins": "*"}})
logging.basicConfig(level=logging.INFO)

# Configuration
MODEL_NAME = 'paraphrase-MiniLM-L3-v2'  # 17MB model
API_KEY = os.getenv('RENDER_API_KEY', 'your-secret-key-123')
MAX_RESULTS = 10
PAPER_SOURCES = ['arxiv', 'pubmed', 'semanticscholar', 'crossref', 'doaj', 'core']

# Initialize model and components
model = SentenceTransformer(MODEL_NAME)
executor = ThreadPoolExecutor(max_workers=4)

# ------------------- Paper Fetching Functions -------------------
def fetch_arxiv_papers(query: str) -> list:
    """Fetch papers from arXiv API"""
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=MAX_RESULTS,
            sort_by=arxiv.SortCriterion.Relevance
        )
        return [{
            "title": result.title,
            "abstract": result.summary,
            "url": result.pdf_url,
            "source": "arXiv",
            "authors": ", ".join([a.name for a in result.authors]),
            "published": result.published.isoformat()
        } for result in client.results(search)]
    except Exception as e:
        logging.error(f"arXiv Error: {str(e)}")
        return []

async def fetch_pubmed_papers(query: str) -> list:
    """Fetch papers from PubMed API"""
    try:
        Entrez.email = "your.email@example.com"  # Replace with your email
        handle = Entrez.esearch(db="pubmed", term=query, retmax=MAX_RESULTS)
        record = Entrez.read(handle)
        id_list = record["IdList"]

        papers = []
        async with aiohttp.ClientSession() as session:
            for pubmed_id in id_list:
                try:
                    async with session.get(
                        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                        params={
                            "db": "pubmed",
                            "id": pubmed_id,
                            "retmode": "xml"
                        }
                    ) as response:
                        xml_data = await response.text()
                        root = etree.fromstring(xml_data.encode())

                        papers.append({
                            "title": root.findtext('.//ArticleTitle') or "No title",
                            "abstract": " ".join([
                                e.text for e in root.findall('.//AbstractText') 
                                if e.text
                            ]),
                            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/",
                            "source": "PubMed",
                            "authors": ", ".join([
                                f"{a.findtext('LastName')}, {a.findtext('ForeName')}"
                                for a in root.findall('.//Author')
                                if a.findtext('LastName') and a.findtext('ForeName')
                            ]),
                            "published": root.findtext('.//PubMedPubDate/Year') or ""
                        })
                except Exception as e:
                    logging.error(f"PubMed Parse Error: {str(e)}")
        return papers
    except Exception as e:
        logging.error(f"PubMed API Error: {str(e)}")
        return []

async def fetch_semantic_scholar_papers(query: str) -> list:
    """Fetch papers from Semantic Scholar API"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={
                    "query": query,
                    "limit": MAX_RESULTS,
                    "fields": "title,abstract,url,authors,year"
                }
            ) as response:
                data = await response.json()
                return [{
                    "title": paper.get("title", "No title"),
                    "abstract": paper.get("abstract", "No abstract"),
                    "url": paper.get("url", "#"),
                    "source": "Semantic Scholar",
                    "authors": ", ".join([a["name"] for a in paper.get("authors", [])]),
                    "published": str(paper.get("year", ""))
                } for paper in data.get("data", [])]
    except Exception as e:
        logging.error(f"Semantic Scholar Error: {str(e)}")
        return []

async def fetch_crossref_papers(query: str) -> list:
    """Fetch papers from Crossref API"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.crossref.org/works",
                params={
                    "query": query,
                    "rows": MAX_RESULTS
                }
            ) as response:
                data = await response.json()
                return [{
                    "title": item.get("title", ["No title"])[0],
                    "abstract": item.get("abstract", "No abstract"),
                    "url": item.get("URL", "#"),
                    "source": "Crossref",
                    "authors": ", ".join([
                        f"{a.get('given', '')} {a.get('family', '')}"
                        for a in item.get("author", [])
                    ]),
                    "published": item.get("created", {}).get("date-time", "")
                } for item in data.get("message", {}).get("items", [])]
    except Exception as e:
        logging.error(f"Crossref Error: {str(e)}")
        return []

# ------------------- Core Processing Pipeline -------------------
async def process_papers(query: str) -> list:
    """Main processing pipeline"""
    try:
        # Parallel API fetching
        arxiv_results = await loop.run_in_executor(executor, fetch_arxiv_papers, query)
        other_sources = await asyncio.gather(
            fetch_pubmed_papers(query),
            fetch_semantic_scholar_papers(query),
            fetch_crossref_papers(query),
            fetch_doaj_papers(query),  # Implement similar to others
            fetch_core_papers(query)    # Implement similar to others
        )
        
        # Combine results
        all_papers = arxiv_results + [p for sublist in other_sources for p in sublist]
        if not all_papers:
            return []

        # Create DataFrame
        df = pd.DataFrame(all_papers)
        df = df.drop_duplicates('title', keep='first')
        df = df[df['abstract'].str.len() > 50]  # Filter low-quality abstracts

        # Generate embeddings in parallel
        df['combined_text'] = df['title'] + ". " + df['abstract']
        with ThreadPoolExecutor() as pool:
            embeddings = list(pool.map(
                lambda text: model.encode(text, convert_to_tensor=True).cpu().numpy(),
                df['combined_text']
            ))
        df['embedding'] = embeddings

        # Calculate similarity
        query_embedding = model.encode(query).reshape(1, -1)
        doc_embeddings = np.stack(df['embedding'].values)
        df['similarity'] = cosine_similarity(query_embedding, doc_embeddings)[0]

        # Sort and filter
        df = df.sort_values('similarity', ascending=False)
        df = df[df['similarity'] > 0.25]  # Adjust threshold as needed
        
        return df.head(MAX_RESULTS).to_dict('records')

    except Exception as e:
        logging.error(f"Processing Error: {str(e)}")
        return []

# ------------------- API Endpoints -------------------
@app.route('/search', methods=['POST'])
async def search_endpoint():
    """Main search endpoint"""
    # Authentication
    if request.headers.get('X-API-Key') != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    
    # Validate input
    data = request.get_json()
    query = data.get('query', '').strip()
    if len(query) < 3:
        return jsonify({"error": "Query must be at least 3 characters"}), 400
    
    try:
        results = await process_papers(query)
        return jsonify({
            "count": len(results),
            "results": results
        })
    except Exception as e:
        logging.error(f"API Error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model": MODEL_NAME,
        "sources": PAPER_SOURCES
    })

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    app.run(host='0.0.0.0', port=5000)
