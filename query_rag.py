#!/usr/bin/env python3
"""
RAG Query Interface for Austin Planning Documents
Combines vector search (ChromaDB) with LLM to answer planning questions.
"""
import os
import sys
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# System prompt for planning questions
SYSTEM_PROMPT = """You are an expert on Austin, Texas planning and zoning regulations. 
You answer questions based on official planning commission documents, board of adjustment decisions, and zoning cases.

When answering:
- Cite specific case numbers when relevant (e.g., "Case C14-2023-0045")
- Mention addresses and locations when available
- Distinguish between approved/denied applications
- Note any conditions or restrictions
- If information is not in the provided context, say so clearly
- Be concise but comprehensive"""

def setup_chromadb(persist_dir):
    """Initialize ChromaDB client and collection."""
    client = chromadb.PersistentClient(path=str(persist_dir))
    
    # Get embedding function
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )
    
    # Get collection
    collection = client.get_collection(
        name="austin_planning_docs",
        embedding_function=openai_ef
    )
    
    logger.info(f"✅ Connected to ChromaDB collection ({collection.count():,} chunks)")
    return collection

def vector_search(collection, query, n_results=10, filters=None):
    """Perform vector search on collection."""
    logger.info(f"🔍 Searching for: '{query}'")
    
    kwargs = {
        "query_texts": [query],
        "n_results": n_results
    }
    
    if filters:
        kwargs["where"] = filters
    
    results = collection.query(**kwargs)
    logger.info(f"📄 Found {len(results['ids'][0])} relevant chunks")
    
    return results

def format_context(results):
    """Format retrieved chunks into LLM context."""
    context_parts = []
    
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    
    for i, (chunk_text, metadata) in enumerate(zip(documents, metadatas), 1):
        # Build source information
        source_info = []
        
        if metadata.get('case_0'):
            source_info.append(f"Case: {metadata['case_0']}")
        if metadata.get('address_0'):
            source_info.append(f"Address: {metadata['address_0']}")
        if metadata.get('parcel_0'):
            source_info.append(f"Parcel: {metadata['parcel_0']}")
        
        # Format chunk
        if source_info:
            header = f"[Source {i}] {' | '.join(source_info)}"
        else:
            header = f"[Source {i}]"
        
        context_parts.append(f"{header}\n{chunk_text}")
    
    return "\n\n---\n\n".join(context_parts)

def ask_llm_anthropic(query, context):
    """Call Claude API with retrieved context."""
    from anthropic import Anthropic
    
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    user_prompt = f"""QUESTION: {query}

RELEVANT DOCUMENTS:
{context}

Based on the documents above, please answer the question. Cite specific case numbers and addresses when mentioned in the sources."""
    
    logger.info("🤖 Asking Claude...")
    
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": user_prompt
        }]
    )
    
    return message.content[0].text

def ask_llm_openai(query, context):
    """Call OpenAI API with retrieved context."""
    from openai import OpenAI
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    user_prompt = f"""QUESTION: {query}

RELEVANT DOCUMENTS:
{context}

Based on the documents above, please answer the question. Cite specific case numbers and addresses when mentioned in the sources."""
    
    logger.info("🤖 Asking GPT-4...")
    
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=2000,
        temperature=0.3
    )
    
    return response.choices[0].message.content

def query_rag(query, collection, n_results=10, llm_provider="anthropic", show_context=False):
    """Main RAG query function."""
    # Vector search
    results = vector_search(collection, query, n_results=n_results)
    
    # Format context
    context = format_context(results)
    
    if show_context:
        logger.info("\n" + "="*80)
        logger.info("RETRIEVED CONTEXT:")
        logger.info("="*80)
        print(context)
        logger.info("="*80 + "\n")
    
    # Call LLM
    if llm_provider == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            logger.error("❌ ANTHROPIC_API_KEY not set")
            return None
        answer = ask_llm_anthropic(query, context)
    else:  # openai
        if not os.getenv("OPENAI_API_KEY"):
            logger.error("❌ OPENAI_API_KEY not set")
            return None
        answer = ask_llm_openai(query, context)
    
    return answer

def interactive_mode(collection, llm_provider="anthropic"):
    """Interactive query mode."""
    logger.info("\n" + "="*80)
    logger.info("🏛️  Austin Planning RAG - Interactive Mode")
    logger.info("="*80)
    logger.info("Ask questions about Austin zoning, planning cases, and regulations.")
    logger.info("Type 'quit' or 'exit' to leave.\n")
    
    while True:
        try:
            query = input("❓ Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                logger.info("\n👋 Goodbye!")
                break
            
            if not query:
                continue
            
            print()  # Spacing
            answer = query_rag(query, collection, llm_provider=llm_provider)
            
            if answer:
                logger.info("\n" + "="*80)
                logger.info("💡 ANSWER:")
                logger.info("="*80)
                print(answer)
                logger.info("="*80 + "\n")
        
        except KeyboardInterrupt:
            logger.info("\n\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"❌ Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Query Austin Planning RAG System")
    parser.add_argument("query", nargs="*", help="Question to ask")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic", 
                       help="LLM provider (default: anthropic)")
    parser.add_argument("--results", "-k", type=int, default=10, help="Number of results to retrieve")
    parser.add_argument("--show-context", action="store_true", help="Show retrieved context")
    
    args = parser.parse_args()
    
    # Setup ChromaDB
    workspace_dir = Path("/Users/chuck/Documents/antigravity_workspace")
    chroma_dir = workspace_dir / "chroma_db"
    
    if not chroma_dir.exists():
        logger.error(f"❌ ChromaDB not found at {chroma_dir}")
        logger.error("Run generate_embeddings.py first")
        sys.exit(1)
    
    collection = setup_chromadb(chroma_dir)
    
    # Interactive or single query mode
    if args.interactive:
        interactive_mode(collection, llm_provider=args.provider)
    else:
        if not args.query:
            parser.print_help()
            sys.exit(1)
        
        query = " ".join(args.query)
        answer = query_rag(query, collection, n_results=args.results, 
                          llm_provider=args.provider, show_context=args.show_context)
        
        if answer:
            logger.info("\n" + "="*80)
            logger.info("💡 ANSWER:")
            logger.info("="*80)
            print(answer)
            logger.info("="*80 + "\n")

if __name__ == "__main__":
    main()
