import streamlit as st
import pandas as pd
import sqlite3
import re
import numpy as np
import faiss
import torchtext.vocab as vocab
import torch as torch
from groq import Groq
import pickle
import os
import tempfile
from datetime import datetime

# Configuration
FAISS_INDEX_PATH = "../embeddings/faiss_index.bin"
CHUNKS_DATA_PATH = "../embeddings/chunks_data.pkl"
METADATA_PATH = "../embeddings/metadata.pkl"
SQLITE_DB_PATH = "../db/reviews_database.db"

# Initialize Groq client
def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
    if not api_key:
        st.error("Please set GROQ_API_KEY in your environment variables or Streamlit secrets")
        return None
    return Groq(api_key=api_key)

# Database functions
def create_database_from_excel(uploaded_file):
    """Convert Excel file to SQLite database"""
    try:
        # Read Excel file
        df = pd.read_excel(uploaded_file)
        
        # Basic data cleaning
        df = clean_dataframe(df)
        
        # Create SQLite database
        conn = sqlite3.connect(SQLITE_DB_PATH)
        
        # Save to database
        df.to_sql('reviews', conn, if_exists='replace', index=False)
        
        # Create additional useful tables/views
        create_analytical_views(conn)
        
        conn.close()
        
        return df, True
    except Exception as e:
        st.error(f"Error creating database: {str(e)}")
        return None, False

def clean_dataframe(df):
    """Clean and standardize the dataframe"""
    # Fill NaN values
    df = df.fillna('')
    
    # Standardize column names (remove spaces, lowercase)
    df.columns = [col.strip().replace(' ', '_').lower() for col in df.columns]
    
    # Ensure essential columns exist, create if missing
    essential_columns = {
        'product_name': 'product_name',
        'review_content': 'review_content', 
        'rating': 'rating',
        'category': 'category',
        'user_name': 'user_name'
    }
    
    for essential_col, default_col in essential_columns.items():
        if essential_col not in df.columns:
            # Try to find similar columns
            similar_cols = [col for col in df.columns if essential_col in col]
            if similar_cols:
                df[essential_col] = df[similar_cols[0]]
            else:
                df[essential_col] = ''
    
    # Add sentiment analysis if not present
    if 'sentiment' not in df.columns:
        df['sentiment'] = df['rating'].apply(lambda x: 'positive' if float(x) >= 4 else 'negative' if float(x) <= 2 else 'neutral')
    
    return df

def create_analytical_views(conn):
    """Create useful views for analysis"""
    cursor = conn.cursor()
    
    # View for product summary
    cursor.execute('''
    CREATE VIEW IF NOT EXISTS product_summary AS
    SELECT 
        product_name,
        category,
        COUNT(*) as review_count,
        AVG(CAST(rating AS REAL)) as avg_rating,
        COUNT(CASE WHEN sentiment = 'positive' THEN 1 END) as positive_reviews,
        COUNT(CASE WHEN sentiment = 'negative' THEN 1 END) as negative_reviews
    FROM reviews
    GROUP BY product_name, category
    ''')
    
    # View for category analysis
    cursor.execute('''
    CREATE VIEW IF NOT EXISTS category_analysis AS
    SELECT 
        category,
        COUNT(*) as total_reviews,
        AVG(CAST(rating AS REAL)) as avg_rating,
        COUNT(DISTINCT product_name) as unique_products
    FROM reviews
    GROUP BY category
    ''')
    
    conn.commit()

def get_database_connection():
    """Get connection to SQLite database"""
    return sqlite3.connect(SQLITE_DB_PATH)

# FAISS and RAG functions
def build_document(row):
    """Enhanced document builder with structured formatting"""
    parts = []
    
    # Product information section
    parts.append(f"PRODUCT INFO:")
    parts.append(f"Name: {row['product_name']}")
    
    if 'category' in row and row['category']:
        parts.append(f"Category: {row['category']}")
    
    if all(col in row for col in ['discounted_price', 'actual_price', 'discount_percentage']):
        parts.append(f"Price: {row['discounted_price']} (Actual: {row['actual_price']}, Discount: {row['discount_percentage']})")
    
    if 'rating' in row and row['rating']:
        rating_count = row['rating_count'] if 'rating_count' in row else 'unknown'
        parts.append(f"Overall Rating: {row['rating']}/5 from {rating_count} reviews")
    
    # Product description
    if 'about_product' in row and pd.notna(row['about_product']) and str(row['about_product']).strip():
        parts.append(f"Description: {row['about_product']}")
    
    # Review section
    parts.append("REVIEW:")
    if 'user_name' in row and row['user_name']:
        sentiment = row['sentiment'] if 'sentiment' in row else 'unknown'
        parts.append(f"User: {row['user_name']} | Sentiment: {sentiment}")
    
    if 'review_title' in row and row['review_title']:
        parts.append(f"Title: {row['review_title']}")
    
    if 'review_content' in row and row['review_content']:
        parts.append(f"Content: {row['review_content']}")
    
    return " | ".join(map(str, parts))

def chunk_text(text, max_words=80):
    """Improved chunking that preserves section boundaries"""
    words = str(text).split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i+max_words])

def create_and_save_faiss_index(df):
    """Create FAISS index and save it locally with all associated data"""
    st.info("Creating FAISS index from uploaded data...")
    
    chunks, meta = [], []
    for _, row in df.iterrows():
        document = build_document(row)
        for chunk in chunk_text(document):
            chunks.append(chunk)
            meta_data = {
                "product_id": row.get("product_id", ""),
                "product_name": row.get("product_name", ""),
                "rating": row.get("rating", ""),
                "sentiment": row.get("sentiment", ""),
                "category": row.get("category", ""),
                "user_name": row.get("user_name", ""),
                "review_id": row.get("review_id", "")
            }
            # Add optional fields if they exist
            for field in ['discounted_price', 'actual_price', 'review_title']:
                if field in row:
                    meta_data[field] = row[field]
            
            meta.append(meta_data)

    st.success(f"Created {len(chunks)} chunks from {len(df)} reviews")

    # Initialize GloVe embeddings
    glove = vocab.GloVe(name="6B", dim=100)

    def text_to_vector(text, embeddings, dim=100):
        words = re.findall(r'\w+', str(text).lower())
        vectors = [embeddings[word] for word in words if word in embeddings.stoi]
        if len(vectors) == 0:
            return torch.zeros(dim)
        return torch.mean(torch.stack(vectors), dim=0)

    # Create embeddings for chunks
    with st.spinner("Creating embeddings..."):
        chunk_embeddings = torch.stack([text_to_vector(ch, glove, 100) for ch in chunks]).float()
        chunk_embeddings_np = chunk_embeddings.detach().cpu().numpy().astype("float32")

    dimension = chunk_embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(chunk_embeddings_np)

    # Save everything to disk
    with st.spinner("Saving FAISS index and data to disk..."):
        # Save FAISS index
        faiss.write_index(index, FAISS_INDEX_PATH)
        
        # Save chunks and metadata
        with open(CHUNKS_DATA_PATH, 'wb') as f:
            pickle.dump(chunks, f)
        
        with open(METADATA_PATH, 'wb') as f:
            pickle.dump(meta, f)
    
    st.success(f"Saved {len(chunks)} review chunks with metadata to local files")
    
    return index, chunks, meta

def load_faiss_index():
    """Load FAISS index and associated data from disk"""
    if not all(os.path.exists(path) for path in [FAISS_INDEX_PATH, CHUNKS_DATA_PATH, METADATA_PATH]):
        return None, None, None
    
    # Load FAISS index
    index = faiss.read_index(FAISS_INDEX_PATH)
    
    # Load chunks and metadata
    with open(CHUNKS_DATA_PATH, 'rb') as f:
        chunks = pickle.load(f)
    
    with open(METADATA_PATH, 'rb') as f:
        meta = pickle.load(f)
    
    return index, chunks, meta

# Search and query functions
def text_to_vector(text, embeddings, dim=100):
    """Text to vector conversion function"""
    words = re.findall(r'\w+', str(text).lower())
    vectors = [embeddings[word] for word in words if word in embeddings.stoi]
    if len(vectors) == 0:
        return torch.zeros(dim)
    return torch.mean(torch.stack(vectors), dim=0)

def search_reviews(query, k=5, product_id=None, category=None, min_rating=None, sentiment=None):
    """Enhanced search with multiple filters"""
    if 'index' not in st.session_state or st.session_state.index is None:
        st.error("FAISS index not loaded. Please process the data first.")
        return []
    
    glove = vocab.GloVe(name="6B", dim=100)
    q_emb = np.array([text_to_vector(query, glove, 100)]).astype("float32")
    
    # Determine how many results to retrieve initially
    initial_k = min(k * 3, len(st.session_state.chunks))
    scores, indices = st.session_state.index.search(q_emb, initial_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1 or idx >= len(st.session_state.chunks):
            continue
            
        entry = st.session_state.meta[idx]
        
        # Apply filters
        if product_id and entry.get("product_id") != product_id:
            continue
        if category and category.lower() not in str(entry.get("category", "")).lower():
            continue
        if min_rating and float(entry.get("rating", 0)) < min_rating:
            continue
        if sentiment and entry.get("sentiment", "").lower() != sentiment.lower():
            continue
            
        results.append((st.session_state.chunks[idx], entry, float(scores[0][i])))
        if len(results) >= k:
            break
    
    return [(chunk, metadata) for chunk, metadata, score in results]

def answer_query(query, product_id=None, category=None, min_rating=None, sentiment=None, 
                query_type="general", require_citations=True):
    """Enhanced query answering with scenario-based prompt engineering"""
    
    retrieved = search_reviews(query, k=8, product_id=product_id, 
                             category=category, min_rating=min_rating, sentiment=sentiment)
    
    if not retrieved:
        return "No relevant reviews found matching your criteria."
    
    # Build structured context
    context_parts = []
    for i, (chunk, metadata) in enumerate(retrieved):
        context_parts.append(f"REVIEW {i+1}:")
        context_parts.append(f"Content: {chunk}")
        context_parts.append(f"Metadata: Product: {metadata.get('product_name', 'Unknown')}, Rating: {metadata.get('rating', 'Unknown')}, Sentiment: {metadata.get('sentiment', 'Unknown')}")
        context_parts.append("---")
    
    context = "\n".join(context_parts)
    
    # Scenario-based system prompts
    scenario_prompts = {
        "comparison": """
You are comparing products based on Amazon reviews. Analyze similarities, differences, strengths, and weaknesses.
Focus on: features, quality, value for money, user satisfaction, and common complaints.
Provide a balanced comparison with specific examples from reviews.
        """,
        "recommendation": """
You are providing product recommendations. Consider: overall ratings, sentiment analysis, specific features mentioned,
price-value ratio, and frequency of positive/negative comments. Highlight best options for different user needs.
        """,
        "summary": """
You are summarizing product reviews. Extract key themes: most praised features, common complaints, overall satisfaction,
and any recurring patterns. Provide a concise yet comprehensive summary.
        """,
        "general": """
You are a helpful assistant analyzing Amazon reviews. Answer questions accurately based only on the provided context.
Be specific, factual, and reference actual review content when possible.
        """
    }
    
    system_prompt = scenario_prompts.get(query_type, scenario_prompts["general"])
    
    # Enhanced user prompt with explicit instructions
    prompt = f"""
{system_prompt}

USER QUESTION: {query}

AVAILABLE REVIEW CONTEXT:
{context}

INSTRUCTIONS:
1. Answer based ONLY on the provided review context
2. Be specific and reference actual review content when possible
3. If multiple reviews mention similar points, note the consensus
4. For conflicting information, acknowledge both perspectives
5. If information is insufficient, clearly state what cannot be determined
6. Consider ratings, sentiment, and specific user experiences
7. {"Include specific review references (e.g., 'Review 3 mentions...')" if require_citations else "Provide a synthesized answer"}
8. Focus on what users actually experienced and reported

ANSWER:
"""
    
    try:
        client = get_groq_client()
        if not client:
            return "Error: Groq API client not available."
            
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1000
        )

        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error generating response: {str(e)}"

def query_database(query):
    """Execute SQL queries on the database and return results"""
    try:
        conn = get_database_connection()
        
        # Simple query classification
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['count', 'how many', 'number of']):
            # Count queries
            if 'product' in query_lower:
                result = pd.read_sql("SELECT COUNT(DISTINCT product_name) as product_count FROM reviews", conn)
                return f"There are {result['product_count'].iloc[0]} unique products in the database."
            elif 'review' in query_lower:
                result = pd.read_sql("SELECT COUNT(*) as review_count FROM reviews", conn)
                return f"There are {result['review_count'].iloc[0]} total reviews in the database."
        
        elif any(word in query_lower for word in ['average', 'avg']):
            # Average rating queries
            result = pd.read_sql("SELECT AVG(CAST(rating AS REAL)) as avg_rating FROM reviews", conn)
            return f"The average rating across all products is {result['avg_rating'].iloc[0]:.2f} stars."
        
        elif any(word in query_lower for word in ['category', 'categories']):
            # Category analysis
            result = pd.read_sql("""
                SELECT category, COUNT(*) as review_count, AVG(CAST(rating AS REAL)) as avg_rating 
                FROM reviews 
                GROUP BY category 
                ORDER BY review_count DESC
            """, conn)
            return result
        
        elif any(word in query_lower for word in ['top', 'best', 'highest']):
            # Top products
            result = pd.read_sql("""
                SELECT product_name, AVG(CAST(rating AS REAL)) as avg_rating, COUNT(*) as review_count
                FROM reviews 
                GROUP BY product_name 
                HAVING COUNT(*) >= 3
                ORDER BY avg_rating DESC 
                LIMIT 10
            """, conn)
            return result
        
        else:
            # General query - return sample data
            result = pd.read_sql("SELECT * FROM reviews LIMIT 5", conn)
            return result
            
    except Exception as e:
        return f"Database query error: {str(e)}"
    finally:
        conn.close()

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Review Analysis Chatbot",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Review Analysis Chatbot")
    st.markdown("Upload your Excel file with review data and get intelligent insights!")
    
    # Initialize session state
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'index' not in st.session_state:
        st.session_state.index = None
    if 'chunks' not in st.session_state:
        st.session_state.chunks = None
    if 'meta' not in st.session_state:
        st.session_state.meta = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader("Upload Excel File", type=['xlsx', 'xls'])
        
        if uploaded_file is not None:
            if st.button("Process Data"):
                with st.spinner("Processing your data..."):
                    # Create database
                    df, success = create_database_from_excel(uploaded_file)
                    
                    if success:
                        # Create FAISS index
                        index, chunks, meta = create_and_save_faiss_index(df)
                        
                        # Store in session state
                        st.session_state.index = index
                        st.session_state.chunks = chunks
                        st.session_state.meta = meta
                        st.session_state.processed = True
                        st.session_state.df = df
                        
                        st.success("Data processed successfully!")
        
        # Display data stats if processed
        if st.session_state.processed:
            st.header("📈 Data Overview")
            st.write(f"**Total Reviews:** {len(st.session_state.df)}")
            st.write(f"**Total Products:** {st.session_state.df['product_name'].nunique()}")
            st.write(f"**Total Categories:** {st.session_state.df['category'].nunique() if 'category' in st.session_state.df.columns else 'N/A'}")
            
            # Quick filters
            st.header("🔍 Quick Filters")
            st.session_state.selected_category = st.selectbox(
                "Filter by Category",
                ["All"] + list(st.session_state.df['category'].unique()) if 'category' in st.session_state.df.columns else ["All"]
            )
            
            st.session_state.min_rating = st.slider("Minimum Rating", 1.0, 5.0, 1.0, 0.5)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Review Analysis Chatbot")
        
        # Chat interface
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # User input
        if prompt := st.chat_input("Ask about your reviews..."):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing reviews..."):
                    # Apply filters
                    category = None
                    if hasattr(st.session_state, 'selected_category') and st.session_state.selected_category != "All":
                        category = st.session_state.selected_category
                    
                    min_rating = None
                    if hasattr(st.session_state, 'min_rating'):
                        min_rating = st.session_state.min_rating
                    
                    # Determine query type
                    query_lower = prompt.lower()
                    if any(word in query_lower for word in ['compare', 'vs', 'versus']):
                        query_type = "comparison"
                    elif any(word in query_lower for word in ['recommend', 'best', 'top']):
                        query_type = "recommendation"
                    elif any(word in query_lower for word in ['summary', 'overview', 'summarize']):
                        query_type = "summary"
                    else:
                        query_type = "general"
                    
                    # Get response
                    response = answer_query(
                        prompt, 
                        category=category, 
                        min_rating=min_rating,
                        query_type=query_type
                    )
                    
                    st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    with col2:
        st.header("📋 Sample Questions")
        
        st.markdown("""
        **General Questions:**
        - What are the most common complaints?
        - What features do users love most?
        - How is the product quality?
        
        **Product Analysis:**
        - Compare [Product A] and [Product B]
        - What do users say about battery life?
        - How is the customer service?
        
        **Recommendations:**
        - Recommend the best products for durability
        - Which products have the best value?
        - Top rated products in [category]
        
        **Data Insights:**
        - How many reviews are positive?
        - What's the average rating?
        - Most reviewed products
        """)
        
        # Quick action buttons
        st.header("⚡ Quick Actions")
        if st.button("Show Top Products"):
            result = query_database("top products")
            st.write(result)
        
        if st.button("Show Category Summary"):
            result = query_database("categories")
            st.write(result)
        
        if st.button("Show Data Overview"):
            result = query_database("count reviews")
            st.write(result)

if __name__ == "__main__":
    main()