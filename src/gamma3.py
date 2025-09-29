import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load data
df = pd.read_excel('../data/processedDF.xlsx')

# Load model
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m")
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m")

def answer_query_from_df(query, df, max_reviews=5):
    # Take top reviews
    reviews_to_use = df.head(max_reviews)
    
    # Build context
    context = ""
    for i, row in reviews_to_use.iterrows():
        context += f"Product: {row['product_name']}\n"
        context += f"Rating: {row['rating']}, Sentiment: {row['sentiment']}\n"
        context += f"Review: {row['review_content']}\n"
        context += "-----\n"
    
    # Build prompt
    prompt = f"Answer this question using the review data: {query}\n\nREVIEWS:\n{context}\nAnswer:"
    
    try:
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=300,
                temperature=0.2,
                repetition_penalty=2.0,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.replace(prompt, "").strip()
        
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit app
st.title("Product Review Chatbot")
st.write("Ask me anything about the product reviews!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Show user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate and show response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = answer_query_from_df(prompt, df)
        st.write(response)
    
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response})