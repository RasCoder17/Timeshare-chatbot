from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import openai
import streamlit as st

openai.api_key = "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"
model = SentenceTransformer('all-MiniLM-L6-v2')

pc = Pinecone(api_key="8eeccd25-faab-4b54-8553-a17cbe358ece")

index_name = "langchain-chatbot"
index = pc.Index('langchain-chatbot')

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(
            cloud="gcp",
            region="us-central1"
        )
    )

def find_match(input_text):
    input_embedding = model.encode(input_text).tolist()
    result = index.query(vector=input_embedding, top_k=1, includeMetadata=True)
    
    metadata_1 = result['matches'][0]['metadata']
    # metadata_2 = result['matches'][1]['metadata']

    if isinstance(metadata_1, dict):
        metadata_1 = str(metadata_1)

    # if isinstance(metadata_2, dict):
    #     metadata_2 = str(metadata_2)

    return metadata_1


def query_refiner(conversation, query):
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response['choices'][0]['text']

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses']) - 1):
        conversation_string += f"Human: {st.session_state['requests'][i]}\n"
        conversation_string += f"Bot: {st.session_state['responses'][i + 1]}\n"
    return conversation_string
