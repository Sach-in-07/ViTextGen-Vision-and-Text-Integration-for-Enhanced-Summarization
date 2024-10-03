import streamlit as st
# from InstructorEmbedding import INSTRUCTOR

from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFaceHub
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# from langchain_openai import OpenAIEmbeddings,ChatOpenAI
import os
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_vectorstore_from_url(url):
    # get the text in document form
    loader = WebBaseLoader(url)
    document = loader.load()
    
    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    # create a vectorstore from the chunks
    vector_store = Chroma.from_documents(document_chunks, embeddings)

    return vector_store

def get_context_retriever_chain(vector_store):
    llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature":0.6, "max_length":512})
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain
    
def get_conversational_rag_chain(retriever_chain): 
    
    llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature":0.6, "max_length":512})
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response['answer']


# def get_response(user_input):
#     return "i don't know"

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response['answer']


st.set_page_config(page_title="Chat with websites",page_icon="aa")
st.title("chat with web")


        
chat_history = [
    AIMessage(content="Hello i m bot,how can i help you?")
]

with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("websites Url")

if website_url is None or website_url =="":
    st.info("Please enter a website link")
else:
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello i m bot,How can i help you")
    ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)

    # retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    
    # # vector_store = get_vectorestore_from_url(website_url)
    
    # # retriever_chain = get_context_retriever_chain(vector_store)
    
    # conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    # with st.sidebar:
    #     st.write(document_chunks)
    
    user_query = st.chat_input("Type your message here..")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        # response = conversation_rag_chain.invoke({
        #     "chat_history": st.session_state.chat_history,
        #     "input": user_query
        # })
        # st.write(response)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
        
        # retrieved_documents = retriever_chain.invoke({
        #    "chat_history": st.session_state.chat_history,
        #    "input": user_query
        # })
        # st.write(retrieved_documents)
        
    # with st.sidebar:
    #     st.write(st.session_state.chat_history)
        # with st.chat_message("Human"):
        #     st.write(user_query)

        # with st.chat_message("AI"):
        #     st.write(response)

    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        
            