import streamlit as st
# from InstructorEmbedding import INSTRUCTOR

from langchain.schema import Document


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

import hashlib, io, requests, pandas as pd
from selenium import webdriver
from selenium.webdriver import ChromeOptions
from bs4 import BeautifulSoup
from pathlib import Path
from PIL import Image

import os
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import numpy as np

def get_content_from_url(url):
    options = ChromeOptions()
    options.add_argument("--headless=new")
    driver = webdriver.Chrome(options=options)

    driver.get(url)
    page_content = driver.page_source
    driver.quit()
    return page_content

def parse_image_urls(content):
    soup = BeautifulSoup(content, "html.parser")
    results = []
    for img in soup.findAll("img"):
        img_src = img.get("src")
        if img_src and img_src not in results:
            results.append(img_src)
    return results


def save_urls_to_csv(image_urls):
    df = pd.DataFrame({"links": image_urls})
    df.to_csv("links.csv", index=False, encoding="utf-8")

# def get_and_save_image_to_file(image_url, output_dir):
#     image_content = requests.get(image_url).content
#     image_file = io.BytesIO(image_content)
#     image = Image.open(image_file).convert("RGB")
#     filename = hashlib.sha1(image_content).hexdigest()[:10] + ".png"
#     file_path = output_dir / filename
#     image.save(file_path, "PNG", quality=80)
from urllib.parse import urljoin

def get_and_save_image_to_file(image_url, output_dir, base_url=None):
    # Check if the image URL is missing the scheme (http/https)
    if not image_url.startswith(('http://', 'https://')):
        if base_url:
            image_url = urljoin(base_url, image_url)  # Combine base URL with relative URL
        else:
            # If no base URL is provided, assume https:// as default
            image_url = 'https://' + image_url.lstrip('/')
    
    try:
        image_content = requests.get(image_url).content
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file).convert("RGB")
        
        # Create a unique filename using a hash of the image content
        filename = hashlib.sha1(image_content).hexdigest()[:10] + ".png"
        file_path = output_dir / filename
        
        # Save the image to the specified directory
        image.save(file_path, "PNG", quality=80)
        return file_path  # Return the path to the saved file
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {image_url}: {e}")
        return None


def get_data(url):
    # url = "https://www.bbc.com/"
    content = get_content_from_url(url)
    image_urls = parse_image_urls(content)
    # print(image_urls)
    save_urls_to_csv(image_urls)

    for image_url in image_urls:
        get_and_save_image_to_file(
            image_url, output_dir=Path("CAP")
        )

# python -m venv env2
# .\env2\Scripts\activate


model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

# def predict_step(image_paths):
#     images = []
#     for image_path in image_paths:
#         i_image = Image.open(image_path)
#         if i_image.mode != "RGB":
#             i_image = i_image.convert(mode="RGB")
        
#         i_image = i_image.resize((224, 224))  # Resize image to model's expected input size
#         i_image = np.array(i_image).astype(np.float32) / 255.0  # Normalize image to [0, 1] range
#         images.append(i_image)

#     images = np.array(images)  # Convert list of images to a numpy array
#     images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)  # Convert to tensor with shape (batch_size, channels, height, width)

#     pixel_values = feature_extractor(images=images, return_tensors="pt", do_rescale=False).pixel_values
#     pixel_values = pixel_values.to(device)

#     output_ids = model.generate(pixel_values, **gen_kwargs)

#     preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
#     preds = [pred.strip() for pred in preds]
#     return preds
# # Example usage
# predictions = predict_step(['2.jpg'])
# print(predictions)
def predict_step(folder_path):
    images = []
    image_paths = []

    # Load all images from the folder
    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".jpeg", ".png")):  # Add more image extensions if needed
            image_path = os.path.join(folder_path, filename)
            image_paths.append(image_path)
            i_image = Image.open(image_path)
            if i_image.mode != "RGB":
                i_image = i_image.convert(mode="RGB")

            i_image = i_image.resize((224, 224))  # Resize image to model's expected input size
            i_image = np.array(i_image).astype(np.float32) / 255.0  # Normalize image to [0, 1] range
            images.append(i_image)

    images = np.array(images)  # Convert list of images to a numpy array
    images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)  # Convert to tensor with shape (batch_size, channels, height, width)

    pixel_values = feature_extractor(images=images, return_tensors="pt", do_rescale=False).pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    preds = [pred.strip() for pred in preds]

    # Combine all predictions into a single string with spaces between captions
    combined_caption = " ".join(preds)
    return combined_caption

# Example usage
# folder_path = "CAP"

# print(combined_captions)





def get_vectorstore_from_url(url):
    # get the text in document form
    loader = WebBaseLoader(url)
    document = loader.load()
    get_data(website_url)
    combined_captions = predict_step("CAP")
    caption_documents = [Document(page_content=caption) for caption in combined_captions]
    
    # Combine original documents and caption documents
    document.extend(caption_documents)  # Add the captions to the document list
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
        
            