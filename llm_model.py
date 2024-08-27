from langchain.vectorstores import FAISS
#from langchain.llms import GooglePalm, CTransformers
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

vector_index_path = "assets/vectordb/faiss_index"

class LlmModel:
    
    def __init__(self):
        # load dot env variables   
        self.load_env_variables()   
        # load llm model
        self.hf_embeddings = self.load_huggingface_embeddings()

    def load_env_variables(self):
        load_dotenv()  # take environment variables from .env
    
    def custom_prompt(self, question, history, context):
        #RAG prompt template
        prompt = "<s>"
        for user_prompt, bot_response in history: # provide chat history
            prompt += f"[INST] {user_prompt} [/INST]"
            prompt += f" {bot_response}</s>"
            
        message_prompt = f"""
        You are a question answer agent and you must strictly follow below prompt template.
        Given the following context and a question, generate an answer based on this context only.
        Keep answers brief and well-structured. Do not give one word answers.
        If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

        CONTEXT: {context}

        QUESTION: {question}
        """
        prompt += f"[INST] {message_prompt} [/INST]"
        # prompt.image("./assets/BCSkills_Logo.png", width=30)
            
        return prompt

    def format_sources(self, sources):
        # format the document sources
        source_results = []
        for source in sources:
            source_results.append(str(source.page_content) + 
                                  "\n Document: " + str(source.metadata['source']) + 
                                  " Page: " + str(source.metadata['page']))            
        return source_results
        
    def mixtral_chat_inference(self, prompt, history, temperature, top_p, repetition_penalty, retriever):
        
        context = retriever.get_relevant_documents(prompt)
        sources = self.format_sources(context)
        # use hugging face infrence api
        client = InferenceClient("mistralai/Mistral-7B-Instruct-v0.1",
                                    token=os.environ["HF_TOKEN"]
                                )
        temperature = float(temperature)
        if temperature < 1e-2:
            temperature = 1e-2
            
        generate_kwargs = dict(
                            temperature = temperature,
                            max_new_tokens = 512,
                            top_p = top_p,
                            repetition_penalty = repetition_penalty,
                            do_sample = True
                            )
        
        formatted_prompt = self.custom_prompt(prompt, history, context)
        
        return client.text_generation(formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False), sources

    

    def load_huggingface_embeddings(self):
        # Initialize instructor embeddings using the Hugging Face model
        #return HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
        return HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2", 
                                     model_kwargs={'device': 'cpu'})        
        
        
 
    def create_vector_db(self, filename):

        if filename.endswith(".pdf"):    
            loader = PyPDFLoader(file_path=filename)
        elif filename.endswith(".doc") or filename.endswith(".docx"):
            loader = Docx2txtLoader(filename)
        elif filename.endswith("txt") or filename.endswith("TXT"):
            loader = TextLoader(filename)

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(loader.load())

        # Create a FAISS instance for vector database from 'data'
        vectordb = FAISS.from_documents(documents = splits,
                                        embedding = self.hf_embeddings)

        # Save vector database locally
        #vectordb.save_local(vector_index_path)
        
        # set vectordb content
        # Load the vector database from the local folder
        #vectordb = FAISS.load_local(vector_index_path, self.hf_embeddings)
        # Create a retriever for querying the vector database
        return vectordb.as_retriever(search_type="similarity")