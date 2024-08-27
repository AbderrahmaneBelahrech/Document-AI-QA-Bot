import streamlit as st
from streamlit_lottie import st_lottie
import fitz  # PyMuPDF
import requests
import os, shutil
import llm_model


SYSTEM_PROMPT = [
    """
    You are not Mistral AI, but rather a Q&A bot trained by Krishna Kumar while building a cool side project based on RAG. Whenever asked, you need to answer as Q&A bot.
    """,
    """You are a RAG based Document Q&A bot. Based on the input prompt and retrieved context from the vector database you will answer questions that are closer to the context. 
    If no context was found then, say "I don't know" instead of making up answer on your own. Follow above rules strictly.
    """
]


@st.cache_data(experimental_allow_widgets=True)
def index_document(_llm_object, uploaded_file):

    if uploaded_file is not None:
        # Specify the folder path where you want to store the uploaded file in the 'assets' folder
        assets_folder = "assets/uploaded_files"
        if not os.path.exists(assets_folder):
            os.makedirs(assets_folder)

        # Save the uploaded file to the specified folder
        file_path = os.path.join(assets_folder, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        file_name = os.path.join(assets_folder, uploaded_file.name)
        st.success(f"File '{file_name}' uploaded !")

        with st.spinner("Indexing document... This is a free CPU version and may take a while ⏳"):
            retriever = _llm_object.create_vector_db(file_name)
                        
        return file_name, retriever
    else:
        return None, None


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


def is_query_valid(query: str) -> bool:
    if not query:
        st.error("Please enter a question!")
        return False
    return True

def init_state() :
    if "filename" not in st.session_state:
        st.session_state.filename = None
        
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "temp" not in st.session_state:
        st.session_state.temp = 0.7

    if "history" not in st.session_state:
        st.session_state.history = [SYSTEM_PROMPT]

    if "repetion_penalty" not in st.session_state:
        st.session_state.repetion_penalty = 1

    if "chat_bot" not in st.session_state:
        st.session_state.chat_bot = "Mixtral-8x7B-Instruct-v0.1"
       
        

def faq():
    st.markdown(
        """
       
        """
    )

def sidebar():
    with st.sidebar:
        # Fixed model parameters
        st.session_state.temperature = 0.7
        st.session_state.top_p = 0.95
        st.session_state.repetition_penalty = 1.0
        
        st.session_state.llm_object = load_model()
        st.markdown("---")
        
        st.session_state.uploaded_file = st.file_uploader("Upload a file", type=["pdf", "doc", "docx", "txt"])
        _, retriever = index_document(st.session_state.llm_object, st.session_state.uploaded_file)
        faq()
        return retriever

    with st.sidebar:
     
        st.session_state.model_name = 'Mixtral-8x7B-Instruct-v0.1'
        # set model temperature
        st.session_state.temperature = st.slider(label="Temperature", min_value=0.0, max_value=1.0, step=0.1, value=0.7)
        st.session_state.top_p = st.slider(label="Top Probablity", min_value=0.0, max_value=1.0, step=0.1, value=0.95)
        st.session_state.repetition_penalty = st.slider(label="Repetition Penalty", min_value=0.0, max_value=1.0, step=0.1, value=1.0)
        
        # load model parameters
        st.session_state.llm_object = load_model()
        st.markdown("---")
        # Upload file through Streamlit
        st.session_state.uploaded_file = st.file_uploader("Upload a file", type=["pdf", "doc", "docx", "txt"])
        _, retriever = index_document(st.session_state.llm_object, st.session_state.uploaded_file)
    
        faq()
        return retriever


def chat_box() :
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            

def generate_chat_stream(prompt, retriever) :
    
    with st.spinner("Fetching relevant answers from source document..."):
        response, sources = st.session_state.llm_object.mixtral_chat_inference(prompt, st.session_state.history, st.session_state.temperature, 
                                                                               st.session_state.top_p, st.session_state.repetition_penalty, retriever)
    
        
    return response, sources

def stream_handler(chat_stream, placeholder) :
    full_response = ''

    for chunk in chat_stream :
        if chunk.token.text!='</s>' :
            full_response += chunk.token.text
            placeholder.markdown(full_response + "▌")
    placeholder.markdown(full_response)

    return full_response

def show_source(sources) :
    with st.expander("Show source") :
        for source in sources:
            st.info(f"{source}")
            

# Function to load model parameters
@st.cache_resource()
def load_model():
    # create llm object
    return llm_model.LlmModel()

st.set_page_config(page_title="Document QA Bot")
# lottie_book = load_lottieurl("https://assets4.lottiefiles.com/temp/lf20_aKAfIn.json")
# st_lottie(lottie_book, speed=1, height=200, key="initial")
# Place the title below the Lottie animation
st.title("Document Q&A Bot 🤖")
# st.image("assets/BCSkills_Logo.png", width=100)  # Adjust the path and size as needed

# initialize session state for streamlit app
init_state()
# Left Sidebar
retriever = sidebar()
chat_box()

if prompt := st.chat_input("Ask a question about your document!"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        chat_stream, sources = generate_chat_stream(prompt, retriever)
  
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = stream_handler(chat_stream, placeholder)
            show_source(sources)

        st.session_state.history.append([prompt, full_response])
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    except Exception as e:
        if not st.session_state.uploaded_file:
            st.error("Kindly provide the document file by uploading it before posing any questions. Your cooperation is appreciated!")
        else:
            st.error(e)
    

#     import streamlit as st
# from streamlit_lottie import st_lottie
# import fitz  # PyMuPDF
# import requests
# import os, shutil
# import llm_model


# SYSTEM_PROMPT = [
#     """
#     You are not Mistral AI, but rather a Q&A bot trained by Krishna Kumar while building a cool side project based on RAG. Whenever asked, you need to answer as Q&A bot.
#     """,
#     """You are a RAG based Document Q&A bot. Based on the input prompt and retrieved context from the vector database you will answer questions that are closer to the context. 
#     If no context was found then, say "I don't know" instead of making up answer on your own. Follow above rules strictly.
#     """
# ]

# @st.cache_data(experimental_allow_widgets=True)
# def index_document(_llm_object, uploaded_file):
#     if uploaded_file is not None:
#         assets_folder = "assets/uploaded_files"
#         if not os.path.exists(assets_folder):
#             os.makedirs(assets_folder)

#         file_path = os.path.join(assets_folder, uploaded_file.name)
#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.getvalue())

#         st.success(f"File '{file_path}' uploaded!")

#         with st.spinner("Indexing document... This is a free CPU version and may take a while ⏳"):
#             retriever = _llm_object.create_vector_db(file_path)
                        
#         return file_path, retriever
#     else:
#         return None, None

# def load_lottieurl(url: str):
#     r = requests.get(url)
#     if r.status_code != 200:
#         return None
#     return r.json()

# def is_query_valid(query: str) -> bool:
#     if not query:
#         st.error("Please enter a question!")
#         return False
#     return True

# def init_state():
#     if "filename" not in st.session_state:
#         st.session_state.filename = None
        
#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#     if "temp" not in st.session_state:
#         st.session_state.temp = 0.7

#     if "history" not in st.session_state:
#         st.session_state.history = [SYSTEM_PROMPT]

#     if "repetion_penalty" not in st.session_state:
#         st.session_state.repetion_penalty = 1

#     if "chat_bot" not in st.session_state:
#         st.session_state.chat_bot = "Mixtral-8x7B-Instruct-v0.1"

# def faq():
#     st.markdown(
#         """
#         FAQ section goes here.
#         """
#     )

# def sidebar():
#     with st.sidebar:
#         st.image("assets/BCSkills_Logo.png", width=150)  # Adjust the path and size as needed

#         st.session_state.temperature = st.slider(label="Temperature", min_value=0.0, max_value=1.0, step=0.1, value=0.7)
#         st.session_state.top_p = st.slider(label="Top Probability", min_value=0.0, max_value=1.0, step=0.1, value=0.95)
#         st.session_state.repetition_penalty = st.slider(label="Repetition Penalty", min_value=0.0, max_value=1.0, step=0.1, value=1.0)
        
#         st.session_state.llm_object = load_model()
#         st.markdown("---")
        
#         st.session_state.uploaded_file = st.file_uploader("Upload a file", type=["pdf", "doc", "docx", "txt"])
#         _, retriever = index_document(st.session_state.llm_object, st.session_state.uploaded_file)
#         faq()
#         return retriever

# def chat_box():
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

# def generate_chat_stream(prompt, retriever):
#     with st.spinner("Fetching relevant answers from source document..."):
#         response, sources = st.session_state.llm_object.mixtral_chat_inference(
#             prompt, st.session_state.history, st.session_state.temperature, 
#             st.session_state.top_p, st.session_state.repetition_penalty, retriever
#         )
#     return response, sources

# def stream_handler(chat_stream, placeholder):
#     full_response = ''

#     for chunk in chat_stream:
#         if chunk.token.text != '</s>':
#             full_response += chunk.token.text
#             placeholder.markdown(full_response + "▌")
#     placeholder.markdown(full_response)

#     return full_response

# def show_source(sources):
#     with st.expander("Show source"):
#         for source in sources:
#             st.info(f"{source}")

# @st.cache_resource()
# def load_model():
#     return llm_model.LlmModel()

# st.set_page_config(page_title="Document QA Bot")
# st.title("Document Q&A Bot 🤖")

# init_state()
# retriever = sidebar()
# chat_box()

# if prompt := st.chat_input("Ask a question about your document!"):
#     st.chat_message("user").markdown(prompt)
#     st.session_state.messages.append({"role": "user", "content": prompt})

#     try:
#         chat_stream, sources = generate_chat_stream(prompt, retriever)

#         with st.chat_message("assistant"):
#             placeholder = st.empty()
#             full_response = stream_handler(chat_stream, placeholder)
#             show_source(sources)

#         st.session_state.history.append([prompt, full_response])
#         st.session_state.messages.append({"role": "assistant", "content": full_response})
#     except Exception as e:
#         if not st.session_state.uploaded_file:
#             st.error("Kindly provide the document file by uploading it before posing any questions. Your cooperation is appreciated!")
#         else:
#             st.error(e)
