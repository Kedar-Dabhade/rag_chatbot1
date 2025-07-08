import os
from flask import Flask, request, jsonify, Response, stream_with_context, session
from dotenv import load_dotenv
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory, FileChatMessageHistory
from langchain.prompts import PromptTemplate
from flask_cors import CORS
from langchain.callbacks.base import BaseCallbackHandler
import queue
import threading
import tiktoken
import uuid
from langchain.globals import set_debug
import json

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
opensearch_host = os.getenv("OPENSEARCH_HOST", "http://localhost:9200")
opensearch_user = os.getenv("OPENSEARCH_USER", "admin")
opensearch_password = os.getenv("OPENSEARCH_PASSWORD", "n864pyAkop5E5WB")
opensearch_index = os.getenv("OPENSEARCH_INDEX", "opensearch_fert")
opensearch_verify_certs = os.getenv("OPENSEARCH_VERIFY_CERTS", "false").lower() == "true"

# Set up embeddings and vector store
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = OpenSearchVectorSearch(
    index_name=opensearch_index,
    opensearch_url=opensearch_host,
    embedding_function=embeddings,
    http_auth=(opensearch_user, opensearch_password),
    verify_certs=opensearch_verify_certs,
    engine="faiss",
    space_type="cosinesimil"
)

# Configurable parameters for retrieval
MAX_CONTEXT_DOCS = 7  # Default 3

# Set up retriever with k using MMR (returns top k after MMR), fetch_k controls candidate pool size
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': MAX_CONTEXT_DOCS, 'fetch_k': 40})

# Set up LLM
llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4o-mini")

# Custom prompt for the agent
custom_prompt = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template="""
You are an expert agriculture consultant. Your task is to recommend appropriate fertiliser products and answer user queries.

IMPORTANT: Do not say \"Based on the provided information\". Start your answer directly with the advice.
Do NOT ask to consult with a local agronomist.
If the answer has more than 3-4 points summarise each of them or try fitting the answer in 2-3 points

- Identify whether the user's input is a query related to fertiliser or a general input(greetings,etc).
- If the user's input is a general input, answer it in a friendly and engaging manner.
- Even if the user's input is a general input, make sure to answer it in a way that is relevant to the fertiliser context in a respectful way.Do not pinpoint to user that it is a general input.
- Make sure to answer the question in the same language as the user's input.
- Use ONLY the following context and the chat history to answer the user's question.
- Never refer to the context as \"the context\" or \"the information\" or \"the data\"
- Do not recommend products that are not in the context.
- By default, provide concise, clear answers using bullet points or numbered lists for readability. If the retrieved context is big, provide a summary of the context without missing any important information and then answer the question.
- Only provide detailed explanations if the user specifically asks for more detail (e.g., 'explain in detail', 'more info', etc).
- Use line breaks and formatting to make your answer easy to read.
- If the answer is long, summarise and present it meaningfully within 5 lines, unless the user requests more detail.
- If asked about a specific product,summarise the context recieved and present it in a concise shorter format which looks good and readable.


Context:
{context}

Chat history:
{chat_history}

Question: {question}
Answer:
"""
)

class QueueStreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.q = queue.Queue()
    def on_llm_new_token(self, token, **kwargs):
        self.q.put(token)
    def get_stream(self):
        while True:
            token = self.q.get()
            if token is None:
                break
            yield token

app = Flask(__name__)
app.secret_key = "dev-secret-key-please-change"  # Set a strong, unique key in production
CORS(app, supports_credentials=True)

@app.before_request
def ensure_session():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())

# Directory to store per-session chat histories
CHAT_HISTORY_DIR = 'sessions'
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

def get_session_history():
    session_id = session['session_id']
    history_path = os.path.join(CHAT_HISTORY_DIR, f"{session_id}.json")
    return FileChatMessageHistory(history_path)

@app.route('/session-chat', methods=['POST'])
def session_chat():
    data = request.get_json()
    user_message = data.get('message', '')
    if not user_message:
        return jsonify({"error": "No message provided."}), 400
    history = get_session_history()
    # Create a ConversationSummaryMemory for this session
    session_memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        max_token_limit=1250,
        chat_memory=history
    )
    # Create a ConversationalRetrievalChain for this session
    session_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=session_memory,
        combine_docs_chain_kwargs={"prompt": custom_prompt},
        return_source_documents=True
    )
    result = session_chain.invoke({"question": user_message})
    bot_response = result["answer"]
    # Store user and bot messages in the session file
    history.add_user_message(user_message)
    history.add_ai_message(bot_response)
    return jsonify({"response": bot_response})

@app.route('/history', methods=['GET'])
def session_history():
    history = get_session_history()
    # Return as list of dicts: [{role: 'user'/'ai', content: ...}]
    messages = [
        {"role": m.type, "content": m.content} for m in history.messages
    ]
    return jsonify(messages)

@app.route('/session-chat-stream', methods=['POST'])
def session_chat_stream():
    data = request.get_json()
    user_message = data.get('message', '')
    if not user_message:
        return jsonify({"error": "No message provided."}), 400

    history = get_session_history()
    # Create a ConversationSummaryMemory for this session
    session_memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        max_token_limit=1250,
        chat_memory=history
    )
    # Streaming LLM
    handler = QueueStreamHandler()
    streaming_llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4o-mini", streaming=True, callbacks=[handler])
    session_chain = ConversationalRetrievalChain.from_llm(
        llm=streaming_llm,
        retriever=retriever,
        memory=session_memory,
        combine_docs_chain_kwargs={"prompt": custom_prompt},
        return_source_documents=True
    )

    def generate():
        bot_message = ""
        def run_chain():
            session_chain.invoke({"question": user_message})
            handler.q.put(None)  # Signal end of stream
        thread = threading.Thread(target=run_chain)
        thread.start()
        for token in handler.get_stream():
            bot_message += token
            yield token
        thread.join()
        # After streaming, store user and bot messages
        history.add_user_message(user_message)
        history.add_ai_message(bot_message)

    return Response(stream_with_context(generate()), mimetype='text/plain')

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    user_message = data.get('user_message')
    bot_message = data.get('bot_message')
    feedback_type = data.get('feedback')  # 'like' or 'dislike'
    session_id = session.get('session_id')
    if not all([session_id, user_message, bot_message, feedback_type]):
        return jsonify({'error': 'Missing required fields.'}), 400
    # Directory structure
    base_dir = os.path.join('feedback', 'liked_responses' if feedback_type == 'like' else 'disliked_responses')
    os.makedirs(base_dir, exist_ok=True)
    session_file = os.path.join(base_dir, f"{session_id}.json")
    # Prepare entry
    entry = {"user_message": user_message, "bot_message": bot_message}
    # Append to file
    if os.path.exists(session_file):
        with open(session_file, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
    else:
        data_list = []
    data_list.append(entry)
    with open(session_file, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=2)
    return jsonify({'status': 'success'})

set_debug(True)

if __name__ == "__main__":
    app.run(debug=True) 