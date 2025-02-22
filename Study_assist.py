#pip install streamlit - for code visualization
#pip install pymupdf - allows you to read, extract, modify and analyze pdf
#pip install python-docx - allows you to read, extract, modify and analyze microsoft word docx
#pip install openai

#Step 1
import streamlit as st
import docx
import openai
import os
import time
import random
import base64
from pypdf import PdfReader
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
#added to make the code work in streamlit
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

st.set_page_config(layout="wide")

#create two columns
col1, col2 = st.columns([1,4])

# Develop streamlit User Interface
with col2:
    st.title("📚 AI-Powered Study Assistant")
    

# Initialize session state variables are initialized

if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = []
if "correct_answers" not in st.session_state:
    st.session_state.correct_answers = {}
if "user_answers" not in st.session_state:
    st.session_state.user_answers = {}
if "submitted" not in st.session_state:
    st.session_state.submitted = False

# Function to extract text from PDF (cached to prevent reprocessing)
@st.cache_data
def extract_text_from_pdf(file):
    try:
        reader = PdfReader(file)
        return "".join([page.extract_text() or "" for page in reader.pages])
    except Exception as e:
        return f"Error extracting text: {e}"

# Function to extract text from DOCX (cached)
@st.cache_data
def extract_text_from_docx(file):
    try:
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    except Exception as e:
        return f"Error extracting text: {e}"
    
# Function to process uploaded file
def process_uploaded_file(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    elif uploaded_file.name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")
    return ""

# Function to get document prompt
def get_document_prompt(docs):
    prompt = "\n"
    for i, doc in enumerate(docs, start=1):
        prompt += f"\nContent {i}:\n{doc}\n\n"
    return prompt


# Sidebar (Collapsible Effect)
with st.sidebar:
    st.subheader("📂 Upload Your Notebook")
    uploaded_file = st.file_uploader("Upload a File (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

# File Upload Section
with col1:
    #with st.expander("Upload a File"):
        #uploaded_file = st.file_uploader(" Upload your Notebook (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
    if uploaded_file:
        file_name_without_ext = os.path.splitext(uploaded_file.name)[0]
        st.session_state.notebook_content = process_uploaded_file(uploaded_file)
        st.session_state.persist_directory = f"./chroma_db_{file_name_without_ext}"

        st.success("✅ Notebook uploaded successfully")


#Step 2
#Initialize Vector Database (cached to avoid reprocessing)
if "notebook_content" not in st.session_state:
    st.session_state.notebook_content = ""
if "persist_directory" not in st.session_state:
    st.session_state.persist_directory = ""
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "db" not in st.session_state:
    st.session_state.db = None

#Cached function to initialize Vector Database (Avoid reprocessing)
@st.cache_resource
def initialize_vector_db(notebook_content, persist_directory):
    # ✅ Ensure the directory exists
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory, exist_ok=True)

    # ✅ Load OpenAI API Key
        api_key = st.secrets["api_key"]

    # ✅ Define OpenAI Embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)

    # ✅ Split text into chunks
    text_splitter = CharacterTextSplitter(separator=" ", chunk_size=5000, chunk_overlap=100)
    docs = text_splitter.split_text(notebook_content)

    try:
        # ✅ Use absolute path for persistence
        db_path = os.path.abspath(persist_directory)

        # ✅ Initialize ChromaDB
        vector_store = Chroma(embedding_function=embeddings, persist_directory=db_path)

        # ✅ Add documents in batches (Avoids exceeding limits)
        batch_size = 50
        for i in range(0, len(docs), batch_size):
            batch = docs[i: i + batch_size]
            vector_store.add_texts(batch)

        return vector_store  # ✅ Return the vector store (Do NOT modify session state here)

    except Exception as e:
        raise RuntimeError(f"🔥 ChromaDB Error: {e}")

#Only update session state AFTER the function call
if st.session_state.notebook_content and st.session_state.vector_store is None:
    st.session_state.vector_store = initialize_vector_db(
        st.session_state.notebook_content,
        st.session_state.persist_directory
    )
    st.session_state.db = st.session_state.vector_store  # ✅ Store the DB reference

with col2: 
    # Step 3
    ####################################################################################################
    # Notebook Summary
    #####################################################################################################
    #Ensure database is loaded before proceeding
    if "db" in st.session_state and st.session_state.db is not None:
        if "summary" not in st.session_state:  #Run summary generation only once
            st.session_state.summary = None  # ✅ Initialize Summary State
            st.session_state.summary_header = "📝 Notebook Summary"  # ✅ Store Header in Session State
            progress_bar = st.progress(0)  #Initialize Progress Bar
            progress_text = st.empty()  #Placeholder for progress percentage updates

            try:
                # ✅ Step 1: Retrieve all documents from ChromaDB
                progress_text.write("🔍 Fetching relevant documents for summary... (0%)")
                all_docs = st.session_state.db.get(include=["documents"])["documents"]
                progress_bar.progress(10)

                # ✅ Step 2: Ensure at least the first 10 documents are selected
                num_available_docs = len(all_docs)
                num_samples = min(15, num_available_docs)  # Ensure at least 10 docs or all available
                selected_docs = all_docs[:num_samples] if num_available_docs > 0 else []
                progress_text.write(f"📖 Selected {num_samples} documents for summarization... (20%)")
                progress_bar.progress(20)

                # ✅ Step 3: Format the selected documents
                formatted_context = get_document_prompt(selected_docs) if selected_docs else "No content available."
                progress_text.write("🔄 Processing selected documents... (40%)")
                progress_bar.progress(40)

                # ✅ Step 4: Generate System Message for Summary
                system_message = (
                    f"Generate a summary of the following notebook content: "
                    f"\n\n###\n{formatted_context}\n###\n\n"
                    "The summary should contain the title of the book and a short sentence about the notebook"
                    "The summary should never be move that 8 sentences"
                    "Be precise, avoid opinions, and summarize the main points in a clear and structured way. "
                    "If the document has multiple sections, break it into meaningful segments."
                )
                progress_text.write("🧠 Preparing AI model for summarization... (60%)")
                progress_bar.progress(60)

                # ✅ Step 5: Load API Key
                api_key = st.secrets["api_key"]

                # ✅ Step 6: Call OpenAI API
                progress_text.write("🚀 Generating summary with AI... (80%)")
                progress_bar.progress(80)
                client = openai.OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": system_message}],
                    temperature=0.02
                )

                # ✅ Step 7: Store AI Summary in Session State (Prevents regeneration)
                st.session_state.summary = response.choices[0].message.content

                progress_text.write("✅ Finalizing results... (90%)")
                progress_bar.progress(90)
                progress_text.write("🎉 Summary generation complete! (100%)")
                progress_bar.progress(100)

            except Exception as e:
                st.error(f"🚨 An error occurred: {str(e)}")
                progress_bar.progress(0)  # Reset progress in case of failure

    # ✅ Display Summary if Available (WITHOUT DUPLICATE HEADER)
    if "summary" in st.session_state and st.session_state.summary:
        st.subheader(st.session_state.summary_header)  # ✅ Show header only once
        st.write(st.session_state.summary)


    # Step 4
    ######################################################################
    # Notebook Querying 
    ######################################################################

    st.subheader("Ask a Question from your Notebook")
    user_question = st.text_input("Enter your question:")
    if user_question:
        # ✅ Check if a document is uploaded before proceeding
        if "db" not in st.session_state or not st.session_state.db:
            st.warning("⚠️ Please upload a notebook before asking a question.")
        else:
            progress_bar = st.progress(0)  # ✅ Initialize Progress Bar
            progress_text = st.empty()  # ✅ Placeholder for status updates

            try:
                progress_text.write("🔍 Retrieving relevant documents... (20%)")
                progress_bar.progress(20)          
                retrieved_docs = st.session_state.db.similarity_search(user_question, k=10)
                formatted_context = get_document_prompt(retrieved_docs)
                progress_text.write("📖 Processing retrieved content... (50%)")
                progress_bar.progress(50)

                system_message = (
                    f"You are a professor teaching a course. Use the following notebook content "
                    f"to answer student questions accurately and concisely:\n\n{formatted_context}\n\n"
                    "Be precise and avoid opinions."
                    "Only state what is in the notebook content"
                    "Do not state what is not in the given notebook and be very precise and straight forward "
                )
                progress_text.write("🧠 Preparing AI model for response... (70%)")
                progress_bar.progress(70)

                #add the api_key
                api_key = st.secrets["api_key"]

                client = openai.OpenAI(api_key=api_key)

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": system_message}, 
                            {"role": "user", "content": user_question}],
                    temperature=0.01
                )
                progress_text.write("🚀 Generating AI response... (90%)")
                progress_bar.progress(90)

                # Store the question and response to avoid re-processing
                st.session_state.last_question = user_question
                st.session_state.last_response = response.choices[0].message.content

                progress_text.write("🎉 Query Completed! (100%)")
                progress_bar.progress(100)
                st.write(st.session_state.last_response)

            except Exception as e:
                st.error(f"⚠️ An error occurred: {str(e)}")
                progress_text.write("❌ Failed to retrieve response.")
                progress_bar.progress(0)  # Reset in case of failure

    #Step 5
    #############################################################################
    # Question Generation section
    #############################################################################
    st.subheader("📝 Generate Quiz Questions")

    # Ensure session state is initialized
    if "quiz_questions" not in st.session_state:
        st.session_state.quiz_questions = []
    if "correct_answers" not in st.session_state:
        st.session_state.correct_answers = {}
    if "user_answers" not in st.session_state:
        st.session_state.user_answers = {}
    if "submitted" not in st.session_state:
        st.session_state.submitted = False
    if "quiz_generated" not in st.session_state:
        st.session_state.quiz_generated = False

    # Allows user to select the number of questions (Minimum 5, Maximum 15)**
    num_questions = st.slider("Select the number of quiz questions:", min_value=1, max_value=50, value=5)

    if st.button("Generate Questions"):
        # Check if the document is uploaded before generating quiz questions
        if "db" not in st.session_state or not st.session_state.db:
            st.warning("⚠️ Please upload a notebook before generating quiz questions.")
        elif st.session_state.quiz_generated:
            st.info("✅ Quiz questions have already been generated! Scroll down to answer them.")
        else:
            progress_bar = st.progress(0)
            progress_text = st.empty()

            progress_text.write("🔍 Retrieving relevant content for quiz... (30%)")
            progress_bar.progress(30)

            # ✅ Ensure we check if `st.session_state.db` is initialized
            if st.session_state.db:
                all_docs = st.session_state.db.get(include=["documents"]).get("documents", [])
            else:
                all_docs = []

            if not all_docs:
                st.error("⚠️ No documents found in the database. Please upload a notebook first.")
            else:
                # Use at least 20 documents selection when available
                num_samples = min(20, len(all_docs)) if all_docs else 0
                selected_docs = random.sample(all_docs, num_samples) if num_samples > 0 else []
                formatted_context = get_document_prompt(selected_docs) if selected_docs else "No content available."

                progress_text.write("🧠 Preparing AI for quiz generation... (60%)")
                progress_bar.progress(60)


                system_message = (
                    f"Generate {num_questions} multiple-choice quiz questions from the following notebook content: "
                    f"\n\n###\n{formatted_context}\n###\n\n"
                    f"Each question should have 4 answer choices (A,B,C,D) and indicate the correct answer at the end:"
                    f"""the format of the reply should be
                Question 1: <question>
                A)  <answer choice A>
                B)  <answer choice B>
                C)  <answer choice C>
                D)  <answer choice D>
                Correct Answer: C

                Question 2: <question>
                ...""")
                
                #add the api_key
                # ✅ Ensure OpenAI API key is loaded only once
                if "api_key" not in st.session_state:
                    api_key = st.secrets["api_key"]

                # ✅ Initialize OpenAI client only once in session state
                if "client" not in st.session_state:
                    st.session_state.client = openai.OpenAI(api_key=api_key)

                response = st.session_state.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": system_message}],
                    temperature=0.02
                )

                progress_text.write("📖 Formatting quiz questions... (90%)")
                progress_bar.progress(90)

                raw_questions = response.choices[0].message.content.strip().split("\n\n")
                quiz_questions, correct_answers = [], {}

                for q in raw_questions:
                    parts = q.split("\n")
                    if len(parts) < 6:
                        continue
                    question_text, choices, correct_choice = parts[0], parts[1:5], parts[5].split(":")[-1].strip()
                    correct_answers[question_text] = correct_choice
                    quiz_questions.append((question_text, choices))

                # Store questions in session state to avoid reloading
                st.session_state.quiz_questions = quiz_questions
                st.session_state.correct_answers = correct_answers
                st.session_state.user_answers = {q[0]: "No Answer" for q in quiz_questions}
                st.session_state.submitted = False
                st.session_state.quiz_generated = True

                progress_text.write("🎉 Quiz Generation Complete! (100%)")
                progress_bar.progress(100)
                st.success(f"✅ {num_questions} Quiz Questions Generated! Select your answers below and click submit.")


    #Display quiz questions if available
    if st.session_state.quiz_questions:
        st.subheader("Answer the Quiz Questions")
        #Using st.form to delay showing results until after submission
        with st.form("quiz_form"):
            for idx, (question, choices) in enumerate(st.session_state.quiz_questions):
                st.write(f"**{question}**")
                selected_answer = st.radio(
                    f"Select your answer for Question {idx + 1}:",
                    choices,
                    index = None,
                    key=f"q{idx}"
                )
                st.session_state.user_answers[question] = selected_answer if selected_answer else "No Answer"

            submitted = st.form_submit_button("Submit Answers")
        if submitted:
            st.session_state.submitted = True  #Ensures submission tracking



    #Displaying results AFTER submission
    if st.session_state.submitted:
        st.subheader("Quiz Results")
        correct_count = 0
        unanswered_count = 0  #Initialize unanswered count

        for question, selected_answer in st.session_state.user_answers.items():
            correct_answer = st.session_state.correct_answers.get(question, "Unknown")

            if selected_answer == "No Answer":
                unanswered_count += 1
                st.warning(f"{question}\n**You did not answer this question.**")
            elif selected_answer.startswith(correct_answer):
                st.success(f"{question}\n**Correct!** The answer is {correct_answer}")
                correct_count += 1
            else:
                st.error(f"{question}\n**Incorrect!** The correct answer is {correct_answer}")

        st.info(f"🏆 You got {correct_count}/{len(st.session_state.quiz_questions)} correct!")

        #Warn user if they left any questions unanswered
        if unanswered_count > 0:
            st.warning(f"You left {unanswered_count} question(s) unanswered!")

    st.divider()  # Separate sections visually



    #Save user scores & progress? 
    #Provide explanations for correct/incorrect answers? 
    #Add a leaderboard for top scores? 