# Test guideline and grading function
import streamlit as st
import openai 
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings # OpenAI Embeddings for text
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from html_template import css, bot_template, user_template, overall_template
import base64 # Encode/decode images
import torch
from PIL import Image # Image processing
import io
import re # Regular expression for text processing
import pdfplumber # Extract text and tables from PDF
import fitz # PyMuPDF for image extraction
import os
import tempfile # Temporary file storage

# Check for GPU availability and set the appropriate device for computation.
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Function to handle topic input and store it in session state
def handle_topic_input(topic_input_text):
    # Store the topic content in the session state
    topic_content = topic_input_text.strip() # Remove extra whitespace
    st.session_state.topic_text = topic_content  # Save topic to session state
    return topic_content 

def get_writing_text(direct_text):
    return direct_text.strip()

def get_pdf_text(pdf_docs, task_type="Task 2"):
    pdf_text = ""
    pdf_images = []
    pdf_tables = []

    for pdf in pdf_docs:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf.read())
            tmp_path = tmp.name

        # Extract text + table by pdfplumber
        try:
            with pdfplumber.open(tmp_path) as plumber_pdf:
                for page in plumber_pdf.pages:
                    text = page.extract_text()
                    if text:
                        pdf_text += text + "\n"

                    if task_type == "Task 1":
                        table = page.extract_table()
                        if table:
                            table_html = "<table>\n"
                            for row in table:
                                table_html += "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>\n"
                            table_html += "</table>"
                            pdf_tables.append(table_html)
        except Exception as e:
            st.warning(f"PDF text/table extraction failed: {str(e)}")
        
        # If Task 1, extract images by fitz
        if task_type == "Task 1":
            try:
                with fitz.open(tmp_path) as doc:
                    for i in range(len(doc)):
                        for img_index, img in enumerate(doc.get_page_images(i)):
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                            pdf_images.append({
                                "page": i + 1,
                                "index": img_index,
                                "image_payload": image_base64
                            })
            except Exception as e:
                st.warning(f"PDF image extraction failed: {str(e)}")
        os.remove(tmp_path)

    # If no text at all, we still want to allow summarizing from images
    return pdf_text.strip(), pdf_images, pdf_tables

def summarize_tables(tables_html):
    prompt_text = """
    You are an assistant tasked with summarizing tables.
    Give a concise summary of the table.

    Respond only with the summary, no additional comment.
    Do not start your message by saying \"Here is a summary\" or anything like that.
    Just give the summary as it is.

    Table: {element}
    """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = ChatOpenAI(model_name="gpt-4o-mini")
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
    return summarize_chain.batch(tables_html, {"max_concurrency": 3})

def summarize_images(pdf_images):
    summaries = []
    for idx, img in enumerate(pdf_images):
        b64_img = img["image_payload"]
        
        try:
            # Gọi GPT-4o với Vision input
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "You are an IELTS writing expert. This image is part of an IELTS Task 1 question, which asks candidates to describe a process. Please describe what the image shows, focusing on the main stages of the process in 1–2 concise sentences.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64_img}"},
                            },
                        ],
                    }
                ],
                max_tokens=300
            )
            summary = response.choices[0].message.content.strip()
            summaries.append(f"Image {idx+1}: {summary}")
        except Exception as e:
            summaries.append(f"Image {idx+1}: Failed to summarize. Error: {str(e)}")
    return summaries

def get_combined_text_chunks(topic_content, writing_input_text):
    
    combined_content = topic_content + "\n" + writing_input_text
    
    # Ensure there is valid input
    if not combined_content:
        st.error("No valid content found in topic or essay.")
        return []

    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(combined_content)
    return chunks

def get_vector_store(text_chunks):
    """Embed text chunks using OpenAI and store in FAISS vectorstore"""
    if not text_chunks:
        return None # Return None if no text chunks

    # Use OpenAI to generate embeddings for FAISS
    embeddings = OpenAIEmbeddings()

    # Create FAISS vector store
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    return vectorstore

def get_conversation_chain(vectorstore):
    """Create retrieval-based chat chain with memory"""
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    if not vectorstore:
        return None

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vectorstore.as_retriever(),
        memory = memory
    )
    return conversation_chain

# Function to generate guideline based on topic
def get_guideline_prompt(topic_text, image_summaries, table_summaries):
    if not topic_text:
        return None

    # Giới hạn độ dài topic để tránh lỗi token
    MAX_TOPIC_CHARS = 5000
    short_topic_text = topic_text[:MAX_TOPIC_CHARS]

    # Prompt structure
    prompt = "You are an IELTS expert. Provide a detailed guideline on how to approach the following writing task. Each part has no more than 3 sentences.\n\n"
    prompt += "The format should be:\n"
    prompt += "Introduction and Overview: ...\n"
    prompt += "Body 1: ...\n"
    prompt += "Body 2: ...\n"
    prompt += "Conclusion: ...\n"
    prompt += "Common mistakes to avoid: ...\n\n"

    prompt += f"Topic:\n{short_topic_text}\n\n"
    
    # Tóm tắt bảng
    if table_summaries:
        prompt += "Here are summaries of tables included in the task:\n"
        for summary in table_summaries:
            prompt += f"- {summary.strip()}\n"

    # Tóm tắt ảnh biểu đồ
    if image_summaries:
        prompt += "\nHere are summaries of charts or visual elements:\n"
        for summary in image_summaries:
            prompt += f"- {summary.strip()}\n"

    return prompt.strip()

# Function to generate a grading prompt combining topic and writing content
def get_grading_prompt(full_input_text):
    if not full_input_text:
        return None
    
    # Grading prompt structure
    prompt = (
        "You are an IELTS examiner.\n"
        "Please grade the following essay based on the four IELTS writing criteria.\n"
        "For each criterion, provide the score (from 0 to 9) followed by specific feedback. Feedback for each criteria should not exceed 2 sentences.\n"
        "Use the following format:\n\n"
        "Task Achievement: [score]\n"
        "[Feedback for Task Achievement]\n\n"
        "Coherence and Cohesion: [score]\n"
        "[Feedback for Coherence and Cohesion]\n\n"
        "Lexical Resource: [score]\n"
        "[Feedback for Lexical Resource]\n\n"
        "Grammatical Range and Accuracy: [score]\n"
        "[Feedback for Grammatical Range and Accuracy]\n\n"
        "Overall Score: [score]\n\n"
        f"Essay:\n{full_input_text}\n"
    )
    return prompt

def extract_section(label, text):
    if label == "Overall Score":
        pattern = rf"{label}:\s*(\d(?:\.\d)?)"
        match = re.search(pattern, text)
        if match:
            score = match.group(1).strip()
            return f"<strong>{label}: {score}</strong>"
        else:
            return f"<strong>{label}: Not found.</strong>"
    else:
        pattern = rf"{label}:\s*(\d(?:\.\d)?)\s*\n(.*?)(?=\n(?:Task Achievement|Coherence and Cohesion|Lexical Resource|Grammatical Range and Accuracy|Overall Score)|\Z)"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            score = match.group(1).strip()
            feedback = match.group(2).strip()
            return f"<strong>{label}: {score}</strong>\n{feedback}"
        else:
            return f"<strong>{label}: Not found.</strong>"

def handle_band_input(target_band):
    # Ensure band_input is within the allowed range
    if 6.0 <= target_band <= 9.0:
        st.session_state.target_band = target_band
        return target_band
    else:
        st.error("Please select a valid band between 6.0 and 9.0.")
        return None

def get_corrected_writing_prompt(writing_input_text, validated_band):
    if not writing_input_text or not validated_band:
        return None
    
    # Generate Improved Writing Prompt Structure
    prompt = (
        f"You are an experienced IELTS writing teacher. Please revise the following essay so that it meets the requirements of a Band {validated_band}\n"
        "Keep the original meaning and ideas, but improve the vocabulary, grammar, and structure. Do not provide scores, explanations, or feedback.\n\n"
        f"{validated_band} without any grading or comments while maintaining the original meaning:\n\n{writing_input_text}\n"
    )   
    return prompt

def main():
    load_dotenv() # Load .env environment variables

    st.set_page_config(page_title="Grade your IELTS essay!") # Set page title
    st.write(css, unsafe_allow_html=True) # Load custom CSS

    # Session state setup
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "topic_text" not in st.session_state:
        st.session_state.topic_text = ""
    if "full_input_text" not in st.session_state:
        st.session_state.full_input_text = ""

    st.header("Grade your IELTS essay")
    st.subheader("📘 Step 1: Input Topic")

    with st.sidebar:
        st.header("Instruction")
        st.markdown("***Step 1: Obtain Guidelines or Evaluate Your Writing***")
        st.markdown("- Choose the :blue-background[Task Type] that you wanna evaluate. Then, enter your writing topic directly or upload an PDF file and click :blue-background[Get Writing Guidelines] to receive tailored advice for your topic.")
        st.markdown("- If you prefer not to receive guidelines, simply enter your writing topic, select your preferred input method, and click :blue-background[Evaluate Writing] to grade your submission.")
        st.markdown("***Step 2: Receive Enhanced Writing Based on Your Target Band***")
        st.markdown("- Specify your target band score and click :blue-background[Generate Improved Writing] to receive your revised essay aligned with your desired level.")

    # Topic input
    task_type = st.selectbox("Choose your task type:", ["Task 1", "Task 2"])
    input_topic = st.radio("Choose input method:", ["Direct Text", "PDF"], key="topic_input_method")

    direct_topic = None
    pdf_topic = None
    pdf_images = []
    table_summaries = []
    image_summaries = []
    topic_input_text = ""

    if input_topic == "Direct Text":
        direct_topic = st.text_area("Enter your writing topic here:")
        topic_input_text = handle_topic_input(direct_topic)

    elif input_topic == "PDF":
        pdf_topic = st.file_uploader("Upload your topic here and click on 'Get Writing Guideline'", accept_multiple_files=True)
        if pdf_topic:
            pdf_text, pdf_images, pdf_tables = get_pdf_text(pdf_topic, task_type=task_type)
            
            # Tóm tắt bảng
            table_summaries = summarize_tables(pdf_tables) if pdf_tables else []

            # Tóm tắt ảnh
            image_summaries = summarize_images(pdf_images) if pdf_images else []
            
            # Nếu có text thì dùng
            if pdf_text.strip():
                topic_input_text = pdf_text.strip()
            else:
                topic_input_text = "This task contains visual input only. Please refer to the chart or diagram for context."
            
            # Gộp tóm tắt ảnh vào cuối để tạo nội dung đầy đủ hơn cho vectorstore
            if image_summaries:
                topic_input_text += "\n" + "\n".join(image_summaries)
                
            # Store the topic content in session state
            topic_input_text = handle_topic_input(topic_input_text)

    # Button for getting guidelines
    if st.button("Get Writing Guidelines"):
        with st.spinner("Processing"):

            if not topic_input_text:
                st.error("Please provide the topic.")
                return
            
            # Generate topic guideline prompt
            guideline_prompt = get_guideline_prompt(topic_input_text, image_summaries, table_summaries)
            if not guideline_prompt:
                st.error("Failed to generate guideline prompt.")
                return

            try:
            # Gọi trực tiếp GPT-4o (không dùng LangChain)
                response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": guideline_prompt}],
                max_tokens=800
            )

                result = response.choices[0].message.content.strip()
                st.session_state.last_guideline_result = result

            except Exception as e:
                st.error(f"Failed to get response from GPT-4o: {str(e)}")

    if st.session_state.get("last_guideline_result"):
        st.subheader("📝 Guideline Result:")
        st.write(st.session_state.last_guideline_result)

     # Separate section for writing evaluation       
    st.subheader("✍️ Step 2: Input Writing")

    # Add a radio button to select the input method
    input_method = st.radio("Choose input method:", ["Direct Text", "PDF"], key="writing_input_method")
    
    writing_input_text = ""

    if input_method == "Direct Text":
        writing_text = st.text_area("Enter your essay here:")
        writing_input_text = get_writing_text(writing_text)

    elif input_method == "PDF":
        writing_pdf = st.file_uploader("Upload your essay here and click on 'Grading'", accept_multiple_files=True)
        if writing_pdf:
            writing_text, _, _ = get_pdf_text(writing_pdf, task_type="Task 2")
            writing_input_text = writing_text
            st.session_state.writing_input_text = writing_input_text  # Lưu vào session để dùng lại ở bước Improved Essay

    # Button for evaluating writing
    if st.button("Evaluate Writing"):
        with st.spinner("Processing"):

            if not writing_input_text:
                st.error("Please provide writing content.")
                return
            
            # Ensure both topic and writing content are available
            if not st.session_state.topic_text or not writing_input_text:
                st.error("Please provide both the topic and the writing content.")
                return

            # Create full input text and handle grading
            full_input_text = st.session_state.topic_text + "\n\n" + writing_input_text
            st.session_state.full_input_text = full_input_text

            # Combine text chunks from topic and writing
            text_chunks = get_combined_text_chunks(st.session_state.topic_text, writing_input_text)

            if not text_chunks:
                st.error("No valid content found in your essay. Please provide a valid input.")
                return

            # Create vector store
            vectorstore = get_vector_store(text_chunks)

            if not vectorstore:
                st.error("Failed to create vectorstore for the essay content.")
                return

            # Get conversation chain
            st.session_state.conversation = get_conversation_chain(vectorstore)

            if not st.session_state.conversation:
                st.error("Failed to initialize conversation chain.")
                return

            # Generate grading prompt
            grading_prompt = get_grading_prompt(full_input_text)

            if not grading_prompt:
                st.error("Failed to create grading prompt. Please provide valid inputs.")
                return
            
            try:
            # Gọi GPT-4o trực tiếp để đánh giá
                response = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": grading_prompt}],
                    max_tokens=800
                )

                grading_text = response.choices[0].message.content.strip()
            except Exception as e:
                st.error(f"Failed to get grading response from GPT-4o: {str(e)}")
                return

            # Trích xuất các phần đánh giá
            st.session_state.last_grading_result = {
                "task_achievement": extract_section('Task Achievement', grading_text),
                "coherence": extract_section('Coherence and Cohesion', grading_text),
                "lexical": extract_section('Lexical Resource', grading_text),
                "grammar": extract_section('Grammatical Range and Accuracy', grading_text),
                "overall": extract_section('Overall Score', grading_text)
            }

    if st.session_state.get("last_grading_result"):
        st.subheader("📊 Grading Result")
        st.write(user_template.replace("{{MSG}}", st.session_state.last_grading_result["task_achievement"]), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", st.session_state.last_grading_result["coherence"]), unsafe_allow_html=True)
        st.write(user_template.replace("{{MSG}}", st.session_state.last_grading_result["lexical"]), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", st.session_state.last_grading_result["grammar"]), unsafe_allow_html=True)
        st.write(overall_template.replace("{{MSG}}", st.session_state.last_grading_result["overall"]), unsafe_allow_html=True)

    st.subheader("✨ Step 3: Improved Essay")

    # Add target band selection
    target_band = st.selectbox("Select your target band:", options=[6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0])

    # Validate and handle the band input
    validated_band = handle_band_input(target_band)

    # Button for generating improved writing
    if st.button("Generate Improved Writing"):
        if validated_band:
            with st.spinner("Processing..."):

                # Check and use store_full_input_text in the "Generate Improved Writing" button action
                writing_input = writing_input_text or st.session_state.get("writing_input_text", "")
                corrected_prompt = get_corrected_writing_prompt(writing_input_text, validated_band)

                if not corrected_prompt:
                    st.error("Failed to generate correction prompt.")
                    return
                   
                # Recreate vectorstore and conversation chain for corrected writing (optional)
                correction_chunks = get_combined_text_chunks(writing_input_text, "")
                vectorstore = get_vector_store(correction_chunks)
                if not vectorstore:
                    st.error("Failed to create vectorstore for correction.")
                    return
                
                st.session_state.conversation = get_conversation_chain(vectorstore)
                if not st.session_state.conversation:
                    st.error("Failed to initialize conversation chain.")
                    return
                
                # Send the prompt to the LLM for correction
                response = st.session_state.conversation({"question": corrected_prompt})
                st.session_state.chat_history = response['chat_history']

                # Display LLM response
                for i, message in enumerate(st.session_state.chat_history):
                    if i % 2 == 1:
                        st.session_state.last_improved_essay = message.content
        
        else:
            st.error("Please select a valid band score.")

    if st.session_state.get("last_improved_essay"):
        st.subheader("📌 Here is your Improved Essay")
        st.write(st.session_state.last_improved_essay)

if __name__ == "__main__":
    # Initialize LLM and embeddings
    main()

