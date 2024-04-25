import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from PIL import Image, ImageTk
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
import os
from langchain.chains.question_answering import load_qa_chain

os.environ["OPENAI_API_KEY"] = "--Enter OpenAI Key--"

try:
    pdf_reader = PdfReader(r"--PDF Path--")
    raw_text = ''
    for i, page in enumerate(pdf_reader.pages):
        content = page.extract_text()
        if content:
            raw_text += content
except Exception as e:
    print(f"Error reading PDF: {e}")
    # Handle the error or display a message to the user

# from typing_extensions import Concatenate
# read text from pdf
raw_text = ''
for i, page in enumerate(pdf_reader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

# We need to split the text using Character Text Split such that it should not increse token size
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 800,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings()

document_search = FAISS.from_texts(texts, embeddings)

document_search

chain = load_qa_chain(OpenAI(), chain_type="stuff")


def get_response():
    user_input = user_input_entry.get("1.0", tk.END)
    bot_response = ""  # Initialize bot_response with an empty string

    if user_input.strip().lower() in ["hi", "hello", "hey","hy","hi ruby","hello ruby","hey ruby","hy ruby"]:
        bot_response = "Hello! How can I assist you today!"
    elif user_input.strip().lower() in ["bye","by","bye ruby","by ruby","thank you","thanks"]:
        bot_response = "Good bye and take care."
    else:
        question = user_input.strip()  # Extract and clean the question from user input
        if len(question) < 4:
            bot_response = "Please enter a valid question!"
        else:
            docs = document_search.similarity_search(user_input)
            bot_response = chain.run(input_documents=docs, question=question)


    chat_window.config(state=tk.NORMAL)
    chat_window.insert(tk.END, "You: " + user_input + "\n", "user")
    chat_window.insert(tk.END, "Ruby: " + bot_response + "\n", "bot")
    chat_window.config(state=tk.DISABLED)
    user_input_entry.delete("1.0", tk.END)



root = tk.Tk()
root.title("Parkinson's Disease Guidance Chatbot")


notebook = ttk.Notebook(root)
notebook.pack(fill=tk.BOTH, expand=True)


home_frame = ttk.Frame(notebook)
about_us_frame = ttk.Frame(notebook)
contact_us_frame = ttk.Frame(notebook)


notebook.add(home_frame, text="Parkinson's Disease Chatbot")

image = Image.open(r"C:\Users\aiwin\OneDrive\Desktop\Hub\P5_Parkinson\P5_Parkinson\image.png")

max_width = 200
max_height = 100
image.thumbnail((max_width, max_height), Image.LANCZOS)

photo = ImageTk.PhotoImage(image)
image_label = tk.Label(home_frame, image=photo)
image_label.image = photo
image_label.pack(pady=20)


chat_window = scrolledtext.ScrolledText(home_frame, wrap=tk.WORD, state=tk.DISABLED, height=20, width=70, bd=0, bg='bisque3')
chat_window.tag_configure("user", foreground="red")
chat_window.tag_configure("bot", foreground="black")
chat_window.pack(pady=20)


user_input_entry = tk.Text(home_frame, height=3, width=50)
user_input_entry.pack()


ask_button = tk.Button(home_frame, text="Ask", command=get_response, height=2, width=10)
ask_button.pack(pady=20)

root.mainloop()
