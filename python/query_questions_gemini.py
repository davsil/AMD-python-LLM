import os 
import sys
import argparse
import time
import fcntl
import markdown
import pandas as pd
import re
import google.generativeai as genai

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate


gemini_model_name = "gemini-1.5-pro"
gemini_store_path = "gemini-faiss-index"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def format_docs(docs):
    return  "\n\n".join(doc.page_content for doc in docs)


def format_string_into_blocks(text, max_line_length):
    words = text.split()
    lines = []
    current_line = ''

    for word in words:
        if len(current_line) + len(word) + 1 <= max_line_length:
            if current_line:
                current_line += ' '
            current_line += word
        else:
            lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return '\n'.join(lines)


def remove_markdown(text):
    # Extended patterns to remove all specified Markdown formatting, excluding quotes
    text = re.sub(r'\#\#*\s+(.*)', r'\1', text)  # Headings like ## Heading or ### Heading
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*(.*?)\*', r'\1', text)  # Italic
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)  # Links
    text = re.sub(r'`{3,}(.*?)`{3,}', r'\1', text, flags=re.DOTALL)  # Multi-line code blocks
    text = re.sub(r'`(.+?)`', r'\1', text)  # Inline code
    # Remove blockquotes if they were previously considered, here's how to keep them
    # If you were removing blockquotes, adjust or remove that part of the regex
    return text


def ask_gemini(llm, model, db, query, api_key, rag=False):

    snippets = []
    context = ""

    if rag:
        docs = db.similarity_search_with_relevance_scores(query, k=9)
        for d in range(docs.__len__()):
           content = docs[d][0].page_content.replace('\n', ' ').lstrip('.')
           source = os.path.basename(docs[d][0].metadata['source']).rstrip('.txt').replace('_', ' ')
           relevance = f"{docs[d][1]*100:.1f}"
           label = f"From {source} with {relevance}% relevance"
           snippets.append((label, content, relevance))

    if rag:
        prompt_template = """Answer as a helpful assistant for AMD technologies and python.
        Always answer the question even when the context does not have relevant information.
        Do not state that the context does not contain the information.
        context = {context}
        question = {question}
        """
        prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"context": docs, "question": query})
    else:
        prompt_template = """Answer as a helpful assistant for AMD technologies and python.
        question = {question}
        """
        prompt = PromptTemplate(input_variables=["question"], template=prompt_template)
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"question": query})

    return response, snippets



def main(file_base):

    genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

    # Read questions and categories from a CSV file
    # Ensure the CSV file has columns 'Question' and 'Category'
    df = pd.read_csv(f'{file_base}.csv', delimiter=';')

    # Add a column for responses
    df['Response'] = ""
    df['Model'] = ""

    # df_norag = pd.DataFrame(columns=['Question', 'Answer', 'Model'])
    answers_norag = []
    answers_snippets = []

    embeddings = GoogleGenerativeAIEmbeddings( model='models/text-embedding-004', google_api_key=GOOGLE_API_KEY)
    db = FAISS.load_local(gemini_store_path, embeddings, allow_dangerous_deserialization=True)

    llm = ChatGoogleGenerativeAI(model=gemini_model_name,
                max_new_tokens=65536, max_tokens=65536,
                system_instruction="",
                google_api_key=GOOGLE_API_KEY)

    qnum = 0
    # Loop through DataFrame and get answers
    for index, row in df.iterrows():
        question = row['Question']
        print(f"Submitting question {index + 1}/{len(df)}: {question}")
        qnum = qnum + 1

        # call LLM service without RAG 
        response, snippets = ask_gemini(llm, gemini_model_name, db,
                                        question, GOOGLE_API_KEY, rag=False)

        answers_norag.append([question, response, gemini_model_name])

        # call LLM service with RAG 
        time.sleep(1)  # Sleep to respect rate limits
        response, snippets = ask_gemini(llm, gemini_model_name, db,
                                        question, GOOGLE_API_KEY, rag=True)

        # clean_response = remove_markdown(response)
        # df.at[index, 'Response'] = clean_response
        df.at[index, 'Response'] = response 
        df.at[index, 'Model'] = gemini_model_name 

        for s in snippets:
             answers_snippets.append([question, f"{s[0]}", f"{s[2]}", f"{s[1]}"])

        time.sleep(1)  # Sleep to respect rate limits

        # Responses saved every 25 questions in case program aborts
        if (qnum % 25) == 0:
            df.to_csv(f'{file_base}_answers_tmp.csv', sep=';', index=False)
            df_norag = pd.DataFrame(answers_norag, columns=['Question', 'Response', 'Model'])
            df_norag.to_csv(f'{file_base}_answers_norag_tmp.csv', sep=';', index=False)
            df_snippets = pd.DataFrame(answers_snippets, columns=['Question', 'Source', 'Relevance', 'Snippet'])
            df_snippets.to_csv(f'{file_base}_snippets_tmp.csv', sep=';', index=False)

    # Save the DataFrame with answers to a CSV file
    df.to_csv(f'{file_base}_answers.csv', sep=';', index=False)

    df_norag = pd.DataFrame(answers_norag, columns=['Question', 'Response', 'Model'])
    df_norag.to_csv(f'{file_base}_answers_norag.csv', sep=';', index=False)

    df_snippets = pd.DataFrame(answers_snippets, columns=['Question', 'Source', 'Relevance', 'Snippet'])
    df_snippets.to_csv(f'{file_base}_snippets.csv', sep=';', index=False)

    print(f"Reponses saved to {file_base}_answers.csv")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="File base for question and answer files")
    parser.add_argument('--file_base', type=str, default="questions", help='File base for question and answer files')
    args = parser.parse_args()
    if args.file_base.endswith('.csv'):
        args.file_base = args.file_base.replace('.csv', '')

    main(args.file_base)

