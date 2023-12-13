# -*- coding: utf-8 -*-

from telegram.ext import Application, CommandHandler, MessageHandler, filters
from dotenv import load_dotenv
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.chains import LLMRequestsChain, LLMChain
from langchain.prompts import PromptTemplate
import markdown2
from markdown2 import Markdown
import threading
import re
import requests
import openai
from openai import OpenAI
from langchain.docstore.document import Document
import logging
from textwrap import wrap
import time
import tiktoken
import requests

# возьмем переменные окружения из .env
load_dotenv('/.env')

# загружаем токен бота
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
BOT_TOKEN = os.environ.get("BOT_TOKEN")
openai_api_key = OPENAI_API_KEY

client = OpenAI()

gptmodel="gpt-4-1106-preview"

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

logging.getLogger("langchain.text_splitter").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)

# Функция загрузки документа из Google Drive
def load_document_text(url: str) -> str:
    # Extract the document ID from the URL
    match_ = re.search('/document/d/([a-zA-Z0-9-_]+)', url)
    if match_ is None:
        raise ValueError('Invalid Google Docs URL')
    doc_id = match_.group(1)

    # Download the document as plain text
    response = requests.get(f'https://docs.google.com/document/d/{doc_id}/export?format=txt')
    response.raise_for_status()
    text = response.text

    return text

# Загружаем инструкцию и базу знаний
system_text = load_document_text('https://docs.google.com/document/d/14ihOt1VLuNMpLruic-gfY7REcAXoACAw')
sum_text = load_document_text('https://docs.google.com/document/d/1Ld6EAMMTEbEpUgUII3sc2NawiG-fc_aW')
database = load_document_text('https://docs.google.com/document/d/11pdeVcF6sOyb_i060KIAi2PsSzxQPvFC')

# предобработаем текст таким образом, чтобы его можно было бы поделить на чанки при помощи MarkdownHeaderTextSplitter
def text_to_markdown(text):
    # Добавляем заголовок 1 уровня на основе римских чисел (без переноса строки)
    # и дублируем его строчкой ниже - иначе эта информация перенесется в метаданные, а порой она бывает полезной.
    def replace_header1(match):
        return f"# {match.group(2)}\n{match.group(2)}"

    text = re.sub(r'^(I{1,3}|IV|V)\. (.+)', replace_header1, text, flags=re.M)

    # Добавляем текст, выделенный жирным шрифтом (он заключен между *)
    # и дублируем его строчкой ниже
    def replace_header2(match):
        return f"## {match.group(1)}\n{match.group(1)}"

    text = re.sub(r'\*([^\*]+)\*', replace_header2, text)

    return text

markdown = text_to_markdown(database)

# Делим текст на чанки и создаем индексную базу
source_chunks = []

splitter = CharacterTextSplitter(separator="\n", chunk_size=1024, chunk_overlap=0)

for chunk in splitter.split_text(database):
     source_chunks.append(Document(page_content=chunk, metadata={}))

def num_tokens_from_string(string: str, encoding_name: str) -> int:
      """Возвращает количество токенов в строке"""
      encoding = tiktoken.get_encoding(encoding_name)
      num_tokens = len(encoding.encode(string))
      return num_tokens

def split_text(text, max_count):
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    fragments = markdown_splitter.split_text(text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_count,
        chunk_overlap=0,
        length_function=lambda x: num_tokens_from_string(x, "cl100k_base")
    )

    source_chunks = [
        Document(page_content=chunk, metadata=fragment.metadata)
        for fragment in fragments
        for chunk in splitter.split_text(fragment.page_content)
    ]

    return source_chunks

source_chunks = split_text(markdown, 1000)

# Инициализирум модель эмбеддингов
embeddings = OpenAIEmbeddings()

# Создадим индексную базу из разделенных фрагментов текста
db = FAISS.from_documents(source_chunks, embeddings)

# Функции Search
def get_input(question: str):
    inputs = {
        "query": question,
        "url": "https://www.google.com/search?q=" + question.replace(" ", "+"),
    }
    return inputs

# Подготовка темплейта для запроса
search_template = """Between >>> and <<< are the raw search result text from google.
Look for information only within the scope of palliative care in accordance with the legislation of the Russian Federation. 
Extract the answer to the question '{query}' or say "not found" if the information is not contained.
Use the format
Extracted:<answer or "not found">
>>> {requests_result} <<<
Extracted:"""

PROMPT = PromptTemplate(
    input_variables=["query", "requests_result"],
    template=search_template,
)

# Инициализация цепочки запросов
search_chain = LLMRequestsChain(llm_chain=LLMChain(llm=ChatOpenAI(temperature=0.4), prompt=PROMPT))

def answer_user_question_combined(system_text, db, user_question, question_history):
    """
    Функция возвращает ответ на вопрос пользователя с учетом истории диалога.
    """

    # Инициализация summarized_history
    summarized_history = ""

    # Выполняем поиск ответа на основе вопроса пользователя
    docs = db.similarity_search_with_score(user_question, k=4)
    filtered_docs = [(doc, score) for doc, score in docs if score < 0.45]
    scores = [str(score) for _, score in docs]  # скоры берем из всех документов, не только отфильтрованных

    # если filtered_docs не пустой список:
    if filtered_docs:
        message_content = re.sub(r'\n{2}', ' ', '\n '.join([f'\nОтрывок документа №{i+1}\n=====================' + doc.page_content + '\n' for i, (doc, score) in enumerate(filtered_docs)]))

        # Формирование истории диалога
        if len(question_history) > 0:
            summarized_history = "Вот краткая тема предыдущего диалога: " + summarize_questions([q + ' ' + (a if a else '') for q, a in question_history])

        # Добавляем явное разделение между историей диалога и текущим вопросом
        input_text = summarized_history + "\n\nТекущий вопрос: " + user_question

        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": f"The answer to the patient's question based on these documents: a document with information to answer the patient: {message_content}\n\Patient's question: \n{input_text}. Don't make up unnecessary information. Answer honestly if you don't know the answer to the question. Never mention the document or excerpts from it. Create the illusion that all the information in the document is your own knowledge. Never make up information about a patient's data. Provide the necessary psychological help only if it is required. Answer only in the language spoken by the user, by default: Russian"}
        ]

        completion = client.chat.completions.create(
            model=gptmodel,
            messages=messages,
            temperature=0.3
        )
        answer_text = completion.choices[0].message.content

    # если filtered_docs пустой:
    else:
        # Получаем данные для поискового запроса
        question_input = get_input(user_question)

        message_content="Информации по вопросу не найдено"
        
        # Выполняем поиск
        search_result = search_chain.run(question_input)
        if 'output' in search_result:
            search_response = search_result['output']
        else:
            search_response = "Информация по запросу не найдена."

        # Формирование истории диалога
        if len(question_history) > 0:
            summarized_history = "Вот краткая тема предыдущего диалога: " + summarize_questions([q + ' ' + (a if a else '') for q, a in question_history])

        # Добавляем явное разделение между историей диалога и текущим вопросом
        input_text = summarized_history + "\n\nТекущий вопрос: " + user_question + "\n" + search_response

        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": f"The answer to the patient's question based on these documents: a document with information to answer the patient: {message_content}\n\Patient's question: \n{input_text}. Don't make up unnecessary information. Answer honestly if you don't know the answer to the question. Never mention the document or excerpts from it. Create the illusion that all the information in the document is your own knowledge. Never make up information about a patient's data. Provide the necessary psychological help only if it is required. Answer only in the language spoken by the user, by default: Russian"}
        ]

        completion = client.chat.completions.create(
            model=gptmodel,
            messages=messages,
            temperature=0.5
        )
        answer_text = completion.choices[0].message.content

    # Добавляем вопрос пользователя и ответ системы в историю
    question_history.append((user_question, answer_text if answer_text else ''))

    return answer_text

def summarize_questions(dialog):
    """
    Функция возвращает саммаризированный текст диалога.
    """
    messages = [
        {"role": "system", "content": sum_text},
        {"role": "user", "content": f"Саммаризируй только краткую тему диалога менеджера поддержки и пациента, не более 100 символов: " + " ".join(dialog)}
    ]

    completion = client.chat.completions.create(
        model="gpt-4",     # используем gpt4 для более точной саммаризации
        messages=messages,
        temperature=0.3,          # Используем более низкую температуру для более определенной суммаризации
    )

    return completion.choices[0].message.content


def run_dialog(system_text, db, user_question, question_history):
    """
    Функция обрабатывает входящие текстовые сообщения от пользователя и возвращает ответ.
    """

    # Получение ответа на вопрос пользователя с использованием функции answer_user_question_combined
    ans = answer_user_question_combined(system_text, db, user_question, question_history)

    # Добавление запроса пользователя и ответа в историю
    question_history.append((user_question, ans))
    
    return ans

global_db = db
global_question_history = []

# функция команды /start
async def start(update, context):
    await update.message.reply_text('Привет! Это тест-бот для паллиативных больных')

# функция для текстовых сообщений
async def text(update, context):
    # использование update
    print(update)
    print('-------------------')
    print(f'text: {update.message.text}')
    print(f'date: {update.message.date}')
    print(f'id message: {update.message.message_id}')
    print(f'name: {update.message.from_user.first_name}')
    print(f'user.id: {update.message.from_user.id}')
    print('-------------------')

    user_text = update.message.text
    response_text = run_dialog(system_text, global_db, user_text, global_question_history)

    await update.message.reply_text(f'Ответ нейро-сотрудника:\n {response_text}')

def main():
    # точка входа в приложение
    application = Application.builder().token(BOT_TOKEN).build()
    print('Бот запущен..!')

    # добавляем обработчик команды /start
    application.add_handler(CommandHandler("start", start))

    # добавляем обработчик текстовых сообщений
    application.add_handler(MessageHandler(filters.TEXT, text))

    # запуск приложения (для остановки нужно нажать Ctrl-C)
    application.run_polling()
    application.idle()

if __name__ == "__main__":
    main()
