import csv
import os
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import Document
import pandas as pd
import time
from langchain.memory import ChatMessageHistory



def read_csv_into_vector_document(file, text_cols):
    # with open(file, encoding='utf-8') as csv_file:
    #     csv_reader = csv.DictReader(csv_file)
    #     text_data = []
    #     for row in csv_reader:
    #         text = ' '.join([row[col] for col in text_cols])
    #         text_data.append(text)
    #     return [Document(page_content=text) for text in text_data]
    # write the above code in pandas
    data = pd.read_csv(file, encoding='latin-1')
    
    data = data[text_cols]
    data = data.values.tolist()
    # print(data)
    for i in range(len(data)):
        data[i] = " ".join(str(data[i]))
    return [Document(page_content=text) for text in data]


api_key = "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"

if not api_key:
    print('OpenAI API key not found in environment variables.')
    exit()

data = read_csv_into_vector_document("D:\\Generative AI\\test 1\\training.csv", ["UserName", "ScreenName", "Location", "TweetAt", "OriginalTweet", "Sentiment",])
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
vectors = FAISS.from_documents(data, embeddings)
chain = ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo', openai_api_key=api_key), retriever=vectors.as_retriever(), condense_question_llm=ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo', openai_api_key=api_key))

history1 = []

while True:
    query = input("Enter Your Query:")
    ai_message = chain({"question": query, "chat_history": history1})
    history = (query, ai_message["answer"])
    history1.append(history)
    # print(history1)
    print(ai_message["answer"])