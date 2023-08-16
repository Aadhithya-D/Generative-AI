from langchain import ConversationChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import Document
import pandas as pd
from langchain.chains.conversational_retrieval.prompts import QA_PROMPT
import time
from langchain.memory import ChatMessageHistory
import os
import openai

api_key = "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"
openai.api_key = api_key


def read_csv_into_vector_document(file, text_cols):
    CSVData = pd.read_csv(file, encoding='latin-1')

    CSVData = CSVData[text_cols]
    CSVData = CSVData.values.tolist()
    # print(data)
    for i in range(len(CSVData)):
        CSVData[i] = " ".join(str(CSVData[i]))
    return [Document(page_content=text) for text in CSVData]


if not api_key:
    print('OpenAI API key not found in environment variables.')
    exit()

# data = read_csv_into_vector_document("D:\\Generative-AI\\test 1\\training.csv", ["UserName", "ScreenName",
# "Location", "TweetAt", "OriginalTweet", "Sentiment", ]) embeddings = OpenAIEmbeddings(openai_api_key=api_key)
# vectors = FAISS.from_documents(data, embeddings)
memory = ConversationBufferMemory(
    # memory_key='chat_history', return_messages=True, output_key='answer'
)
chain = ConversationChain(
    llm=ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo', openai_api_key=api_key),
    # retriever=vectors.as_retriever(),
    memory=memory,
    # return_source_documents=True,
    # get_chat_history=lambda h: h,
    # qa_prompt=QA_PROMPT
)
history = []

while True:
    query = input("Human: ")
    ai_message = chain.predict(input=query)
    # history = (query, ai_message["answer"])
    # history1.append(history)
    # # print(history1)
    print("AI: "+ai_message)
