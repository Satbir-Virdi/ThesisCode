# from openai import OpenAI
# from langchain.indexes import VectorstoreIndexCreator
# from langchain.document_loaders import TextLoader
# from langchain.chat_models import ChatOpenAI

from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.chains import ConversationalRetrievalChain, RetrievalQA, ConversationChain
import openai
import sys
import os
from openai import OpenAI
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import TextLoader
from langchain.chat_models import ChatOpenAI

# client = OpenAI(
#     api_key="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",
# )
# #ft:gpt-3.5-turbo-1106:personal::8S6eJd5T


def create_index():
    loader = TextLoader('data2.txt')
    index = VectorstoreIndexCreator().from_loaders([loader])
    return index


def chat_with_gpt(conversation):
    return index.query(conversation, llm=ChatOpenAI(OPENAI_API_KEY="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"))
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=conversation,
        max_tokens=1000,
        temperature=0,
        presence_penalty=0.5,
        frequency_penalty=0.5,
        top_p=0.3)

    return response.choices[0].message.content


client = OpenAI(
    api_key="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",
)

os.environ["OPENAI_API_KEY"] = "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"

query = None
if len(sys.argv) > 1:
    query = sys.argv[1]

loader = TextLoader("data2.txt")  # Use this line if you only need data.txt
index = VectorstoreIndexCreator().from_loaders([loader])

chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-3.5-turbo-1106"),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)


chat_history = []

while True:
    if not query:
        query = input("you: ")
    if query in ['quit', 'q', 'exit']:
        sys.exit()
    result = chain({"question": query, "chat_history": chat_history})
    print(f"model: {result['answer']}\n\n")

    chat_history.append((query, result['answer']))
    # chat_with_gpt(result)
    query = None


# if __name__ == '__main__':

#     index = create_index()

#     conversation = [
#         {"role": "system", "content": "You are an investment advisor chatbot that wants to provide the funds with the best average net IRR"},
#     ]

#     start = True
#     while True:
#         # start the conversation
#         if start:
#             print(
#                 "Chatbot: Hello, how may I assist you with your endowment-related investments?")
#             start = False

#         # get user input
#         user_input = input("You: ")
#         if user_input.lower() in ['quit', 'end', 'bye']:
#             break

#         # add it to the conversation
#         conversation.append({"role": "user", "content": user_input})

#         response = chat_with_gpt(conversation)
#         print("Chatbot: ", response)
#         conversation.append({"role": "assistant", "content": response})
