from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
memory = ConversationBufferMemory()
llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0, streaming=True)

chain = ConversationChain(llm=llm, memory=memory)

user_input = "What is the capital of France?"
response = chain.predict(input=user_input)
print(response)

user_input = "What is the capital of Japan?"
response = chain.predict(input=user_input)
print(response)

print(memory.chat_memory.messages)
