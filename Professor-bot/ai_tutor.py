# Trying to integrating PromptTemplate and memory into one and also using some addtional things to make our output better 

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage , HumanMessage

# Creating the model:
llm = OllamaLLM(model = 'llama3.1:8b')

# Setting the memory for conversation
memory = ConversationBufferMemory()

# Creating a prompt template to make sure the model throws the error as per the parameters I am setting for it.
prompt = PromptTemplate(
    input_variables = ['topic', 'audience'],
    template = """
    You are a Professor from IIT who knows all about Machine Learning and AI.
    You explain the concepts in a very simple way with daily life examples.
    
    Topic = {topic}
    Audience = {audience}
    
    Task :
    - Give a very brief explanation on {topic} such that it is easily understandable to the {audience}
    - Give 5 tips on how the {audience} can improve or use those tips in their work.
    - End with a short Python code example  explanation.
    """
)
chain = LLMChain(llm = llm, prompt = prompt)

# Setting the memory so that the model retains the information
conversation = ConversationChain(
    llm = llm,
    memory = memory, 
    verbose = True
)

print("--------- Type 'exit' anytime to quit. ---------")
while True:
    question = input("Ask Me Anything 😤  :  ")
    
    if question.lower() == 'exit':
        break
    else:
        topic = question
        audience = input('Who is the Audience🧑‍🦱  :  ')
        result = chain.run(topic = topic, audience = audience)
        print('\n🤖AI Response : \n', result)

        convo_input = f"""
            Topic: {topic}
            Audience: {audience}
            Structured Response: {result}
        """
        convo_response = conversation.predict(input = convo_input)
        print('\n🧠Conversation Memory: \n', convo_response)