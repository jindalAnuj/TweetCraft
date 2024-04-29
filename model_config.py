import os
from dotenv import load_dotenv
load_dotenv(".env")

class LLMModel():
    def __init__(self):
        pass
    def provide_model(self,local=False,gpt=False):
        if local:
           from langchain_community.llms import Ollama
           print("using local model")
           return Ollama(model="llama2")
        else:
            from langchain_openai import ChatOpenAI
            if gpt:
                print("using gpt model")
                os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
                os.environ["OPENAI_MODEL_NAME"] =os.environ.get("OPENAI_MODEL_NAME")
                return ChatOpenAI(model_name="gpt-3.5", temperature=0.7)
            else:
                print("using grok model")
                os.environ["OPENAI_API_KEY"] = os.environ.get("GROQ_API_KEY")
                os.environ["OPENAI_API_BASE"] = os.environ.get("GROQ_API_BASE")
                os.environ["OPENAI_MODEL_NAME"] =os.environ.get("GROK_MODEL_NAME")
                return ChatOpenAI(model_name="llama3-70b-8192", temperature=0.7)

    