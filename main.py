from dotenv import load_dotenv
from tools.search_tools import SearchTools
from model_config import LLMModel


load_dotenv(".env")

llm = LLMModel().provide_model(local=False, gpt=False)



# scrap the pages and get text content

def get_webpage_content(url):
    from langchain_community.document_loaders import AsyncHtmlLoader
    from langchain_community.document_transformers import Html2TextTransformer
    print("Getting content from the webpage")
    # urls = ["https://www.espn.com", "https://lilianweng.github.io/posts/2023-06-23-agent/"]
    # urls = ["https://lilianweng.github.io/posts/2023-06-23-agent/"]
    # https://aws.amazon.com/what-is/transformers-in-artificial-intelligence/
    if url is "":
        url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
    urls = [url]
    loader = AsyncHtmlLoader(urls)
    docs = loader.load()

    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)
    page_content = docs_transformed[0].page_content[520:2000]
    print("page content",page_content)
    return page_content





    
# get_webpage_content("https://aws.amazon.com/what-is/transformers-in-artificial-intelligence/")

default_schema = {
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
    },
    "required": ["title", "summary"],
}

def extract(content: str, schema: dict = default_schema):
    from langchain.chains import create_extraction_chain
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
    output = create_extraction_chain(schema=schema,llm=llm).run(content)
    print("output",output)
    return output


def summarize_text(docs):
    from langchain_openai import ChatOpenAI
    from langchain.chains.combine_documents.stuff import StuffDocumentsChain
    from langchain.chains.llm import LLMChain
    from langchain_core.prompts import PromptTemplate
    prompt_template = """Write a concise summary of the following:
    "{text}"
    CONCISE SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)

    # Define LLM chain
    # llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Define StuffDocumentsChain
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
    if docs is type(list) and len(docs) > 0:
        docs = docs[0]

    # string length in python
    if len(docs) >=1000:
        docs = docs[0:1000]
    print("docs",docs)
    
    # Converting a string to a document object
    from langchain.docstore.document import Document
    doc =  Document(page_content=docs, metadata={"source": "local"})
    print("doc",doc)
    output = stuff_chain.invoke([doc])
    print("output",output)
    return output








# Create a summary from the content 
def create_summary_from_structured_data(url):
    web_content = get_webpage_content(url)
    structured_output = extract(web_content)
    print("structured_output",structured_output)
    return (structured_output,web_content)




# create_summary() 

def generate_tweets(text: str = None):
    # text = {
    #     "text": [
    #         {
    #         "news_article_title": "Building agents with LLM as its core controller",
    #         "news_article_summary": "Building agents with LLM (large language model) as its core controller is a cool concept. Several proof-of-concepts demos, such as AutoGPT, GPT-Engineer and BabyAGI, serve as inspiring examples. The potentiality of LLM extends beyond generating well-written copies, stories, essays and programs; it can be framed as a powerful general problem solver."
    #         },
    #         {
    #         "news_article_title": "Agent System Overview",
    #         "news_article_summary": "In a LLM-powered autonomous agent system, LLM functions as the agent's brain, complemented by several key components: Planning, Memory, and Tool use."
    #         }
    #     ]
    # }


    # text = create_summary("https://lilianweng.github.io/posts/2023-06-23-agent/")

    from langchain_openai import ChatOpenAI
    print(text)
    # llm = ChatOpenAI()

    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are expert in creating viral tweets. Your tweets should be engaging and informative and appeal to mass audience. You have to create tweets from the given text."),
        ("user", "{context}")
    ])

   
    chain = prompt | llm 
    story = chain.invoke({"context": text })
    print("story",story.content)
    return story.content

# realtime_knowledge_article = SearchTools.search_internet("transformers in artificial intelligence?") 
# content = get_webpage_content("")
# summary = summarize_text(content)
# generate_tweets(summary)

search_result_memo =  [
        {
            "title": "What Are Transformers In Artificial Intelligence? - Amazon AWS",
            "link": "https://aws.amazon.com/what-is/transformers-in-artificial-intelligence/",
            "snippet": "Transformers are a type of neural network architecture that transforms or changes an input sequence into an output sequence. They do this by learning ...",
            "sitelinks": [
                {
                    "title": "What are transformers in...",
                    "link": "https://aws.amazon.com/what-is/transformers-in-artificial-intelligence/#seo-faq-pairs#what-are-transformers-in-artificial-intelligence"
                },
                {
                    "title": "Why are transformers important?",
                    "link": "https://aws.amazon.com/what-is/transformers-in-artificial-intelligence/#seo-faq-pairs#why-are-transformers-important"
                }
            ],
            "position": 1
        },
        {
            "title": "What Is a Transformer Model? | NVIDIA Blogs",
            "link": "https://blogs.nvidia.com/blog/what-is-a-transformer-model/",
            "snippet": "A transformer model is a neural network that learns context and thus meaning by tracking relationships in sequential data like the words in this ...",
            "date": "Mar 25, 2022",
            "position": 2
        },
        {
            "title": "How Transformers Work - Towards Data Science",
            "link": "https://towardsdatascience.com/transformers-141e32e69591",
            "snippet": "Transformers are a type of neural network architecture that have been gaining popularity. Transformers were recently used by OpenAI in their language models, ...",
            "sitelinks": [
                {
                    "title": "Making Things Think: How Ai...",
                    "link": "https://towardsdatascience.com/transformers-141e32e69591#:~:text=Making%20Things%20Think%3A%20How%20AI%20and%20Deep%20Learning%20Power%20the%20Products%20We%20Use%20%2D%20Holloway"
                },
                {
                    "title": "Recurrent Neural Networks",
                    "link": "https://towardsdatascience.com/transformers-141e32e69591#:~:text=Recurrent%20Neural%20Networks,-Recurrent%20Neural%20Networks"
                },
                {
                    "title": "Attention",
                    "link": "https://towardsdatascience.com/transformers-141e32e69591#:~:text=Attention,-To%20solve%20some%20of%20these"
                }
            ],
            "position": 3
        },
        {
            "title": "Transformer (deep learning architecture) - Wikipedia",
            "link": "https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)",
            "snippet": "A transformer is a deep learning architecture developed by Google and based on the multi-head attention mechanism, proposed in a 2017 paper \"Attention Is ...",
            "sitelinks": [
                {
                    "title": "Attention Is All You Need",
                    "link": "https://en.wikipedia.org/wiki/Attention_Is_All_You_Need"
                },
                {
                    "title": "Attention (machine learning)",
                    "link": "https://en.wikipedia.org/wiki/Attention_(machine_learning)"
                },
                {
                    "title": "Word embedding",
                    "link": "https://en.wikipedia.org/wiki/Word_embedding"
                },
                {
                    "title": "Wu Dao",
                    "link": "https://en.wikipedia.org/wiki/Wu_Dao"
                }
            ],
            "position": 4
        },
        {
            "title": "Transformers Revolutionized AI. What Will Replace Them? - Forbes",
            "link": "https://www.forbes.com/sites/robtoews/2023/09/03/transformers-revolutionized-ai-what-will-replace-them/",
            "snippet": "Transformers have become the foundation of modern artificial intelligence. Virtually every advanced AI system is based on transformers; every AI ...",
            "date": "Sep 3, 2023",
            "position": 5
        }
    ]

# realtime_knowledge_snippets = SearchTools.search_result_converter(search_result_memo)

# print("realtime_knowledge_snippets",realtime_knowledge_snippets)
# summary = summarize_text(realtime_knowledge_snippets)
# generate_tweets(summary)


def main():
    import streamlit as st
    st.set_page_config(page_title="TweetCraft", page_icon="📸")
    st.title("TweetCraft")
    st.write("TweetCraft is a tool to generate tweets or tweet threads from a given text.Based on latest information")
   

    with st.form('keywords_form'):
        st.info("Add keywords for real time research and generate tweets")
        text = st.text_input("Enter the Keywords:")
        submitted = st.form_submit_button("Submit")
        if submitted:
            if text is "":
                st.error("Please enter the keywords to generate tweets.")
                return
            else: 
                st.write("Generating tweets from the given keywords {text}".format(text=text))
                # search_result_memo = SearchTools.search_internet(text)
                # search_result_memo = 
                realtime_knowledge_snippets = SearchTools.search_result_converter(search_result_memo)
                summary = summarize_text(realtime_knowledge_snippets)
                with st.expander("Realtime Knowledge Snippets"):
                    st.write(realtime_knowledge_snippets)
                with st.expander("Clean Structured Output"):
                    st.write(summary)
                output = generate_tweets(summary)
                st.write(output)


    with st.form('url_form'):
        st.info("Add url to generate tweets from the webpage.")
        url = st.text_input("Enter the Web Url:")
        submitted = st.form_submit_button("Submit")
        if submitted:
            if url is None or url == "":
                st.error("Please enter the url to generate tweets.")
                return
            else:
                st.write("Generating tweets from the given url {url}".format(url=url))
                # if url is not None:
                # content = get_webpage_content(url)
                # structured_output = extract(content)
                structured_output, web_content = create_summary_from_structured_data(url)  # Assign the return value of create_summary to structured_output and content separately
                with st.expander("Webpage Content"):
                    st.write(web_content)
                with st.expander("Clean Structured Output"):
                    st.write(structured_output)
                output = generate_tweets(structured_output[0]["summary"])  # Use structured_output as an argument for generate_tweets
                st.write(output)




    

    # if uploaded_file is not None:
    #     print(uploaded_file)
    #     bytes_data = uploaded_file.getvalue()
    #     st.image(bytes_data, caption="Uploaded Image.", use_column_width=True)
       

if __name__ == "__main__":
    main()



# TODO: Add serper api 
# TODO: Pick url from the serper api and generate tweets
# TODO: Add chunking for scrapped webpage context.
# TODO: Save the scrapped content to a file