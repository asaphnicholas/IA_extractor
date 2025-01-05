import logging
import streamlit as st
import requests 
from bs4 import BeautifulSoup
from openai import OpenAI
from abc import ABC, abstractmethod
from typing import List, Optional

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def parse_generic_page(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    structured_data = {
        "title": soup.title.string if soup.title else "Not found",
        "metadata": {},
        "links": [],
        "headings": [],
        "paragraphs": [],
    }

    # Extract metadata
    for meta in soup.find_all('meta'):
        if 'name' in meta.attrs and 'content' in meta.attrs:
            structured_data["metadata"][meta['name']] = meta['content']

    # Extract important links
    for link in soup.find_all('a', href=True):
        structured_data["links"].append({
            "text": link.text,
            "href": link['href']
        })
    
    # Extract headings
    for heading in soup.find_all(['h1', 'h2', 'h3']):
        structured_data["headings"].append({
            "level": heading.name,
            "text": heading.text.strip()
        })

    # Extract paragraphs
    for paragraph in soup.find_all('p'):
        if paragraph.text.strip():
            structured_data["paragraphs"].append(paragraph.text.strip())
    
    return structured_data

class BaseNode(ABC):
    def __init__(self, node_name: str, input: str, output: List[str], node_config: Optional[dict] = None):
        self.node_name = node_name
        self.input = input
        self.output = output
        self.node_config = node_config or {}

    @abstractmethod
    def execute(self, state: dict) -> dict:
        pass

class FetchNode(BaseNode):
    def __init__(self, input: str, output: List[str], node_config: Optional[dict] = None):
        super().__init__("Fetch", input, output, node_config)
    
    def execute(self, state: dict) -> dict:
        url = state.get(self.input)
        if not url:
            raise ValueError(f'URL not found in state with key: {self.input}')
        
        response = requests.get(url)
        html_content = response.text
        state[self.output[0]] = html_content

        # Parse the HTML content
        parsed_data = parse_generic_page(html_content)
        state['parsed_data'] = parsed_data

        return state
    
class AnalyzeNode(BaseNode):
    def __init__(self, input: List[str], output: List[str], node_config: Optional[dict] = None):
        super().__init__("Analyze", input, output, node_config)
        self.client = OpenAI(api_key=node_config.get('api_key'))
    
    def execute(self, state: dict) -> dict:
        parsed_data = state.get(self.input[0])
        user_query = state.get(self.input[1])

        if not parsed_data or not user_query:
            raise ValueError('Parsed data or user query not found in state')
        
        prompt = f"""Analyze the following structured data from a web page and the user's query:
    
Structured Data:
{parsed_data}

User Query: {user_query}

Plase provide a detailed answer on the structured data:"""
        
        response = self.client.chat.completions.create(
            model = self.node_config.get('model', 'gpt-3.5-turbo'),
            messages = [
                {"role": "system", "content": "You are a helpful assistant that analyzes web content."},
                {"role": "user", "content": prompt}
            ]
        )

        analysis = response.choices[0].message.content.strip()
        state[self.output[0]] = analysis
        return state
    
class SmartScraperGraph:
    def __init__(self, prompt: str, source: str, config: dict):
        self.prompt = prompt
        self.source = source
        self.config = config
        self.nodes = [
            FetchNode("source", ["html_content"]),
            AnalyzeNode(["parsed_data", "prompt"], ["analysis"], config['llm'])
        ]
    
    def run(self):
        state = {"source": self.source, "prompt": self.prompt}
        for node in self.nodes:
            state = node.execute(state)
        return state
    
# Streamlit setup
st.set_page_config(page_title="Web Scraping AI Agent", page_icon="🕵🏼", layout='wide')
st.title("Web scraping AI Agent 🕵🏼")
st.caption("Esse AI App permite que você faça scrape num site e analise seu conteúdo usando LLM's")

# Sidebar for configuration
st.sidebar.header("Configuration")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

if openai_api_key:
    model = st.sidebar.selectbox("Select the model", ["gpt-3.5-turbo", "gpt-4o"], index=0)

    url = st.text_input("Coloque o site que você quer fazer a consulta:")
    user_prompt = st.text_area("O que você quer encontrar:", height=100)

    if st.button("Scrap and Analyze", type="primary"):
        if not url or not user_prompt:
            st.error("Plase enter both a URL and a query.")
        else:
            try:
                graph_config = {"llm": {"model": model, "api_key": openai_api_key}}
                scraper = SmartScraperGraph(prompt = user_prompt, source = url, config = graph_config)

                with st.spinner("Scraping and analyzing..."):
                    result = scraper.run()

                st.success("Scraping and analysis completed sucessfully!")

                st.subheader("Analysis:")
                st.write(result.get("analysis", "No analysis available."))

                st.subheader("Parsed Data:")
                st.json(result.get('parsed_data', {}))

                st.subheader('HTML Content Preview:')
                st.code(result.get("html_content", "")[:500] + "...", language = "html")

            except Exception as e:
                logger.exception("An error occurred during scraping or analysis:")
                st.error(f"An error occurred: {str(e)}")
else:
    st.warning("Insira sua chave de API OpenAI na barra lateral para começar.")

st.sidebar.markdown("---")
st.sidebar.caption("Extrator de Conteúdo e Documentação IA")