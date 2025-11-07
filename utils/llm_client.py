"""LLM client wrapper for LangChain and Ollama integration."""

from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from config.settings import Settings
from typing import Optional


class LLMClient:
    """
    Wrapper for LangChain + Ollama LLM interactions.
    """
    
    def __init__(self, model_name: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the LLM client.
        
        Args:
            model_name: Ollama model name (defaults to Settings.OLLAMA_MODEL)
            base_url: Ollama base URL (defaults to Settings.OLLAMA_BASE_URL)
        """
        self.model_name = model_name or Settings.get_model_name()
        self.base_url = base_url or Settings.get_ollama_url()
        
        # Initialize ChatOllama for chat-based interactions
        self.chat_model = ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=Settings.TEMPERATURE
        )
        
        # Initialize regular Ollama for simple completions
        self.llm = Ollama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=Settings.TEMPERATURE
        )
    
    def invoke(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Invoke the LLM with a prompt.
        
        Args:
            prompt: User prompt/question
            system_prompt: Optional system prompt for context
            
        Returns:
            LLM response string
        """
        messages = []
        
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        
        messages.append(HumanMessage(content=prompt))
        
        response = self.chat_model.invoke(messages)
        return response.content
    
    def simple_invoke(self, prompt: str) -> str:
        """
        Simple LLM invocation without chat formatting.
        
        Args:
            prompt: Prompt string
            
        Returns:
            LLM response string
        """
        return self.llm.invoke(prompt)

