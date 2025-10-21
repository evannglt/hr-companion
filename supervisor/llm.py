"""LLM configuration for the supervisor system."""

from langchain_ollama import ChatOllama

# Using ChatOllama for better structured output support
llm = ChatOllama(model="deepseek-r1:1.5b")

