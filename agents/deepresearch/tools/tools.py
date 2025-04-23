# agents/deepresearch/tools/tools.py
import os
import logging
import requests
from newspaper import Article
from google.adk.tools import FunctionTool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_web_content_func(url: str) -> str:
    logger.info(f"Attempting to extract content from URL: {url}")
    try:
        article = Article(url)
        article.download()
        article.parse()
        if not article.text:
             logger.warning(f"No main text content found at {url} after parsing.")
             return f"Content Extracted (Title: {article.title}):\n\n[No main text content found]"
        max_length = 15000
        content = article.text[:max_length] + ('...' if len(article.text) > max_length else '')
        return f"Content Extracted (Title: {article.title}):\n\n{content}"
    except Exception as e:
        logger.error(f"Failed to extract content from {url}: {e}", exc_info=True)
        return f"Error: Could not extract content from {url}. Reason: {str(e)}"

extract_tool = FunctionTool(func=extract_web_content_func)

