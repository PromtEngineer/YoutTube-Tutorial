# ── requirements ─────────────────────────────────────────────────────────
# pip install crawl4ai openai pydantic python-dotenv
# playwright install

import os, json, asyncio
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CacheMode,
    LLMConfig
)

from crawl4ai.extraction_strategy import LLMExtractionStrategy



# ── 1. load keys ─────────────────────────────────────────────────────────
load_dotenv()                                    # puts keys in env vars
URL_TO_SCRAPE = "https://web.lmarena.ai/leaderboard/"

# ── 2. declare a schema that matches the *instruction* ───────────────────
class Model(BaseModel):
    rank: int = Field(..., description="Position in the list")
    model_name: str
    score: float
    ci: float
    votes: int
    org: str
    license: str

INSTRUCTION_TO_LLM = """
You are given a LLM leaderboard page. 
Return an array of model objects (rank, model, score, 95% CI, Votes, Org, License). 
For model name, extract the complete name e.g. Claude 3.7 Sonnet (20250219) instead of just "Anthropic"
Return **only** valid JSON matching the schema – no markdown.
"""


# ── 3. DeepSeek is OpenAI-compatible, so pass base_url + model name ──────
llm_cfg = LLMConfig(
    provider="gemini/gemini-2.5-flash-preview-05-20",          # ✅ include model in the provider string
    api_token=os.getenv('GEMINI_API_KEY'),
    # base_url="https://api.deepseek.com/v1"
)

# ── 4. attach the extraction strategy ────────────────────────────────────
llm_strategy = LLMExtractionStrategy(
    llm_config=llm_cfg,
    schema=Model.model_json_schema(),      
    extraction_type="schema",
    instruction=INSTRUCTION_TO_LLM,
    chunk_token_threshold=1000,
    apply_chunking=True, overlap_rate=0.0,
    input_format="markdown",
)

crawl_cfg = CrawlerRunConfig(
    extraction_strategy=llm_strategy,
    cache_mode=CacheMode.DISABLED,
    remove_overlay_elements=True,
    exclude_external_links=True,
)

browser_cfg = BrowserConfig(headless=True, verbose=True, text_mode=True)

# ── 5. run the crawl ─────────────────────────────────────────────────────
async def main():
    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        result = await crawler.arun(URL_TO_SCRAPE, config=crawl_cfg)

        if result.success:
            data = json.loads(result.extracted_content)
            print("✅ extracted", len(data), "items")
            for p in data[:10]: print(p)
        else:
            print("❌ error:", result.error_message)
            print(llm_strategy.show_usage())   # token cost insight


if __name__ == "__main__":
    asyncio.run(main())
