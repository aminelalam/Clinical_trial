from dotenv import load_dotenv
load_dotenv(override=True)
from trial_matcher.config import get_settings
get_settings.cache_clear()
import asyncio
from trial_matcher.llm.client import UnifiedLLM

async def test():
    llm = UnifiedLLM()
    # Test 1: json_object format
    r = await llm.achat(
        [{"role": "user", "content": "Return a JSON object with key name and value Alice."}],
        model="mini", max_tokens=200, cache=False,
        response_format={"type": "json_object"}
    )
    print("json_object mode:", repr(r.text[:300]))
    print("tokens:", r.prompt_tokens, r.completion_tokens)
    
    # Test 2: no response format, just ask for JSON
    r2 = await llm.achat(
        [{"role": "user", "content": "Return a JSON object with key name and value Bob. Output only the JSON."}],
        model="mini", max_tokens=200, cache=False,
    )
    print("plain mode:", repr(r2.text[:300]))
    print("tokens:", r2.prompt_tokens, r2.completion_tokens)

asyncio.run(test())
