"""Debug why structured output is returning empty from gpt-5-mini."""
import asyncio
from dotenv import load_dotenv
load_dotenv(override=True)
from trial_matcher.config import get_settings
get_settings.cache_clear()
from trial_matcher.llm.client import UnifiedLLM
from trial_matcher.llm.structured import structured_complete
from trial_matcher.models.search_plan import SearchPlan

async def main():
    llm = UnifiedLLM()
    
    # Test 1: Direct acomplete with a simple patient
    print("=== Test 1: Direct acomplete ===")
    r = await llm.acomplete(
        "Extract the primary diagnosis from this patient note: "
        "'45-year-old male with stage IIIA non-small cell lung cancer, ECOG 1, no prior treatment.' "
        "Return ONLY a JSON object with keys: primary_diagnosis, age, sex",
        model="mini",
        max_tokens=500,
        response_format={"type": "json_object"},
    )
    print(f"Response: {r!r}")
    print(f"Text length: {len(r)}")
    
    # Test 2: Without json_object format
    print("\n=== Test 2: Without json_object ===")
    r2 = await llm.acomplete(
        "Extract the primary diagnosis from this patient note: "
        "'45-year-old male with stage IIIA non-small cell lung cancer, ECOG 1, no prior treatment.' "
        "Return ONLY a JSON object with keys: primary_diagnosis, age, sex",
        model="mini",
        max_tokens=500,
    )
    print(f"Response: {r2!r}")
    
    # Test 3: achat with higher budget
    print("\n=== Test 3: achat with bigger budget ===")
    r3 = await llm.achat(
        [{"role": "user", "content": "Return a JSON: {\"primary_disease_query\": \"lung cancer\", \"expansion_terms\": [\"NSCLC\"]}"}],
        model="mini",
        max_tokens=2000,
        cache=False,
        response_format={"type": "json_object"},
    )
    print(f"Response text: {r3.text!r}")
    print(f"Tokens: prompt={r3.prompt_tokens}, completion={r3.completion_tokens}")

asyncio.run(main())
