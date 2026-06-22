import httpx, os, json
from dotenv import load_dotenv

load_dotenv(override=True)

base = os.environ['AZURE_OPENAI_ENDPOINT'].rstrip('/')
key = os.environ['AZURE_OPENAI_API_KEY']

print(f"Conectando al endpoint: {base}\n")

names = ['gpt-5-mini', 'gpt-5.4', 'o4-mini']

# 1. CAMBIO: Actualizamos la versión de la API a una más reciente
api_version = "2024-12-01-preview"

for n in names:
    url = f"{base}/openai/deployments/{n}/chat/completions?api-version={api_version}"
    
    # 2. CAMBIO: Usamos 'max_completion_tokens' en lugar de 'max_tokens'
    payload = {
        "messages": [{"role": "user", "content": "Hola"}],
        "max_completion_tokens": 5
    }
    
    try:
        r = httpx.post(url, headers={"api-key": key, "Content-Type": "application/json"}, json=payload, timeout=10)
        
        if r.status_code == 200:
            status = "EXISTS & WORKING (200)"
        else:
            status = f"ERROR ({r.status_code}): {r.text}"
            
        print(f"  {n}: {status}")
    except Exception as e:
        print(f"  {n}: ERROR DE CONEXIÓN ({e})")