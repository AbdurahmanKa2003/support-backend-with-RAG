from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional

# =========================
# 1. Настройка FastAPI
# =========================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # позже можешь ограничить доменами Nutralux
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# 2. OpenAI клиент и модели
# =========================

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# =========================
# 3. Pydantic-модель запроса
# =========================

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict]] = None  # [{role, content}, ...]


# =========================
# 4. Каталог продуктов и эмбеддинги
# =========================

PRODUCTS: List[Dict] = []
PRODUCT_EMBEDDINGS: Optional[np.ndarray] = None  # shape (N, dim) или None


def build_product_text(p: Dict) -> str:
    """
    Текст, из которого делаем embedding для продукта Nutralux.
    Адаптировано под твою структуру JSON (slug, name, short, description, facts[]).
    """
    facts_list = p.get("facts", [])
    facts_text = "; ".join(
        f"{f.get('name', '')}: {f.get('value', '')}"
        for f in facts_list
    )

    parts = [
        f"Slug: {p.get('slug', '')}",
        f"Name: {p.get('name', '')}",
        f"Category: {p.get('category', '')}",
        f"Short: {p.get('short', '')}",
        f"Description: {p.get('description', '')}",
        f"Facts: {facts_text}",
    ]
    return "\n".join(parts)


def load_products_and_embeddings() -> None:
    """
    Загружает products.json и строит эмбеддинги для всех продуктов.
    """
    global PRODUCTS, PRODUCT_EMBEDDINGS

    file_path = os.path.join(os.path.dirname(__file__), "products.json")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            PRODUCTS = json.load(f)
    except FileNotFoundError:
        print("⚠️ products.json not found. RAG will be disabled.")
        PRODUCTS = []
        PRODUCT_EMBEDDINGS = None
        return

    if not PRODUCTS:
        print("⚠️ products.json is empty. RAG will be disabled.")
        PRODUCT_EMBEDDINGS = None
        return

    texts = [build_product_text(p) for p in PRODUCTS]

    print(f"🔄 Creating embeddings for {len(texts)} products...")
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    embeddings = [np.array(item.embedding, dtype="float32") for item in response.data]
    PRODUCT_EMBEDDINGS = np.vstack(embeddings)
    print("✅ Product embeddings ready.")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def search_products(query: str, top_k: int = 3) -> List[Tuple[Dict, float]]:
    """
    Ищет top_k самых похожих продуктов по тексту запроса.
    Возвращает список (product_dict, similarity_score).
    """
    if PRODUCT_EMBEDDINGS is None or PRODUCT_EMBEDDINGS.shape[0] == 0:
        return []

    # embedding для запроса
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query],
    )
    q_emb = np.array(resp.data[0].embedding, dtype="float32")

    sims: List[Tuple[Dict, float]] = []
    for idx, p in enumerate(PRODUCTS):
        sim = cosine_similarity(q_emb, PRODUCT_EMBEDDINGS[idx])
        sims.append((p, sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_k]


def format_products_context(products_with_scores: List[Tuple[Dict, float]]) -> str:
    """
    Формирует контекст каталога для prompt'a.
    """
    lines: List[str] = []

    for p, score in products_with_scores:
        facts_lines: List[str] = []
        for f in p.get("facts", []):
            facts_lines.append(f"    • {f.get('name', '')}: {f.get('value', '')}")

        facts_block = "\n".join(facts_lines) if facts_lines else "    • (no extra facts)"

        lines.append(
            f"- Slug: {p.get('slug')}\n"
            f"  Name: {p.get('name')}\n"
            f"  Category: {p.get('category')}\n"
            f"  Short: {p.get('short')}\n"
            f"  Description: {p.get('description')}\n"
            f"  Facts:\n{facts_block}\n"
        )

    return "\n".join(lines)


# =========================
# 5. Инициализация при старте
# =========================

@app.on_event("startup")
def startup_event():
    if client is None:
        print("❌ OPENAI_API_KEY is not set. API will not work.")
    else:
        load_products_and_embeddings()


# =========================
# 6. Эндпоинты
# =========================

@app.get("/")
def home():
    return {"message": "Nutralux Chat Bot API with RAG is running"}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if client is None:
        return {"answer": "Server configuration error: OPENAI_API_KEY is missing."}

    # 1) Поиск релевантных продуктов (RAG)
    try:
        retrieved = search_products(req.message, top_k=3)
        context_text = format_products_context(retrieved) if retrieved else ""
    except Exception as e:
        print("RAG error:", e)
        retrieved = []
        context_text = ""

    # 2) System prompt + контекст
    base_system_prompt = (
        "Ты — дружелюбный консультант интернет-магазина Nutralux.\n"
        "Отвечай только на основе каталога Nutralux, который получаешь в контексте.\n"
        "Если нужной информации в каталоге нет, честно говори, что не уверен, "
        "и предложи обратиться к врачу или в официальную поддержку.\n"
        "Не придумывай дозировки, противопоказания или медицинские рекомендации, "
        "если они не указаны явно. Отвечай кратко, конкретно и понятным языком."
    )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": base_system_prompt}
    ]

    if context_text:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Вот несколько наиболее подходящих продуктов Nutralux "
                    "из каталога (используй только эти данные в ответе):\n\n"
                    f"{context_text}"
                ),
            }
        )

    # История диалога (если есть)
    if req.history:
        messages.extend(req.history)

    # Текущее сообщение пользователя
    messages.append({"role": "user", "content": req.message})

    # 3) Вызов чат-модели
    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
    )

    answer = completion.choices[0].message.content
    return {"answer": answer}