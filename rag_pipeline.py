import os
import json
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------
# 14. REDIS MEMORY MANAGER (5 TURNS WITH TTL)
# ---------------------------------------------------------
class RedisMemoryManager:
    """Manages multi-turn conversation history in Redis Cloud (Up to 5 turns, 30 min TTL)."""

    def __init__(self, session_id: str = "default_session"):
        self.redis_url = os.getenv("REDIS_URL")
        self.session_id = f"doc_chat:{session_id}"
        self.max_turns = int(os.getenv("MEMORY_TURNS", "5"))
        self.ttl_seconds = int(os.getenv("MEMORY_TTL_SECONDS", "1800"))
        self.client = None

        if self.redis_url:
            try:
                import redis
                self.client = redis.from_url(self.redis_url, socket_timeout=5, decode_responses=True)
                self.client.ping()
            except Exception as e:
                print(f"⚠️ Redis connection failed ({e}). Falling back to local in-memory store.")
                self.client = None
        
        # Local in-memory fallback if Redis is unavailable
        self._local_history: List[Dict[str, str]] = []

    def get_history(self) -> List[Dict[str, str]]:
        """Retrieves history up to max_turns."""
        if self.client:
            try:
                raw_data = self.client.lrange(self.session_id, 0, -1)
                history = [json.loads(item) for item in raw_data]
                return history
            except Exception as e:
                print(f"⚠️ Error reading from Redis: {e}")
                return self._local_history
        return self._local_history

    def add_turn(self, user_query: str, assistant_response: str):
        """Appends a conversation turn (user + assistant) and trims to max_turns."""
        turn = [
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": assistant_response}
        ]

        if self.client:
            try:
                for msg in turn:
                    self.client.rpush(self.session_id, json.dumps(msg))
                
                # Keep only last 2 * max_turns (user + assistant messages)
                self.client.ltrim(self.session_id, -2 * self.max_turns, -1)
                # Refresh TTL
                self.client.expire(self.session_id, self.ttl_seconds)
                return
            except Exception as e:
                print(f"⚠️ Error writing to Redis: {e}")

        # Local fallback
        self._local_history.extend(turn)
        self._local_history = self._local_history[-2 * self.max_turns:]

    def clear(self):
        """Clears memory for this session."""
        if self.client:
            try:
                self.client.delete(self.session_id)
            except Exception as e:
                pass
        self._local_history.clear()


# ---------------------------------------------------------
# 15. QUERY REWRITER (HISTORICAL CONTEXT RESOLUTION)
# ---------------------------------------------------------
class QueryRewriter:
    """Rewrites follow-up queries using LLM into standalone search queries based on chat history."""

    def __init__(self):
        self.api_key = os.getenv("LLM_API_KEY")
        self.base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    def rewrite(self, query: str, history: List[Dict[str, str]]) -> str:
        """Rewrites ambiguous or follow-up query into standalone search query."""
        if not history:
            return query

        if not self.api_key or self.api_key.startswith("sk-proj-placeholder"):
            print("ℹ️ No active LLM API key configured for Query Rewriter. Using raw query.")
            return query

        try:
            import httpx
            
            history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history[-6:]])
            prompt = (
                "Given the following conversation history and a follow-up question, rewrite the follow-up question "
                "to be a completely standalone search query that preserves all context and entity names. "
                "Do NOT answer the question, only output the rewritten standalone query in English.\n\n"
                f"Conversation History:\n{history_text}\n\n"
                f"Follow-up Question: {query}\n\n"
                "Standalone Search Query:"
            )

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a precise search query reformulation assistant."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0
            }

            res = httpx.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=8.0)
            if res.status_code == 200:
                data = res.json()
                rewritten = data["choices"][0]["message"]["content"].strip().strip('"')
                print(f"🔄 Query Rewritten: '{query}' -> '{rewritten}'")
                return rewritten
            else:
                print(f"⚠️ LLM API status {res.status_code}. Using raw query.")
                return query

        except Exception as e:
            print(f"⚠️ Query rewriter exception ({e}). Using raw query.")
            return query


# ---------------------------------------------------------
# 16. CONTEXT BUILDER (LLM PROMPT CONTEXT ASSEMBLER)
# ---------------------------------------------------------
class ContextBuilder:
    """Assembles reranked top-k document chunks into structured LLM context prompts with source citations."""

    def __init__(self, system_instruction: Optional[str] = None):
        self.system_instruction = system_instruction or (
            "You are an intelligent, accurate, and professional AI Document Assistant.\n\n"
            "Your goal is to provide clear, direct, and factually grounded answers based on the uploaded document(s).\n\n"
            "## ACCURACY & GROUNDING\n\n"
            "- Use the retrieved context as your primary source of truth.\n"
            "- Never fabricate, invent, or guess details not present in the documents.\n"
            "- If the document does not contain enough information to answer a question, politely say:\n"
            '  "I don\'t have enough information in the provided document to answer that accurately."\n\n'
            "## ENTITIES & COMPANY NAMES\n\n"
            "- ALWAYS explicitly mention the actual company name, organization, person, or product mentioned in the document (e.g. use the real entity name rather than generic phrases like 'the company' or 'the document').\n"
            "- When asked 'tell me about your company' or 'what does this company do', explain who the organization described in the document is and what they offer.\n\n"
            "## SUMMARIES VS KEY POINTS\n\n"
            "- **When asked to summarize**: Provide a well-structured overview explaining what the document is, the organization/subject behind it, its primary objective, and main offerings.\n"
            "- **When asked for key points / highlights**: Provide crisp, distinct bullet points emphasizing specific facts, metrics, policies, and key takeaways.\n\n"
            "## CITATION & CLEAN FORMATTING\n\n"
            "- Present answers cleanly in natural markdown.\n"
            "- Do NOT append artificial tags like '[Source: filename.pdf, Page 1]' inside the body paragraphs.\n"
            "- Format all email addresses and website URLs as clickable markdown links (e.g. [support@example.com](mailto:support@example.com) or [Website](https://example.com)).\n"
            "- NEVER add robotic conversational fillers or trailing offers at the end (such as 'Let me know if you need anything else' or 'Would you like more details?').\n\n"
            "## LANGUAGE\n\n"
            "- Always communicate in clear, fluent English by default.\n"
            "- Do NOT switch to Urdu or other languages unless the user explicitly asks you to respond in that specific language."
        )

    def build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Formats top retrieved chunks into a clean, cited markdown context block."""
        if not chunks:
            return "No relevant context found."

        context_blocks = []
        for idx, chunk in enumerate(chunks, 1):
            source = chunk.get("source", "Unknown Source")
            page = chunk.get("page", 1)
            content = chunk.get("content", "").strip()
            
            block = f"--- [DOCUMENT CHUNK {idx} | Source: {source}, Page {page}] ---\n{content}"
            context_blocks.append(block)

        return "\n\n".join(context_blocks)

    def build_prompt_payload(self, query: str, chunks: List[Dict[str, Any]], history: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
        """Constructs full message list (system + history + context payload + query) for OpenAI compatible LLM API."""
        context_str = self.build_context(chunks)

        if history:
            history_lines = []
            for turn in history[-10:]:
                role_label = "User" if turn.get("role") == "user" else "Assistant"
                history_lines.append(f"{role_label}: {turn.get('content', '')}")
            history_str = "\n".join(history_lines)
        else:
            history_str = "None"

        if not chunks:
            user_content = (
                "## RETRIEVED CONTEXT\n\nNo relevant context found in the knowledge base.\n\n"
                f"## PREVIOUS CONVERSATION\n\n{history_str}\n\n"
                f"## CURRENT QUESTION\n\n{query}\n\n"
                "## CRITICAL MANDATORY RULE\n\n"
                "No document context was retrieved from the knowledge base for this query. "
                "Do NOT use previous conversation history or general assumptions to fabricate information about deleted or missing documents. "
                "If the user is asking about document content, specific files, or facts from documents, respond ONLY with:\n"
                '"I don\'t have enough information in the provided document to answer that accurately."'
            )
        else:
            user_content = (
                f"## RETRIEVED CONTEXT\n\n{context_str}\n\n"
                f"## PREVIOUS CONVERSATION\n\n{history_str}\n\n"
                f"## CURRENT QUESTION\n\n{query}\n\n"
                "## FINAL RULE\n\n"
                "Answer the user's current question naturally and helpfully using the retrieved context.\n"
                "If the answer is not supported by the retrieved context, do not guess."
            )

        messages = [
            {"role": "system", "content": self.system_instruction},
            {"role": "user", "content": user_content}
        ]
        return messages


# ---------------------------------------------------------
# 17. LLM RESPONSE GENERATOR
# ---------------------------------------------------------
class LLMGenerator:
    """Generates answers using OpenAI-compatible LLM API from context payload."""

    def __init__(self):
        self.api_key = os.getenv("LLM_API_KEY")
        self.base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    def generate(self, messages: List[Dict[str, str]]) -> str:
        """Sends payload to LLM API and returns the generated answer."""
        if not self.api_key or self.api_key.startswith("sk-proj-placeholder"):
            return "⚠️ Error: No valid LLM_API_KEY configured in .env file."

        try:
            import httpx
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.3
            }

            res = httpx.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=30.0)
            if res.status_code == 200:
                data = res.json()
                return data["choices"][0]["message"]["content"].strip()
            else:
                return f"⚠️ LLM API Error ({res.status_code}): {res.text}"
        except Exception as e:
            return f"⚠️ Exception generating response: {e}"
