from typing import List, Tuple, Optional
from src.config import settings
from src.models.query import SearchResultItem


class LLMService:
    """Manages grounded response synthesis from retrieved document context."""

    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        self.model = settings.LLM_MODEL
        self.api_key = settings.OPENAI_API_KEY

    def synthesize_answer(
        self, 
        question: str, 
        contexts: List[SearchResultItem],
        custom_api_key: Optional[str] = None,
        custom_provider: Optional[str] = None,
        custom_system_prompt: Optional[str] = None
    ) -> Tuple[str, int, int]:
        """Synthesize a factual, grounded answer using retrieved contexts, or a friendly fallback.
        Returns: (answer_text, prompt_tokens, completion_tokens)
        """

        if self.provider == "mock" or not self.api_key:
            # Deterministic, grounded mock answer for offline development and tests
            titles = list(dict.fromkeys([c.title for c in contexts]))
            top_snippet = contexts[0].snippet[:180] + "..." if len(contexts[0].snippet) > 180 else contexts[0].snippet
            return (
                f"Berdasarkan dokumen resmi yang tersedia ({', '.join(titles)}): "
                f"{top_snippet}",
                0,
                0
            )

        # Real LLM call via OpenAI or Gemini (OpenAI compatibility)
        try:
            import random
            from openai import OpenAI
            
            # Rotate API keys randomly per request to avoid free-tier rate limits
            # Override with custom API key/provider if provided (BYOK)
            active_key = custom_api_key if custom_api_key else (random.choice(settings.api_keys_list) if settings.api_keys_list else self.api_key)
            active_provider = custom_provider if custom_provider else self.provider
            
            kwargs = {"api_key": active_key}
            if active_provider == "gemini":
                kwargs["base_url"] = "https://generativelanguage.googleapis.com/v1beta/openai/"
                
            client = OpenAI(**kwargs)
            
            if not contexts:
                # Conversational Fallback Prompt (Extremely low token cost)
                system_prompt = custom_system_prompt if custom_system_prompt else (
                    "Anda adalah Asisten Lodexi AI yang sangat ramah. Pengguna memberikan sapaan atau "
                    "pertanyaan di luar konteks dokumen. "
                    "Balas sapaan mereka dengan hangat, dan ingatkan mereka dengan sopan bahwa Anda adalah asisten "
                    "yang dikhususkan untuk menjawab pertanyaan seputar dokumen yang telah mereka unggah."
                )
                user_prompt = f"Pesan Pengguna:\n{question}\n\nBalasan Ramah:"
            else:
                # Heavy RAG Prompt
                context_block = "\n\n".join(
                    [f"[{i+1}] Dokumen: {c.title} (ID: {c.external_id})\n{c.snippet}" for i, c in enumerate(contexts)]
                )
                base_system_prompt = custom_system_prompt if custom_system_prompt else "Anda adalah Asisten Auditor Dokumen Resmi (Lodexi AI). Tugas Anda HANYA menjawab pertanyaan berdasarkan Konteks Dokumen yang diberikan."
                
                system_prompt = (
                    f"{base_system_prompt}\n\n"
                    "ATURAN SUPER KETAT:\n"
                    "1. Anda DILARANG KERAS menggunakan pengetahuan umum Anda sendiri. Jawaban HANYA boleh berasal dari teks yang ada di Konteks.\n"
                    "2. Jika jawaban tidak ada di dalam Konteks, Anda WAJIB menjawab: 'Maaf, informasi tersebut tidak ditemukan di dalam dokumen yang Anda unggah.'\n"
                    "3. Anda WAJIB menyertakan kutipan sumber (sitasi) untuk setiap fakta yang Anda sebutkan dengan format [Nama Dokumen].\n"
                    "4. Jangan berhalusinasi atau menebak-nebak."
                )
                user_prompt = f"KONTEKS DOKUMEN:\n{context_block}\n\nPERTANYAAN USER:\n{question}\n\nJAWABAN TERPERINCI DAN BERSITASI:"
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
            )
            
            answer_text = response.choices[0].message.content or "Tidak dapat menghasilkan jawaban."
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0
            
            return answer_text, prompt_tokens, completion_tokens
        except Exception as e:
            return f"Error saat menghubungi LLM ({str(e)}).", 0, 0


llm_service = LLMService()
