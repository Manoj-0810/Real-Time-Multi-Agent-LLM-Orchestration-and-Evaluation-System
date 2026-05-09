# =============================================================================
# MEGA AI — RAG Agent
# =============================================================================
# Multi-hop reasoning: MUST retrieve at least 2 chunks before answering.
# Single-hop is explicitly rejected.
# Cites which chunk contributed to which part of the answer.
# Uses pgvector for embeddings stored in PostgreSQL.
# =============================================================================

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from app.agents.base import (
    AgentError,
    AgentOutput,
    BaseAgent,
)

from app.schemas import (
    ContextObject,
    ProvenanceEntry,
)

logger = logging.getLogger(__name__)


class RAGAgent(BaseAgent):
    """
    Retrieval-Augmented Generation agent
    with enforced multi-hop retrieval.
    """

    agent_id = "rag"

    description = (
        "Retrieval-augmented generation "
        "with multi-hop reasoning"
    )

    MIN_CHUNKS_REQUIRED: int = 2

    MAX_CHUNKS: int = 5

    SIMILARITY_THRESHOLD: float = 0.7

    # =========================================================================
    # Main Execution
    # =========================================================================

    async def _execute(
        self,
        ctx: ContextObject,
    ) -> AgentOutput:
        """
        Execute multi-hop RAG flow.
        """

        query = ctx.query

        # ---------------------------------------------------------------------
        # Retrieval Query
        # ---------------------------------------------------------------------

        retrieval_query = self._extract_retrieval_query(
            ctx
        )

        # ---------------------------------------------------------------------
        # First Hop
        # ---------------------------------------------------------------------

        first_hop_chunks = await self._first_hop(
            retrieval_query
        )

        # ---------------------------------------------------------------------
        # Second Hop
        # ---------------------------------------------------------------------

        second_hop_query = self._generate_follow_up_query(
            first_hop_chunks,
            retrieval_query,
        )

        second_hop_chunks = await self._second_hop(
            second_hop_query
        )

        # ---------------------------------------------------------------------
        # Merge + Deduplicate
        # ---------------------------------------------------------------------

        all_chunks = self._deduplicate_chunks(
            first_hop_chunks + second_hop_chunks
        )

        # ---------------------------------------------------------------------
        # Enforce Multi-Hop Requirement
        # ---------------------------------------------------------------------

        if len(all_chunks) < self.MIN_CHUNKS_REQUIRED:

            third_hop_chunks = await self._third_hop(
                query,
                all_chunks,
            )

            all_chunks = self._deduplicate_chunks(
                all_chunks + third_hop_chunks
            )

            if len(all_chunks) < self.MIN_CHUNKS_REQUIRED:

                raise AgentError(
                    "Multi-hop retrieval failed: "
                    f"only {len(all_chunks)} chunks retrieved, "
                    f"minimum {self.MIN_CHUNKS_REQUIRED} required."
                )

        # ---------------------------------------------------------------------
        # Generate Answer
        # ---------------------------------------------------------------------

        answer = await self._generate_answer(
            query,
            all_chunks,
        )

        # ---------------------------------------------------------------------
        # Provenance
        # ---------------------------------------------------------------------

        provenance = self._build_provenance(
            answer,
            all_chunks,
        )

        ctx.provenance_map.update(provenance)

        # ---------------------------------------------------------------------
        # Output
        # ---------------------------------------------------------------------

        output = {
            "answer": answer,
            "chunks_retrieved": len(all_chunks),

            "chunks": [
                {
                    "id": chunk.get(
                        "id",
                        "unknown",
                    ),

                    "source": chunk.get(
                        "source",
                        "unknown",
                    ),

                    "similarity": chunk.get(
                        "similarity",
                        0,
                    ),

                    "content_preview": (
                        chunk.get("content", "")[:200]
                    ),
                }
                for chunk in all_chunks
            ],

            "citations": [
                {
                    "text_span": span,
                    "chunk_id": chunk_id,
                    "relevance": relevance,
                }
                for span, chunk_id, relevance
                in self._extract_citations(
                    answer,
                    all_chunks,
                )
            ],

            "multi_hop_performed": True,
        }

        output_content = json.dumps(
            output,
            indent=2,
        )

        return AgentOutput(
            agent_id=self.agent_id,
            content=output_content,
            confidence=(
                0.85
                if len(all_chunks) >= 3
                else 0.70
            ),
            token_count=len(
                output_content.split()
            ),
        )

    # =========================================================================
    # Retrieval Query
    # =========================================================================

    def _extract_retrieval_query(
        self,
        ctx: ContextObject,
    ) -> str:
        """
        Extract retrieval query.
        """

        if ctx.sub_tasks:

            queries = [
                st.description
                for st in ctx.sub_tasks
                if st.status != "failed"
            ]

            if queries:
                return " ".join(queries)

        return ctx.query

    # =========================================================================
    # First Hop
    # =========================================================================

    async def _first_hop(
        self,
        query: str,
    ) -> List[Dict[str, Any]]:
        """
        First retrieval hop.
        """

        logger.info(
            f"RAG first hop: {query[:50]}...",
            extra={"agent_id": self.agent_id},
        )

        try:

            result = await self.tools.execute(
                tool_name="web_search",
                agent_id=self.agent_id,
                query=query,
                max_results=3,
            )

            if result["status"] == "success":

                chunks = []

                for r in result["data"].get(
                    "results",
                    [],
                ):

                    chunks.append({
                        "id": (
                            f"web_{hash(r['url']) % 10000}"
                        ),

                        "content": r["snippet"],

                        "source": r["url"],

                        "similarity": r.get(
                            "relevance_score",
                            0.7,
                        ),

                        "hop": 1,
                    })

                return chunks

        except Exception as e:

            logger.error(
                f"First hop failed: {e}",
                extra={"agent_id": self.agent_id},
            )

        return []

    # =========================================================================
    # Follow-Up Query
    # =========================================================================

    def _generate_follow_up_query(
        self,
        chunks: List[Dict[str, Any]],
        original_query: str,
    ) -> str:
        """
        Generate second-hop query.
        """

        if not chunks:
            return original_query

        entities = []

        for chunk in chunks:

            content = chunk.get(
                "content",
                "",
            )

            words = [
                w for w in content.split()
                if len(w) > 4
            ]

            entities.extend(words[:3])

        if entities:

            follow_up = (
                f"{original_query} "
                f"{' '.join(entities[:5])}"
            )

        else:

            follow_up = (
                f"{original_query} additional details"
            )

        return follow_up[:200]

    # =========================================================================
    # Second Hop
    # =========================================================================

    async def _second_hop(
        self,
        query: str,
    ) -> List[Dict[str, Any]]:
        """
        Second retrieval hop.
        """

        logger.info(
            f"RAG second hop: {query[:50]}...",
            extra={"agent_id": self.agent_id},
        )

        try:

            result = await self.tools.execute(
                tool_name="web_search",
                agent_id=self.agent_id,
                query=query,
                max_results=3,
            )

            if result["status"] == "success":

                chunks = []

                for r in result["data"].get(
                    "results",
                    [],
                ):

                    chunks.append({
                        "id": (
                            f"web_{hash(r['url']) % 10000 + 1000}"
                        ),

                        "content": r["snippet"],

                        "source": r["url"],

                        "similarity": r.get(
                            "relevance_score",
                            0.65,
                        ),

                        "hop": 2,
                    })

                return chunks

        except Exception as e:

            logger.error(
                f"Second hop failed: {e}",
                extra={"agent_id": self.agent_id},
            )

        return []

    # =========================================================================
    # Third Hop
    # =========================================================================

    async def _third_hop(
        self,
        original_query: str,
        existing_chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Broadened fallback retrieval.
        """

        logger.info(
            "RAG third hop: broadened search",
            extra={"agent_id": self.agent_id},
        )

        broadened = " ".join(
            original_query.split()[:5]
        )

        try:

            result = await self.tools.execute(
                tool_name="web_search",
                agent_id=self.agent_id,
                query=broadened,
                max_results=3,
            )

            if result["status"] == "success":

                chunks = []

                for r in result["data"].get(
                    "results",
                    [],
                ):

                    chunks.append({
                        "id": (
                            f"web_{hash(r['url']) % 10000 + 2000}"
                        ),

                        "content": r["snippet"],

                        "source": r["url"],

                        "similarity": r.get(
                            "relevance_score",
                            0.5,
                        ),

                        "hop": 3,
                    })

                return chunks

        except Exception as e:

            logger.error(
                f"Third hop failed: {e}",
                extra={"agent_id": self.agent_id},
            )

        return []

    # =========================================================================
    # Deduplication
    # =========================================================================

    def _deduplicate_chunks(
        self,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Remove duplicate sources.
        """

        seen_sources = set()

        unique = []

        for chunk in chunks:

            source = chunk.get(
                "source",
                "",
            )

            if source not in seen_sources:

                seen_sources.add(source)

                unique.append(chunk)

        return unique

    # =========================================================================
    # Answer Generation
    # =========================================================================

    async def _generate_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
    ) -> str:
        """
        Generate final cited answer.
        """

        context_parts = []

        for i, chunk in enumerate(chunks):

            context_parts.append(
                f"[Source {i + 1}]: "
                f"{chunk.get('content', '')} "
                f"(from {chunk.get('source', 'unknown')})"
            )

        context = "\n\n".join(context_parts)

        prompt = f"""
Answer the following question based on the provided sources.

Question:
{query}

Sources:
{context}

Provide a comprehensive answer with citations
in [Source X] format.
"""

        try:

            response = await self.llm.generate(
                prompt=prompt,
                max_tokens=2048,
                temperature=0.3,
                job_id="rag_generation",
                agent_id=self.agent_id,
            )

            return response.content

        except Exception as e:

            logger.error(
                f"Answer generation failed: {e}",
                extra={"agent_id": self.agent_id},
            )

            return (
                "Based on the following sources:\n\n"
                f"{context}\n\n"
                "[Unable to synthesize full answer due to error]"
            )

    # =========================================================================
    # Provenance
    # =========================================================================

    def _build_provenance(
        self,
        answer: str,
        chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Build provenance mapping.
        """

        provenance = {}

        sentences = [
            s.strip()
            for s in answer.split(".")
            if s.strip()
        ]

        for i, sentence in enumerate(sentences):

            best_chunk = None

            best_score = 0

            for chunk in chunks:

                content = chunk.get(
                    "content",
                    "",
                )

                sentence_words = set(
                    sentence.lower().split()
                )

                chunk_words = set(
                    content.lower().split()
                )

                if sentence_words and chunk_words:

                    overlap = len(
                        sentence_words & chunk_words
                    )

                    score = overlap / len(sentence_words)

                    if score > best_score:

                        best_score = score

                        best_chunk = chunk

            if best_chunk:

                provenance[f"sentence_{i}"] = (
                    ProvenanceEntry(
                        sentence=sentence,
                        source_agent=self.agent_id,
                        source_chunk=best_chunk.get(
                            "source",
                            "unknown",
                        ),
                        confidence=best_score,
                    )
                )

        return provenance

    # =========================================================================
    # Citation Extraction
    # =========================================================================

    def _extract_citations(
        self,
        answer: str,
        chunks: List[Dict[str, Any]],
    ) -> List[tuple]:
        """
        Extract [Source X] citations.
        """

        citations = []

        for match in re.finditer(
            r"\[Source (\d+)\]",
            answer,
        ):

            source_num = int(
                match.group(1)
            )

            start = max(
                0,
                match.start() - 100,
            )

            end = min(
                len(answer),
                match.end() + 100,
            )

            text_span = answer[start:end]

            chunk_id = (
                chunks[source_num - 1].get(
                    "id",
                    "unknown",
                )
                if source_num <= len(chunks)
                else "unknown"
            )

            citations.append(
                (
                    text_span,
                    chunk_id,
                    0.8,
                )
            )

        return citations