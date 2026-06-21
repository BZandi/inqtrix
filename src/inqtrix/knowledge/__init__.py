"""Native knowledge retrieval (RAG) over user-provided documents.

First cut: a SYNCHRONOUS, in-process engine — documents are chunked
and embedded at upload time, retrieval is exact cosine search over an
in-memory store, and answering runs through the same algorithm
registry as web research. Postgres persistence, a hybrid Qdrant store,
reranking, and the ingestion worker are staged upgrades behind the
``VectorStore`` / ``KnowledgeStore`` ports; the algorithm and HTTP
surface do not change when they land.

Hard rule carried from day one: a collection's embedding model (and
its dimension) is fixed at creation. Mixing models inside one
collection is rejected loudly — never padded or truncated.
"""

from inqtrix.knowledge.algorithm import KnowledgeAlgorithm
from inqtrix.knowledge.stores.ports import KnowledgeProviderContext

__all__ = ["KnowledgeAlgorithm", "KnowledgeProviderContext"]
