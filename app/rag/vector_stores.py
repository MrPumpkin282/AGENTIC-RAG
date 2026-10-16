import uuid
from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from app.core.config import get_settings

settings = get_settings()


_client = None
_embeddings = None
_dimension = None
_vectorstore = None


def _namespace():
    return settings.qdrant_name_space or "default"


def _parents_name():
    return f"{settings.qdrant_index_name}_parents"


def _point_id(key):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{_namespace()}:{key}"))


def get_client():
    global _client
    if _client is None:
        if not settings.cluster_end_point or not settings.qdrant_api_key:
            raise RuntimeError("CLUSTER_END_POINT or QDRANT_API_KEY is missing")
        _client = QdrantClient(url=settings.cluster_end_point, api_key=settings.qdrant_api_key)
    return _client


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        if not settings.nvidia_api_key:
            raise RuntimeError("NVIDIA_API_KEY is missing")
        if not settings.nvidia_embedding_model:
            raise RuntimeError("Embedding model is not configured")
        _embeddings = NVIDIAEmbeddings(
            model=settings.nvidia_embedding_model,
            api_key=settings.nvidia_api_key,
            truncate="END",
        )
    return _embeddings


def get_embedding_dimension():
    global _dimension
    if _dimension is None:
        _dimension = len(get_embeddings().embed_query("dimension probe"))
    return _dimension


def ensure_collections():
    client = get_client()
    name = settings.qdrant_index_name
    if not name:
        raise RuntimeError("QDRANT_INDEX_NAME is missing")

    desired_dimension = get_embedding_dimension()

    if client.collection_exists(name):
        vectors = client.get_collection(name).config.params.vectors
        current_dimension = getattr(vectors, "size", None)
        if current_dimension is not None and current_dimension != desired_dimension:
            raise RuntimeError(
                f"Collection '{name}' has dimension {current_dimension} but the embedding "
                f"model produces {desired_dimension}. Use a new QDRANT_INDEX_NAME or delete "
                "the old collection manually."
            )
    else:
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=desired_dimension, distance=models.Distance.COSINE),
        )
        for field in ("metadata.file_name", "metadata.parent_id", "metadata.namespace"):
            client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    if not client.collection_exists(_parents_name()):
        client.create_collection(
            collection_name=_parents_name(),
            vectors_config=models.VectorParams(size=1, distance=models.Distance.DOT),
        )


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        ensure_collections()
        _vectorstore = QdrantVectorStore(
            client=get_client(),
            collection_name=settings.qdrant_index_name,
            embedding=get_embeddings(),
        )
    return _vectorstore


def _namespace_filter():
    return models.Filter(
        must=[models.FieldCondition(key="metadata.namespace", match=models.MatchValue(value=_namespace()))]
    )


def get_parents(parent_ids):
    if not parent_ids:
        return []
    points = get_client().retrieve(
        collection_name=_parents_name(),
        ids=[_point_id(pid) for pid in parent_ids],
        with_payload=True,
    )
    by_id = {
        p.payload["metadata"]["parent_id"]: Document(
            page_content=p.payload["page_content"], metadata=p.payload["metadata"]
        )
        for p in points
    }
    return [by_id[pid] for pid in parent_ids if pid in by_id]


def add_documents(children, parents):
    store = get_vectorstore()
    client = get_client()

    for doc in parents.values():
        doc.metadata["namespace"] = _namespace()

    parent_points = [
        models.PointStruct(
            id=_point_id(pid),
            vector=[0.0],
            payload={"page_content": doc.page_content, "metadata": doc.metadata},
        )
        for pid, doc in parents.items()
    ]
    for i in range(0, len(parent_points), 100):
        client.upsert(collection_name=_parents_name(), points=parent_points[i:i + 100])

    for c in children:
        c.metadata["namespace"] = _namespace()
    ids = [_point_id(c.metadata["child_id"]) for c in children]
    return store.add_documents(children, ids=ids)


    