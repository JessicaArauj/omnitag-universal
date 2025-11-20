import openai
import csv
import json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


class RankingAnalyser:
    def __init__(
            self,
            gpt: openai.OpenAI,
            qdrant_url: str = "http://localhost:6333",
    ):
        self.gpt = gpt
        self.dataset = []
        self.qdrant = QdrantClient(
            url=qdrant_url,
            prefer_grpc=True,
        )

    def load_dataset(self, file_path: str):
        with open(file_path, "r", newline="", encoding="latin1") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) >= 4:
                    try:
                        rating = float(row[2])
                    except ValueError:
                        continue
                    self.dataset.append(
                        {"text": row[3], "rating": rating})

    def transform_and_store_embeddings(
            self,
            collection_name: str,
            model="text-embedding-3-large"
    ):
        vector_count = len(self.dataset)
        if not vector_count:
            raise RuntimeError("empty dataset")

        embedding = (self.gpt
                     .embeddings
                     .create(model=model, input=self._get_texts()))

        self.qdrant.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=len(embedding.data[0].embedding),
                distance=Distance.COSINE,
            ),
        )

        vectors = [
            PointStruct(id=i, vector=embedding.data[i].embedding,
                        payload=self.dataset[i])
            for i in range(vector_count)
        ]
        self.qdrant.upload_points(
            collection_name=collection_name,
            points=vectors,
        )

    def _get_texts(self) -> list[str]:
        return [
            f"rating = {entry['rating']:1.1f}\ntext = {entry['text']}"
            for entry in self.dataset
        ]

    def classify(
            self,
            collection_name: str,
            query: str,
            top_k: int = 20,
            model: str = "gpt-4o-mini",
    ) -> str:
        embedding = self.gpt.embeddings.create(
            model=model,
            input=query
        )

        results = self.qdrant.query_points(
            collection_name=collection_name,
            query=embedding.data[0].embedding,
            limit=top_k,
            with_payload=True,
        )

        context = "\n".join(
            f"Answer: {point.payload['text']}\n"
            f"Rating: {float(point.payload['rating']):1.1f}"
            for point in results.points
        )

        prompt = (
            "Based on the context of ranking question/answer pairs, "
            "provide a response to the following:\n\n"
            f"User's question: {query}\n\n"
            f"Similar context answers and ratings:\n{context}\n\n"
            "Please provide a reasoned response that also includes a rating "
            "with one decimal place (like 5.0 or 6.6 or 8.0) and "
            "it must contain not only a boolean but also a reasoned answer."
        )
        response = self.gpt.responses.create(
            model=model,
            input=prompt,
        )
        return json.loads(response.output[0].content[text].value)
