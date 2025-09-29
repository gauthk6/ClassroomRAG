import json
from pathlib import Path
from typing import List, Dict, Tuple
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

class ImageRAG:
    def __init__(self, jsonl_path: str, project_dir: str = "./image_rag"):
        """al
        Initialize the RAG pipeline for image retrieval
        
        Args:
            jsonl_path: Path to the image descriptions JSONL file
            project_dir: Directory to store all RAG-related files
        """
        self.jsonl_path = jsonl_path
        
        # Set up project directory structure
        self.project_dir = Path(project_dir)
        self.vector_db_path = self.project_dir / "vector_db"
        
        # Create directories
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.image_data = self._load_jsonl()
        self.vector_store = None

    def _load_jsonl(self) -> List[Dict]:
        """Load and parse the JSONL file"""
        data = []
        with open(self.jsonl_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def create_vector_store(self):
        """Create a vector store from image descriptions"""
        descriptions = [item["description"] for item in self.image_data]
        img_paths = [item["img_path"] for item in self.image_data]
        
        self.vector_store = Chroma.from_texts(
            texts=descriptions,
            embedding=self.embeddings,
            persist_directory=str(self.vector_db_path),
            metadatas=[{"img_path": path} for path in img_paths]
        )
        self.vector_store.persist()

    def load_vector_store(self):
        """Load an existing vector store"""
        self.vector_store = Chroma(
            persist_directory=str(self.vector_db_path),
            embedding_function=self.embeddings
        )

    def retrieve_images(self, query: str, k: int = 3) -> List[Tuple[str, str, float]]:
        """
        Retrieve the top k most relevant images for a given query
        
        Args:
            query: The question or description to search for
            k: Number of images to retrieve
            
        Returns:
            List of tuples containing (image_path, description, similarity_score)
        """
        # Ensure the vector store is initialized
        if self.vector_store is None:
            if self.vector_db_path.exists() and any(self.vector_db_path.iterdir()):
                self.load_vector_store()
            else:
                self.create_vector_store()

        # Perform the similarity search
        results = self.vector_store.similarity_search_with_relevance_scores(query, k=k)

        return [(doc.metadata.get("img_path", "N/A"), 
                doc.page_content or "No description available", 
                score) for doc, score in results]

def main():
    # Example usage
    rag = ImageRAG("image_descriptions.jsonl")
    
    # Example query
    query = "In Figure 4.5, how does the nucleoid region's structure reflect its function compared to a eukaryotic nucleus shown in Figure 4.8?"
    try:
        results = rag.retrieve_images(query)
        print("\nQuery:", query)
        print("\nTop 3 most relevant images:")
        if results:
            for img_path, description, score in results:
                print(f"\nImage: {img_path}")
                print(f"Description: {description}")
                print(f"Relevance Score: {score:.4f}")
        else:
            print("No results found.")
    except Exception as e:
        print(f"Error during retrieval: {e}")

if __name__ == "__main__":
    main()
