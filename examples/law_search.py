"""
HNSWDB Example - Law Document Search

This example shows how to use HNSWDB for storing and searching
embeddings from law documents (or any text documents).
"""

import hnswdb
import random

# Simulate embedding generation (in real use, use OpenAI, HuggingFace, etc.)
def mock_embeddings(texts, dim=384):
    """Generate random embeddings for demo purposes"""
    return [[random.random() for _ in range(dim)] for _ in texts]


def main():
    print("=" * 60)
    print("HNSWDB - Law Document Search Example")
    print("=" * 60)

    # Sample law documents
    law_documents = [
        "The right to freedom of speech shall not be infringed.",
        "All persons are equal before the law.",
        "No person shall be deprived of life without due process.",
        "The right to privacy is protected under the constitution.",
        "Freedom of religion is guaranteed to all citizens.",
        "The right to bear arms shall not be infringed.",
        "No cruel or unusual punishment shall be inflicted.",
        "The right to a fair trial is guaranteed.",
        "Property rights are protected under the law.",
        "The freedom of the press shall not be abridged.",
    ]

    print(f"\n1. Creating embeddings for {len(law_documents)} law documents...")
    embeddings = mock_embeddings(law_documents, dim=384)
    print(f"   Embedding dimension: {len(embeddings[0])}")

    # Create and store in database
    print("\n2. Creating HNSW database and storing embeddings...")
    db = hnswdb.HNSWDatabase()
    db.start()
    db.store(embeddings)
    print(f"   Stored {db.count} vectors")
    print(f"   Dimension: {db.dimension}")

    # Search for similar documents
    print("\n3. Searching for documents similar to 'freedom of expression'...")
    query_text = "freedom of expression"
    query_embedding = mock_embeddings([query_text], dim=384)[0]

    # Using find_ids to get indices and distances
    results = db.find_ids(query_embedding, k=3)
    print(f"   Found {len(results)} nearest neighbors:")
    for idx, distance in results:
        print(f"   - Doc {idx}: \"{law_documents[idx][:50]}...\" (distance: {distance:.4f})")

    # Save to disk
    print("\n4. Saving database to disk...")
    save_path = db.save("laws.hnsw")
    print(f"   Saved to: {save_path}")

    # Close and reopen
    print("\n5. Closing and reopening database...")
    db.close()

    db2 = hnswdb.HNSWDatabase()
    db2.start()
    db2.open(save_path)
    print(f"   Reopened with {db2.count} vectors")

    # Append new documents
    print("\n6. Appending new law document...")
    new_law = "The right to peaceful assembly is protected."
    new_embedding = mock_embeddings([new_law], dim=384)
    db2.append(new_embedding)
    print(f"   New count: {db2.count} vectors")

    # Search again
    print("\n7. Searching again with new document included...")
    results = db2.find_ids(query_embedding, k=3)
    print(f"   New search results:")
    # Note: Index 10 would be the new document
    all_docs = law_documents + [new_law]
    for idx, distance in results:
        doc = all_docs[idx] if idx < len(all_docs) else "Unknown"
        print(f"   - Doc {idx}: \"{doc[:50]}...\" (distance: {distance:.4f})")

    # Cleanup
    print("\n8. Cleaning up...")
    db2.clear()  # This deletes the saved file too
    print("   Database cleared")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
