from processing import extract_events, embedding, create_database

class PreProcessing:
    """Permet de lancer chaque étape du pré-processing séparément"""

    def extract_events(self):
        """Télécharge et trie les événements."""
        return extract_events.main()

    def generate_embeddings(self):
        """Génère et sauvegarde les embeddings."""
        return embedding.main()

    def create_database(self):
        """Construit et sauvegarde la base de données FAISS"""
        return create_database.main()