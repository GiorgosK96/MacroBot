"""
RAG System for Macroeconomics Agent
Μεταφορά και οργάνωση του κώδικα από το Jupyter notebook
"""

import os
import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from typing import Dict, List, Optional

# Import το RAG prompt από prompts.py
from agent.prompts import generate_rag_response_prompt

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


class MacroeconomicsRAG:
    """
    RAG System για μακροοικονομικές ερωτήσεις
    """
    
    def __init__(self, data_path: str = "data", faiss_index_path: str = "faiss_index_books"):
        """
        Αρχικοποίηση του RAG system
        
        Args:
            data_path: Path στα δεδομένα
            faiss_index_path: Path για το FAISS index
        """
        load_dotenv("agent.env")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.data_path = data_path
        self.faiss_index_path = faiss_index_path
        
        # Αρχικοποίηση LLM και Embeddings
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=self.openai_api_key)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=self.openai_api_key)
        
        # Φόρτωση ή δημιουργία του vectorstore
        self.vectorstore = self._load_or_create_vectorstore()
        
        # Δημιουργία του RAG chain
        self.qa_chain = self._create_qa_chain()
        
        self.book_display_names = {
                "Blanchard_Macro": "Blanchard & Johnson - Macroeconomics (6th Edition)",
                "Mankiw_Macro": "Mankiw - Macroeconomics (7th Edition)",
                "Mankiw_Principles_Econ": "Mankiw - Principles of Economics (7th Edition)", 
                "Krugman_Macro": "Krugman & Wells - Macroeconomics",
                "Dornbusch_Macro": "Dornbusch, Fischer & Startz - Macroeconomics (11th Edition)",
                "Mishkin_Money": "Mishkin & Serletis - The Economics of Money, Banking & Financial Markets",
                "OpenStax_Macro": "OpenStax - Principles of Macroeconomics"
            }

        logger.info("MacroeconomicsRAG initialized successfully")
    
    def _load_documents(self) -> List:
        """
        Φορτώνει όλα τα έγγραφα από τα αρχεία δεδομένων
        """
        try:
            # Βάσει του κώδικά σου - φόρτωση συγκεκριμένων αρχείων
            def load_with_metadata(path: str, source_name: str):
                loader = TextLoader(path, encoding="utf-8")
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = source_name
                return docs
            
            # Φόρτωση των βιβλίων (προσάρμοσε τα paths σύμφωνα με τα δικά σου)
            docs_b = load_with_metadata("data/Blanchard - Book.txt", "Blanchard_Macro")
            docs_m_old = load_with_metadata("data/Mankiw - Book.txt", "Mankiw_Macro")
            docs_m_principles = load_with_metadata("data/Mankiw - Principles of Economics.txt", "Mankiw_Principles_Econ")
            docs_krugman = load_with_metadata("data/Krugman.txt", "Krugman_Macro")
            docs_mishkin = load_with_metadata("data/Mishkin.txt", "Mishkin_Money")
            docs_openstax = load_with_metadata("data/OpenStax.txt", "OpenStax_Macro")
            docs_dornbusch = load_with_metadata("data/Dornbusch.txt", "Dornbusch_Macro")
            
            # Συνδυασμός όλων των documents
            all_docs = (docs_b + docs_m_old + docs_m_principles + 
                       docs_krugman + docs_mishkin + docs_openstax + docs_dornbusch)
            
            logger.info(f"Loaded {len(all_docs)} documents")
            return all_docs
            
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return []
    
    def _create_chunks(self, documents: List) -> List:
        """
        Χωρίζει τα έγγραφα σε chunks
        """
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200, 
                add_start_index=True
            )
            chunks = text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error creating chunks: {e}")
            return []
    
    def _load_or_create_vectorstore(self) -> Optional[FAISS]:
        """
        Φορτώνει υπάρχον FAISS index ή δημιουργεί νέο
        """
        try:
            # Έλεγχος αν υπάρχει υπάρχον index
            if os.path.exists(f"{self.faiss_index_path}/index.faiss"):
                logger.info("Loading existing FAISS index...")
                vectorstore = FAISS.load_local(
                    self.faiss_index_path, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                logger.info("FAISS vectorstore loaded successfully")
                return vectorstore
            else:
                logger.info("Creating new FAISS index...")
                # Φόρτωση documents
                documents = self._load_documents()
                if not documents:
                    logger.error("No documents loaded")
                    return None
                
                # Δημιουργία chunks
                chunks = self._create_chunks(documents)
                if not chunks:
                    logger.error("No chunks created")
                    return None
                
                # Δημιουργία vectorstore
                vectorstore = FAISS.from_documents(chunks, self.embeddings)
                
                # Αποθήκευση
                vectorstore.save_local(self.faiss_index_path)
                logger.info("New FAISS vectorstore created and saved")
                return vectorstore
                
        except Exception as e:
            logger.error(f"Error with vectorstore: {e}")
            return None
    
    def _create_qa_chain(self) -> Optional[RetrievalQA]:
        """
        Δημιουργεί το QA chain για retrieval
        """
        try:
            if not self.vectorstore:
                logger.error("No vectorstore available")
                return None
            
            # Δημιουργία retriever
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": 30}  
            )
            
            # Δημιουργία QA chain με το imported prompt
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": generate_rag_response_prompt}
            )
            
            logger.info("QA chain created successfully")
            return qa_chain
            
        except Exception as e:
            logger.error(f"Error creating QA chain: {e}")
            return None
    
    def query(self, question: str) -> Dict:
        """
        Εκτελεί ερώτημα στο RAG system
        
        Args:
            question: Η ερώτηση του χρήστη
            
        Returns:
            Dict με την απάντηση και metadata
        """
        try:
            if not self.qa_chain:
                return {
                    "result": "Σφάλμα: Το RAG σύστημα δεν είναι διαθέσιμο",
                    "source_documents": [],
                    "success": False
                }
            
            # Εκτέλεση του query
            response = self.qa_chain.invoke(question)
            
            # Επεξεργασία αποτελεσμάτων
            result = {
                "result": response["result"],
                "source_documents": response.get("source_documents", []),
                "success": True,
                "num_sources": len(response.get("source_documents", []))
            }
            
            logger.info(f"RAG query executed successfully. Found {result['num_sources']} source documents")
            return result
            
        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            return {
                "result": f"Σφάλμα κατά την εκτέλεση του ερωτήματος: {e}",
                "source_documents": [],
                "success": False
            }
    
    def get_sources_info(self, source_documents: List, top_n: int = 10) -> str:
        """
        Εμφανίζει τις καλύτερες πηγές που χρησιμοποιήθηκαν
        
        Args:
            source_documents: Όλα τα source documents
            top_n: Αριθμός top πηγών προς εμφάνιση
        """
        if not source_documents:
            return "Δεν βρέθηκαν πηγές!"
        
        # Κρατάμε μόνο τις top N πηγές (FAISS τις επιστρέφει ήδη sorted by relevance)
        top_sources = source_documents[:top_n]
        total_sources = len(source_documents)
        
        result = f"\n**ΕΜΦΑΝΙΖΟΝΤΑΙ:** {len(top_sources)} από {total_sources} πηγές (top {top_n})\n\n"
        
        # Grouping by source
        source_groups = {}
        for doc in top_sources:
            source = doc.metadata.get('source', 'Unknown')
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append(doc)
        
        # Handle singular/plural for books
        book_count = len(source_groups)
        book_text = "βιβλίο" if book_count == 1 else "βιβλία"
            
        result += f"**ΒΙΒΛΙΑ:** {book_count} {book_text}\n\n"
        
        # Display each source with its chunks
        for source_name, docs in source_groups.items():
            display_name = self.book_display_names.get(source_name, source_name)
            
            # Handle singular/plural for sources
            source_count = len(docs)
            source_text = "πηγή" if source_count == 1 else "πηγές"
            
            result += f"## **{display_name}** ({source_count} {source_text})\n\n"            
            for i, doc in enumerate(docs, 1):
                # Clean and format chunk content
                content = doc.page_content.strip()
                
                result += f"**Πηγή {i}:**\n"
                result += f"```\n{content}\n```\n\n"
        
        # Add note about remaining sources
        if total_sources > top_n:
            remaining = total_sources - top_n
            result += f"\n *Υπάρχουν ακόμη {remaining} επιπλέον πηγές που δεν εμφανίζονται*\n"
        
        return result
    
    def is_rag_result_good(self, rag_result: Dict, min_sources: int = 5) -> bool:
        """
        Ελέγχει αν το αποτέλεσμα του RAG είναι καλής ποιότητας
        
        Args:
            rag_result: Το αποτέλεσμα από το RAG
            min_sources: Ελάχιστος αριθμός πηγών
            
        Returns:
            True αν το αποτέλεσμα είναι καλό
        """
        if not rag_result.get("success", False):
            return False
        
        if rag_result.get("num_sources", 0) < min_sources:
            return False
        
        # Έλεγχος αν η απάντηση είναι πολύ σύντομη
        result_text = rag_result.get("result", "")
        if len(result_text.strip()) < 100:
            return False
        
        return True
