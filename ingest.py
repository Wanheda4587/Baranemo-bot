from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

print("PDF yükleniyor...")
loader = PyPDFLoader("baranemo.pdf")
documents = loader.load()
print(f"Toplam sayfa: {len(documents)}")

print("Parçalara bölünüyor...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)
print(f"Toplam parça: {len(chunks)}")

print("Veritabanı oluşturuluyor...")
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma.from_documents(chunks, embeddings, persist_directory="./veritabani")
print("Tamamdı! Veritabanı oluşturuldu.")