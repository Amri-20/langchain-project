from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader

loader=DirectoryLoader(
    path='BOOKS',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs=loader.lazy_load()

for documents in docs:
    print(documents.metadata)