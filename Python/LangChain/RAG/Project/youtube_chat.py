from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace,HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

video_id = "fsLh-NYhOoU"

try:
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)

    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    print(transcript)

except TranscriptsDisabled:
    print("No Captions Available for this video")