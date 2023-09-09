import dataclasses
import os

import openai
import pandas as pd
from llama_hub.youtube_transcript import YoutubeTranscriptReader
from llama_index import ServiceContext, VectorStoreIndex, set_global_service_context
from llama_index.indices.prompt_helper import PromptHelper
from llama_index.llms import OpenAI
from tqdm.auto import tqdm
from youtube_transcript_api._errors import TranscriptsDisabled

openai.api_key = os.environ["OPENAI_API_KEY"]


@dataclasses.dataclass
class VideoSummary:
    title: str
    url: str
    summary: str


def main():
    llm = OpenAI(model="gpt-4", temperature=0, max_tokens=4096)
    prompt_helper = PromptHelper(
        4000,  # max_tokens
        1024,  # input_tokens
        0,  # chunk_overlap_ratio
        chunk_size_limit=2000,
    )
    service_context = ServiceContext.from_defaults(llm=llm, prompt_helper=prompt_helper)
    set_global_service_context(service_context)

    loader = YoutubeTranscriptReader()

    video_df = pd.read_csv("data/remopoke_videos.csv")
    video_list = video_df["url"].tolist()
    summary_list: list[VideoSummary] = []

    for video in tqdm(video_list):
        try:
            documents = loader.load_data(ytlinks=[video], languages=["ja"])
        except TranscriptsDisabled:
            continue

        index = VectorStoreIndex.from_documents(documents=documents)
        query_engine = index.as_query_engine(response_mode="tree_summarize")
        response = query_engine.query("この動画の内容をできるだけ長く要約してほしいです。")
        summary = response.response
        summary_list.append(
            VideoSummary(
                title=video_df[video_df["url"] == video]["title"].values[0],
                url=video,
                summary=summary,
            )
        )

    video_df2 = pd.DataFrame(summary_list)
    video_df2.to_csv("output/remopoke_videos_summary.csv", index=False)


if __name__ == "__main__":
    main()
