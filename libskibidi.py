#!/usr/bin/env python3
import asyncio
import json
import logging
import os
import random
from dataclasses import dataclass
import string
import aiohttp
import numpy as np
import openai
import xml.etree.ElementTree as ET
from aiolimiter import AsyncLimiter
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio, pad_or_trim
from faster_whisper.tokenizer import Tokenizer
import nltk
import yt_dlp

nltk.download("punkt")

MODEL = "tts-1"
FORMAT = "flac"
MAX_INPUT_LENGTH = 4096
REQUESTS_PER_MIN = 3 if MODEL == "tts-1-hd" else 50
VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
DEFAULT_VIDEOS = [
    "qK1VjY_cU9w",
    "37_wnD3jUbw",
    "7yl7Wc1dtWc",
    "8KdXeDSGvZw",
    "_H2cLn-OlIU",
    "s600FYgI5-s",
]
YT_DLP_ARGV = ["-f", "bestvideo"]

logging.basicConfig(level=logging.DEBUG)


@dataclass
class Paper:
    id: str
    title: str
    abstract: str

    @classmethod
    def from_entry(cls, entry):
        return cls(
            id=entry.find("{http://www.w3.org/2005/Atom}id").text.split("/")[-1],
            title=entry.find("{http://www.w3.org/2005/Atom}title").text,
            abstract=entry.find("{http://www.w3.org/2005/Atom}summary").text,
        )


@dataclass
class SourceVideo:
    path: str
    duration: float
    width: int
    height: int
    format: str
    codec: str

    @classmethod
    async def from_file(cls, file_path: str):
        proc = await asyncio.create_subprocess_exec(
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            file_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        metadata = json.loads(stdout)
        return cls(
            path=file_path,
            duration=float(metadata["format"]["duration"]),
            width=int(metadata["streams"][0]["width"]),
            height=int(metadata["streams"][0]["height"]),
            format=metadata["format"]["format_name"],
            codec=metadata["streams"][0]["codec_name"],
        )


class ArxivVideoGenerator:
    def __init__(self):
        self.source_videos = []

        self.whisper = WhisperModel("distil-large-v3")
        self.tokenizer = Tokenizer(self.whisper.hf_tokenizer, True, "transcribe", "en")

        self.ratelimit = AsyncLimiter(REQUESTS_PER_MIN)
        with open("openai_api_key", "r") as f:
            api_key = f.read().strip()
        self.client = openai.AsyncOpenAI(api_key=api_key)

    async def get_source_videos(self):
        if not self.source_videos:
            os.makedirs("source_videos", exist_ok=True)
            videos = os.listdir("source_videos")
            if not videos:
                os.chdir("source_videos")
                try:
                    args = yt_dlp.parse_options(YT_DLP_ARGV).ydl_opts
                    with yt_dlp.YoutubeDL(args) as ydl:
                        ydl.download(DEFAULT_VIDEOS)
                finally:
                    os.chdir("..")

                videos = os.listdir("source_videos")
                assert videos

            self.source_videos = await asyncio.gather(
                *(SourceVideo.from_file(f"source_videos/{f}") for f in videos)
            )

        return self.source_videos

    async def get_papers(self):
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "http://export.arxiv.org/api/query?search_query=cat:cs.LG&sortBy=lastUpdatedDate&sortOrder=descending"
            ) as response:
                content = await response.text()

        root = ET.fromstring(content)
        entries = root.findall("{http://www.w3.org/2005/Atom}entry")
        for entry in entries:
            yield Paper.from_entry(entry)

        # TODO: get more than returned by default

    async def say(self, input, voice, filename):
        logging.debug(f"speaking '{input[:50]}...' using '{voice}'")
        async with self.ratelimit, self.client.audio.speech.with_streaming_response.create(
            model=MODEL,
            voice=voice,
            input=input,
            response_format=FORMAT,
        ) as response:
            await response.stream_to_file(filename)
        logging.debug(f"spoke '{input[:50]}...' using '{voice}'")

    async def say_sentences(self, input, voice, out_dir):
        sentences = nltk.sent_tokenize(input)
        filenames = []
        os.makedirs(out_dir, exist_ok=True)
        async with asyncio.TaskGroup() as tg:
            for i, sentence in enumerate(sentences):
                filename = f"{out_dir}/sentence{i:02d}.{FORMAT}"
                filenames.append(filename)
                if not os.path.isfile(filename):
                    tg.create_task(self.say(sentence, voice, filename))

        return sentences, filenames

    def align_sentence(self, sentence, audio_path):
        @dataclass(frozen=True, slots=True)
        class Alignment:
            words: list[str]
            word_start_times: list[float]
            sentence_duration: float

        print(f"aligning '{sentence[:50]}...' with {audio_path}")

        audio = decode_audio(
            audio_path,
            sampling_rate=self.whisper.feature_extractor.sampling_rate,
        )
        features = self.whisper.feature_extractor(audio)

        content_frames = features.shape[-1]

        text_tokens = [*self.tokenizer.encode(sentence), self.tokenizer.eot]
        token_start_times = np.full(len(text_tokens), np.inf)
        seen_tokens = 0

        for seek in range(
            0, content_frames, self.whisper.feature_extractor.nb_max_frames
        ):
            segment_size = min(
                self.whisper.feature_extractor.nb_max_frames,
                content_frames - seek,
            )
            segment = features[:, seek : seek + segment_size]
            segment = pad_or_trim(segment, self.whisper.feature_extractor.nb_max_frames)

            encoder_output = self.whisper.encode(segment)

            result = self.whisper.model.align(
                encoder_output,
                self.tokenizer.sot_sequence,
                [text_tokens[seen_tokens:]],
                segment_size,
            )[0]

            token_indices = np.array([pair[0] for pair in result.alignments])
            time_indices = np.array([pair[1] for pair in result.alignments])

            # Update token_start_times for newly aligned tokens
            new_seen_tokens = seen_tokens
            seen_time = seek * self.whisper.feature_extractor.time_per_frame
            for local_token, local_time in zip(token_indices, time_indices):
                token = seen_tokens + local_token
                new_seen_tokens = max(seen_tokens, token)
                if token < len(token_start_times):
                    time = local_time / self.whisper.tokens_per_second + seen_time
                    token_start_times[token] = min(token_start_times[token], time)

            seen_tokens = new_seen_tokens
            if seen_tokens == len(text_tokens):
                break

        np.minimum.accumulate(token_start_times[::-1], out=token_start_times[::-1])

        # words, word_tokens = self.tokenizer.split_tokens_on_unicode(text_tokens)
        words, word_tokens = self.tokenizer.split_to_word_tokens(text_tokens)
        word_boundaries = np.cumsum([len(t) for t in word_tokens])

        sentence_duration = len(audio) / self.whisper.feature_extractor.sampling_rate
        word_start_times = [
            min(token_start_times[boundary - len(t)], sentence_duration)
            for boundary, t in zip(word_boundaries, word_tokens)
        ]

        # print("words", words)
        # print("word_start_times", word_start_times)
        # print("sentence_duration", sentence_duration)

        return Alignment(words, word_start_times, sentence_duration)

    def create_ass_subtitle(
        self, sentences, alignments, paper_dir, video_width, video_height
    ):
        def format_time(seconds):
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            seconds = seconds % 60
            centiseconds = int((seconds - int(seconds)) * 100)
            return f"{hours:01d}:{minutes:02d}:{int(seconds):02d}.{centiseconds:02d}"

        with open(f"{paper_dir}/subtitles.ass", "w", encoding="utf-8") as f:
            # Write minimal ASS header
            f.write("[Script Info]\n")
            f.write("ScriptType: v4.00+\n")
            f.write(f"PlayResX: {video_width}\n")
            f.write(f"PlayResY: {video_height}\n\n")

            # Minimal style definition with Comic Sans MS, large font, and very thick outline
            f.write("[V4+ Styles]\n")
            f.write(
                "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            )
            f.write(
                "Style: Default,Comic Sans MS,96,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,8,0,2,10,10,10,1\n\n"
            )

            f.write("[Events]\n")
            f.write(
                "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
            )

            cumulative_duration = 0
            for alignment in alignments:
                for i in range(len(alignment.word_start_times) - 1):
                    start_time = alignment.word_start_times[i] + cumulative_duration
                    end_time = alignment.word_start_times[i + 1] + cumulative_duration
                    assert start_time < float("inf")
                    assert end_time < float("inf")

                    def valid(word):
                        return any(c not in string.punctuation + string.whitespace for c in word)
                    
                    def escape(word):
                        return word.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")

                    start_str = format_time(start_time)
                    end_str = format_time(end_time)
                    line = "".join(
                        f"{{\\b1}}{word}{{\\b0}}" if j == i and valid(word) else word
                        for j, word in enumerate(map(escape, alignment.words))
                    )

                    f.write(
                        f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{{\\an5}}{line}\n"
                    )

                cumulative_duration += alignment.sentence_duration

    async def process_paper(self, paper, out_dir):
        print(f"processing {paper.id} '{paper.title}'")
        abstract = paper.abstract.replace("\n", " ")
        voice = random.choice(VOICES)
        sentences, filenames = await self.say_sentences(abstract, voice, out_dir)
        with open(f"{out_dir}/sentences.txt", "w") as f:
            for filename in filenames:
                print(f"file '{filename.split("/")[-1]}'", file=f)

        alignments = [self.align_sentence(s, f) for s, f in zip(sentences, filenames)]
        audio_duration = sum(a.sentence_duration for a in alignments)

        source_video = random.choice(await self.get_source_videos())
        max_start = max(0, source_video.duration - audio_duration)
        start_time = random.uniform(0, max_start)

        self.create_ass_subtitle(
            sentences, alignments, out_dir, source_video.width, source_video.height
        )

        proc = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-y",
            "-ss",
            str(start_time),
            "-i",
            source_video.path,
            "-f",
            "concat",
            "-i",
            f"{out_dir}/sentences.txt",
            "-vf",
            f"ass={out_dir}/subtitles.ass",
            "-c:v",
            "libx264",
            "-preset",
            "veryslow",
            "-crf",
            "18",
            "-movflags",
            "+faststart",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-map_metadata",
            "-1",
            f"{out_dir}/video.mp4",
        )
        await proc.wait()
        print(f"created {out_dir}/video.mp4")

    async def run(self):
        i = 50
        async for paper in self.get_papers():
            out_dir = paper.id
            os.makedirs(out_dir, exist_ok=True)
            await self.process_paper(paper, out_dir)
            i -= 1
            if i == 0:
                break


if __name__ == "__main__":
    generator = ArxivVideoGenerator()
    asyncio.run(generator.run())
