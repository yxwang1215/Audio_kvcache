from .audio_process import process_audio_info
from .vision_process import (
    extract_vision_info,
    fetch_image,
    fetch_video,
    process_vision_info,
    smart_resize,
)


def process_mm_info(conversations, use_audio_in_video, return_video_kwargs=False):
    audios = process_audio_info(conversations, use_audio_in_video)
    vision = process_vision_info(conversations, return_video_kwargs=return_video_kwargs)
    print("Audio :", audios, "Type :", type(audios))    # 返回一个列表
    print("Vision :", vision, "Type :", type(vision))   # 返回一个元组，（None, None）
    return (audios,) + vision
