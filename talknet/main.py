import sieve

metadata = sieve.Metadata(
    description="An active speaker detection model to detect which people are speaking in a video.",
    image=sieve.Image(
        url="https://d3i71xaburhd42.cloudfront.net/a832f8978c55d6b127b70e1941604bfd3d1a06e6/2-Figure1-1.png"
    ),
    code_url="https://github.com/sieve-community/fast-asd/talknet",
    tags=["Video"],
    readme=open("README.md", "r").read()
)

@sieve.Model(
    name="talknet-asd",
    python_packages=[
        "torch>=1.6.0",
        "torchaudio>=0.6.0",
        "numpy",
        "scipy",
        "scikit-learn",
        "tqdm",
        "scenedetect",
        "opencv-python",
        "python_speech_features",
        "torchvision",
        "ffmpeg",
        "gdown",
        "youtube-dl",
    ],
    system_packages=[
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "ffmpeg"
    ],
    run_commands=[
        "pip install pandas",
        "mkdir -p /root/.cache/models",
        "gdown --id 1J-PDWDAkYCdT8T2Nxn3Q_-iOHH_t-9YP -O /root/.cache/models/pretrain_TalkSet.model",
        "pip install supervision",
    ],
    cuda_version="11.8",
    gpu=sieve.gpu.L4(split=3),
    metadata=metadata
)
class TalkNetASD:
    def __setup__(self):
        from demoTalkNet import setup
        self.s, self.DET = setup()

    def __predict__(
        self,
        video: sieve.File,
        start_time: float = 0,
        end_time: float = -1,
        return_visualization: bool = False,
        face_boxes: str = "",
        in_memory_threshold: int = 0,
    ):
        """
        :param video: a video to process
        :param start_time: the start time of the video to process (in seconds)
        :param end_time: the end time of the video to process (in seconds). If -1, process until the end of the video.
        :param return_visualization: whether to return the visualization of the video.
        :param face_boxes: a string of face boxes in the format "frame_number,x1,y1,x2,y2,x1,y1,x2,y2,..." separated by new lines per frame. If not provided, the model will detect the faces in the video itself to then detect the active speaker.
        :param in_memory_threshold: the maximum number of frames to load in memory at once. can speed up processing. if 0, this feature is disabled.
        :return: if return_visualization is True, the first element of the tuple is the output of the model, and the second element is the visualization of the video. Otherwise, the first element is the output of the model.
        """
        from demoTalkNet import main
        def transform_out(out):
            outputs = []
            for o in out:
                outputs.append({
                    "frame_number": o['frame_number'],
                    "boxes": [b for b in o['faces']]
                })
            return outputs
            
        if return_visualization:
            out, video_path = main(self.s, self.DET, video.path, start_seconds=start_time, end_seconds=end_time, return_visualization=return_visualization, face_boxes=face_boxes, in_memory_threshold=in_memory_threshold)
            return sieve.Video(path=video_path)
        else:
            out = main(self.s, self.DET, video.path, start_seconds=start_time, end_seconds=end_time, return_visualization=return_visualization, face_boxes=face_boxes, in_memory_threshold=in_memory_threshold)
            return transform_out(out)

if __name__ == "__main__":
    TEST_URL = "https://storage.googleapis.com/sieve-prod-us-central1-public-file-upload-bucket/d979a930-f2a5-4e0d-84fe-a9b233985c4e/dba9cbf3-8374-44bc-8d9d-cc9833d3f502-input-file.mp4"
    model = TalkNetASD()
    # change "url" to "path" if you want to test with a local file
    out = model(sieve.Video(url=TEST_URL), return_visualization=False)
    print(list(out))