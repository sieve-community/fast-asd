# fast-asd

This repository is an optimized, production-ready implementation of active speaker detection. Read more about the research area [here](https://paperswithcode.com/task/audio-visual-active-speaker-detection).

It contains of two parts:
- The open-source implementation of the [active speaker detection](https://www.sievedata.com/functions/sieve/active_speaker_detection) application that runs on the [Sieve](https://www.sievedata.com/) platform.
- The standalone, optimized implementation of [TalkNet](https://github.com/TaoRuijie/TalkNet-ASD), a leading model for active speaker detection.

## Usage

### TalkNet
If you plan to just use the standalone implementation of TalkNet, follow the steps below:

1. go to the `talknet` directory
2. run `pip install -r requirements.txt`
3. run `python main.py`

You can change the input video file being used by modifying the `main` function in `main.py`.

### Active Speaker Detection

The easiest way to run active speaker detection is to use the version already deployed on the Sieve platform available [here](https://www.sievedata.com/functions/sieve/active_speaker_detection).

While the core application can be run locally, it still calls public functions available on Sieve, such at the YOLO object detection model so you will need to sign up for a free account and get an API key. You can do so [here](https://www.sievedata.com/).

After you've signed up and run `sieve login`, you can run `main.py` from the root directory of this repository to run the active speaker detection application.