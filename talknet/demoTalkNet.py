import sys, time, os, tqdm, torch, glob, subprocess, warnings, cv2, pickle, numpy, math, python_speech_features

from scipy import signal
from shutil import rmtree
from scipy.io import wavfile
from scipy.interpolate import interp1d

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

from model.faceDetector.s3fd import S3FD
from talkNet import talkNet

warnings.filterwarnings("ignore")

pretrained_model_path = "/root/.cache/models/pretrain_TalkSet.model"
save_path = "save/"
data_loader_thread = 10
face_detection_scale = 0.25
min_track = 10 # Number of min frames for each shot
num_failed_det = 10 # Number of missed detections allowed before tracking is stopped
min_face_size = 1 # Minimum face size in pixels
crop_scale = 0.40 # Scale bounding box
start = 0 # The start time of the video
duration = 0 # The duration of the video, when set as 0, will extract the whole video
pyaviPath = os.path.join(save_path, 'pyavi')
pyframesPath = os.path.join(save_path, 'pyframes')
pyworkPath = os.path.join(save_path, 'pywork')
pycropPath = os.path.join(save_path, 'pycrop')
videoFilePath = os.path.join(pyaviPath, 'video.avi')
audioFilePath = os.path.join(pyaviPath, 'audio.wav')

if os.path.isfile(pretrained_model_path) == False: # Download the pretrained model
    Link = "1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea"
    cmd = "gdown --id %s -O %s"%(Link, pretrained_model_path)
    subprocess.call(cmd, shell=True, stdout=None)

def scene_detect(video_path, save = False, start_frame = 0, end_frame = None):
	# CPU: Scene detection, output is the list of each shot's time duration
	videoManager = VideoManager([video_path])
	statsManager = StatsManager()
	sceneManager = SceneManager(statsManager)
	sceneManager.add_detector(ContentDetector())
	baseTimecode = videoManager.get_base_timecode()
	if start_frame != 0 or end_frame is not None:
		start_timecode = FrameTimecode(timecode=start_frame, fps=videoManager.get_framerate())
		end_timecode = FrameTimecode(timecode=end_frame, fps=videoManager.get_framerate())
		videoManager.set_duration(start_time=start_timecode, end_time=end_timecode)
	videoManager.set_downscale_factor()
	videoManager.start()
	sceneManager.detect_scenes(frame_source = videoManager)
	sceneList = sceneManager.get_scene_list(baseTimecode)
	if sceneList == []:
		if start_frame == 0 and end_frame is None:
			sceneList = [(videoManager.get_base_timecode(),videoManager.get_current_timecode())]
		else:
			sceneList = [(start_timecode, end_timecode)]
	if save:
		savePath = os.path.join(pyworkPath, 'scene.pckl')
		with open(savePath, 'wb') as fil:
			pickle.dump(sceneList, fil)
			# sys.stderr.write('%s - scenes detected %d\n'%(video_path, len(sceneList)))
	return sceneList

def initialize_detector(device='cuda'):
	# Initialize the face detector
	DET = S3FD(device=device)
	return DET

def predict_faces(DET, start_frame = 0):
	# GPU: Face detection, output is the list contains the face location and score in this frame
	flist = glob.glob(os.path.join(pyframesPath, '*.jpg'))
	flist.sort()
	dets = []
	for fidx, fname in enumerate(flist):
		image = cv2.imread(fname)
		imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		bboxes = DET.detect_faces(imageNumpy, conf_th=0.9, scales=[face_detection_scale])
		dets.append([])
		for bbox in bboxes:
			dets[-1].append({'frame':fidx + start_frame, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]}) # dets has the frames info, bbox info, conf info

	savePath = os.path.join(pyworkPath,'faces.pckl')
	with open(savePath, 'wb') as fil:
		pickle.dump(dets, fil)
	return dets

def predict_faces_from_frames(DET, frames, start_frame = 0):
	# GPU: Face detection, output is the list contains the face location and score in this frame
	dets = []
	import time
	for fidx, frame in enumerate(frames):
		t = time.time()
		imageNumpy = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		bboxes = DET.detect_faces(imageNumpy, conf_th=0.9, scales=[face_detection_scale])
		dets.append([])
		for bbox in bboxes:
			dets[-1].append({'frame':fidx + start_frame, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]}) # dets has the frames info, bbox info, conf info

	savePath = os.path.join(pyworkPath,'faces.pckl')
	with open(savePath, 'wb') as fil:
		pickle.dump(dets, fil)
	return dets

def bb_intersection_over_union(boxA, boxB, evalCol = False):
	# CPU: IOU Function to calculate overlap between two image
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	interArea = max(0, xB - xA) * max(0, yB - yA)
	boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
	boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
	if evalCol == True:
		iou = interArea / float(boxAArea)
	else:
		iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou

def track_shot(sceneFaces):
	# CPU: Face tracking
	iouThres  = 0.5     # Minimum IOU between consecutive face detections
	tracks    = []
	while True:
		track     = []
		for frameFaces in sceneFaces:
			for face in frameFaces:
				if track == []:
					track.append(face)
					frameFaces.remove(face)
				elif face['frame'] - track[-1]['frame'] <= num_failed_det:
					iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
					if iou > iouThres:
						track.append(face)
						frameFaces.remove(face)
						continue
				else:
					break
		if track == []:
			break
		elif len(track) > min_track:
			frameNum    = numpy.array([ f['frame'] for f in track ])
			bboxes      = numpy.array([numpy.array(f['bbox']) for f in track])
			frameI      = numpy.arange(frameNum[0],frameNum[-1]+1)
			bboxesI    = []
			for ij in range(0,4):
				interpfn  = interp1d(frameNum, bboxes[:,ij])
				bboxesI.append(interpfn(frameI))
			bboxesI  = numpy.stack(bboxesI, axis=1)
			if max(numpy.mean(bboxesI[:,2]-bboxesI[:,0]), numpy.mean(bboxesI[:,3]-bboxesI[:,1])) > min_face_size:
				tracks.append({'frame':frameI,'bbox':bboxesI})
	return tracks

def crop_video(track, cropFile, start_frame = 0):
	# CPU: crop the face clips
	flist = glob.glob(os.path.join(pyframesPath, '*.jpg')) # Read the frames
	flist.sort()
	vOut = cv2.VideoWriter(cropFile + 't.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (224,224))# Write video
	dets = {'x':[], 'y':[], 's':[]}
	for det in track['bbox']: # Read the tracks
		dets['s'].append(max((det[3]-det[1]), (det[2]-det[0]))/2) 
		dets['y'].append((det[1]+det[3])/2) # crop center x 
		dets['x'].append((det[0]+det[2])/2) # crop center y
	dets['s'] = signal.medfilt(dets['s'], kernel_size=13)  # Smooth detections 
	dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
	dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
	for fidx, frame in enumerate(track['frame']):
		cs  = crop_scale # Crop scale
		bs  = dets['s'][fidx]   # Detection box size
		bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount 
		# print(fidx, frame, len(flist))
		image = cv2.imread(flist[frame])
		frame = numpy.pad(image, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))
		my  = dets['y'][fidx] + bsi  # BBox center Y
		mx  = dets['x'][fidx] + bsi  # BBox center X
		face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
		vOut.write(cv2.resize(face, (224, 224)))
	audioTmp    = cropFile + '.wav'
	audioStart  = (track['frame'][0]) / 25
	audioEnd    = (track['frame'][-1]+1) / 25
	vOut.release()
	command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic" % \
		      (audioFilePath, data_loader_thread, audioStart, audioEnd, audioTmp)) 
	output = subprocess.call(command, shell=True, stdout=None) # Crop audio file
	_, audio = wavfile.read(audioTmp)
	command = ("ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a aac %s.avi -loglevel panic" % \
			  (cropFile, audioTmp, data_loader_thread, cropFile)) # Combine audio and video file
	output = subprocess.call(command, shell=True, stdout=None)
	os.remove(cropFile + 't.avi')
	return {'track':track, 'proc_track':dets}

def crop_video_from_frames(track, cropFile, frames, start_frame = 0):
	# CPU: crop the face clips
	vOut = cv2.VideoWriter(cropFile + 't.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (224,224))# Write video
	dets = {'x':[], 'y':[], 's':[]}
	for det in track['bbox']: # Read the tracks
		dets['s'].append(max((det[3]-det[1]), (det[2]-det[0]))/2) 
		dets['y'].append((det[1]+det[3])/2) # crop center x 
		dets['x'].append((det[0]+det[2])/2) # crop center y
	dets['s'] = signal.medfilt(dets['s'], kernel_size=13)  # Smooth detections 
	dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
	dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
	for fidx, frame in enumerate(track['frame']):
		cs  = crop_scale # Crop scale
		bs  = dets['s'][fidx]   # Detection box size
		bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount 
		# print(fidx, frame, len(flist))
		# print(fidx, frame, len(frames))
		image = frames[frame]
		frame = numpy.pad(image, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))
		my  = dets['y'][fidx] + bsi  # BBox center Y
		mx  = dets['x'][fidx] + bsi  # BBox center X
		face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
		vOut.write(cv2.resize(face, (224, 224)))
	audioTmp    = cropFile + '.wav'
	audioStart  = (track['frame'][0]) / 25
	audioEnd    = (track['frame'][-1]+1) / 25
	vOut.release()
	command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic" % \
		      (audioFilePath, data_loader_thread, audioStart, audioEnd, audioTmp)) 
	output = subprocess.call(command, shell=True, stdout=None) # Crop audio file
	_, audio = wavfile.read(audioTmp)
	command = ("ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a aac %s.avi -loglevel panic" % \
			  (cropFile, audioTmp, data_loader_thread, cropFile)) # Combine audio and video file
	output = subprocess.call(command, shell=True, stdout=None)
	os.remove(cropFile + 't.avi')
	return {'track':track, 'proc_track':dets}

def extract_MFCC(file, outPath):
	# CPU: extract mfcc
	sr, audio = wavfile.read(file)
	mfcc = python_speech_features.mfcc(audio,sr) # (N_frames, 13)   [1s = 100 frames]
	featuresPath = os.path.join(outPath, file.split('/')[-1].replace('.wav', '.npy'))
	numpy.save(featuresPath, mfcc)

def evaluate_network(s, files):
	# GPU: active speaker detection by pretrained TalkNet
	allScores = []
	# durationSet = {1,2,4,6} # To make the result more reliable
	durationSet = {1,1,1,2,2,2,3,3,4,5,6} # Use this line can get more reliable result
	for file in tqdm.tqdm(files, total = len(files)):
		fileName = os.path.splitext(file.split('/')[-1])[0] # Load audio and video
		# print(os.path.join(pycropPath, fileName + '.wav'))
		_, audio = wavfile.read(os.path.join(pycropPath, fileName + '.wav'))
		audioFeature = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025, winstep = 0.010)
		# print(os.path.join(pycropPath, fileName + '.avi'))
		video = cv2.VideoCapture(os.path.join(pycropPath, fileName + '.avi'))
		videoFeature = []
		while video.isOpened():
			ret, frames = video.read()
			if ret == True:
				face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
				face = cv2.resize(face, (224,224))
				face = face[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
				videoFeature.append(face)
			else:
				break
		video.release()
		videoFeature = numpy.array(videoFeature)
		length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0] / 25)
		audioFeature = audioFeature[:int(round(length * 100)),:]
		videoFeature = videoFeature[:int(round(length * 25)),:,:]
		allScore = [] # Evaluation use TalkNet
		for duration in durationSet:
			batchSize = int(math.ceil(length / duration))
			scores = []
			with torch.no_grad():
				for i in range(batchSize):
					inputA = torch.FloatTensor(audioFeature[i * duration * 100:(i+1) * duration * 100,:]).unsqueeze(0).cuda()
					inputV = torch.FloatTensor(videoFeature[i * duration * 25: (i+1) * duration * 25,:,:]).unsqueeze(0).cuda()
					embedA = s.model.forward_audio_frontend(inputA)
					embedV = s.model.forward_visual_frontend(inputV)	
					embedA, embedV = s.model.forward_cross_attention(embedA, embedV)
					out = s.model.forward_audio_visual_backend(embedA, embedV)
					score = s.lossAV.forward(out, labels = None)
					scores.extend(score)
			allScore.append(scores)
		allScore = numpy.round((numpy.mean(numpy.array(allScore), axis = 0)), 1).astype(float)
		allScores.append(allScore)	
	return allScores

def visualization(tracks, scores):
	# CPU: visulize the result for video format
	flist = glob.glob(os.path.join(pyframesPath, '*.jpg'))
	flist.sort()
	faces = [[] for i in range(len(flist))]
	for tidx, track in enumerate(tracks):
		score = scores[tidx]
		for fidx, frame in enumerate(track['track']['frame'].tolist()):
			# print(fidx, frame, len(flist))
			s = score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)] # average smoothing
			s = numpy.mean(s)
			faces[frame].append({'track':tidx, 'score':float(s),'s':track['proc_track']['s'][fidx], 'x':track['proc_track']['x'][fidx], 'y':track['proc_track']['y'][fidx]})
	firstImage = cv2.imread(flist[0])
	fw = firstImage.shape[1]
	fh = firstImage.shape[0]
	vOut = cv2.VideoWriter(os.path.join(pyaviPath, 'video_only.avi'), cv2.VideoWriter_fourcc(*'XVID'), 25, (fw, fh))
	colorDict = {0: 0, 1: 255}
	for fidx, fname in tqdm.tqdm(enumerate(flist), total = len(flist)):
		image = cv2.imread(fname)
		for face in faces[fidx]:
			clr = colorDict[int((face['score'] >= 0))]
			txt = round(face['score'], 1)
			cv2.rectangle(image, (int(face['x']-face['s']), int(face['y']-face['s'])), (int(face['x']+face['s']), int(face['y']+face['s'])),(0,clr,255-clr),10)
			cv2.putText(image,'%s'%(txt), (int(face['x']-face['s']), int(face['y']-face['s'])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,clr,255-clr),5)
		vOut.write(image)
	vOut.release()
	command = ("ffmpeg -y -i %s -i %s -threads %d %s" % \
		(os.path.join(pyaviPath, 'video_only.avi'), os.path.join(pyaviPath, 'audio.wav'), \
		data_loader_thread, os.path.join(pyaviPath,'video_out.mp4')))
	output = subprocess.call(command, shell=True, stdout=None)

def visualization_from_frames(tracks, scores, frames):
	faces = [[] for i in range(len(frames))]
	for tidx, track in enumerate(tracks):
		score = scores[tidx]
		for fidx, frame in enumerate(track['track']['frame'].tolist()):
			# print(fidx, frame, len(flist))
			s = score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)] # average smoothing
			s = numpy.mean(s)
			faces[frame].append({'track':tidx, 'score':float(s),'s':track['proc_track']['s'][fidx], 'x':track['proc_track']['x'][fidx], 'y':track['proc_track']['y'][fidx]})
	firstImage = frames[0]
	fw = firstImage.shape[1]
	fh = firstImage.shape[0]
	vOut = cv2.VideoWriter(os.path.join(pyaviPath, 'video_only.avi'), cv2.VideoWriter_fourcc(*'XVID'), 25, (fw, fh))
	colorDict = {0: 0, 1: 255}
	for fidx, frame in tqdm.tqdm(enumerate(frames), total = len(frames)):
		image = frame
		for face in faces[fidx]:
			clr = colorDict[int((face['score'] >= 0))]
			txt = round(face['score'], 1)
			cv2.rectangle(image, (int(face['x']-face['s']), int(face['y']-face['s'])), (int(face['x']+face['s']), int(face['y']+face['s'])),(0,clr,255-clr),10)
			cv2.putText(image,'%s'%(txt), (int(face['x']-face['s']), int(face['y']-face['s'])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,clr,255-clr),5)
		vOut.write(image)
	vOut.release()
	command = ("ffmpeg -y -i %s -i %s -threads %d %s" % \
		(os.path.join(pyaviPath, 'video_only.avi'), os.path.join(pyaviPath, 'audio.wav'), \
		data_loader_thread, os.path.join(pyaviPath,'video_out.mp4')))
	output = subprocess.call(command, shell=True, stdout=None)

def setup():
	# return initialized model
	s = talkNet()
	s.loadParameters(pretrained_model_path)
	sys.stderr.write("Model %s loaded from previous state! \r\n"%(pretrained_model_path))
	s.eval()

	DET = initialize_detector()

	return s, DET

# Main function
def main(
	s,
    DET,
    video_path,
    start_seconds = 0,
    end_seconds = None,
    return_visualization = False,
    face_boxes = "",
	in_memory_threshold = 0
):
	import subprocess
	# This preprocesstion is modified based on this [repository](https://github.com/joonson/syncnet_python).
	# ```
	# .
	# ├── pyavi
	# │   ├── audio.wav (Audio from input video)
	# │   ├── video.avi (Copy of the input video)
	# │   ├── video_only.avi (Output video without audio)
	# │   └── video_out.avi  (Output video with audio)
	# ├── pycrop (The detected face videos and audios)
	# │   ├── 000000.avi
	# │   ├── 000000.wav
	# │   ├── 000001.avi
	# │   ├── 000001.wav
	# │   └── ...
	# ├── pyframes (All the video frames in this video)
	# │   ├── 000001.jpg
	# │   ├── 000002.jpg
	# │   └── ...	
	# └── pywork
	#     ├── faces.pckl (face detection result)
	#     ├── scene.pckl (scene detection result)
	#     ├── scores.pckl (ASD result)
	#     └── tracks.pckl (face tracking result)
	# ```

	# Initialization 
	if os.path.exists(save_path):
		rmtree(save_path)
	os.makedirs(pyaviPath, exist_ok = True) # The path for the input video, input audio, output video
	os.makedirs(pyframesPath, exist_ok = True) # Save all the video frames
	os.makedirs(pyworkPath, exist_ok = True) # Save the results in this process by the pckl method
	os.makedirs(pycropPath, exist_ok = True) # Save the detected face clips (audio+video) in this process
	
	STORE_FRAMES_IN_MEMORY_THRESHOLD = in_memory_threshold

    # Extract video information
    # get video duration
	print("Extracting video information...")
	command = ("ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 '%s'" % video_path)
	video_num_frames = float(subprocess.check_output(command, shell=True))
	if video_num_frames == 0 or math.isnan(video_num_frames):
		raise ValueError("Video has no frames or is corrupted.")
	
	# get video length
	command = ("ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 '%s'" % video_path)
	video_duration = float(subprocess.check_output(command, shell=True))
	if video_duration == 0 or math.isnan(video_duration):
		raise ValueError("Video has no duration or is corrupted.")
	
	# get fps
	command = ("ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 '%s'" % video_path)
	fps_output = subprocess.check_output(command, shell=True).decode('utf-8').strip()
	fps = float(fps_output.split('/')[0]) / float(fps_output.split('/')[1]) if '/' in fps_output else float(fps_output)
	if fps == 0 or math.isnan(fps):
		raise ValueError("Video has no frames or is corrupted.")
	
	print("Video duration: ", video_duration)
	print("FPS: ", fps)
	print("Video num frames: ", video_num_frames)

	if start_seconds < 0 or start_seconds > video_duration:
		raise ValueError("Start time is out of bounds.")
	if end_seconds is not None and end_seconds != -1 and (end_seconds < 0 or end_seconds > video_duration):
		end_seconds = video_duration
		print("End time is out of bounds. Setting to video duration.")
	
	if end_seconds is None or end_seconds == -1:
		end_seconds = video_duration

	
	# Extract video between start_seconds and end_seconds and save it to extracted_video.mp4
	start = start_seconds
	start_frame = int(start * fps)
	end_frame = int(end_seconds * fps)
	duration = end_seconds - start_seconds

	# Extract audio
	print("Extracting audio...")
	t = time.time()
	command = ("ffmpeg -y -i '%s' -ss %s -t %s -qscale:a 0 -ac 1 -vn -threads %d -ar 16000 %s -loglevel panic" % \
		(video_path, start_seconds, duration, data_loader_thread, audioFilePath))
	subprocess.call(command, shell=True, stdout=None)
	print("Audio extracted in %.3f seconds."%(time.time() - t))
	# sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the audio and save in %s \r\n" %(audioFilePath))

	print("Extracting video frames...")
	t = time.time()
	# if less than 5000 frames, store in memory
	if (end_seconds - start_seconds) * fps < STORE_FRAMES_IN_MEMORY_THRESHOLD:
		# use cv2 to extract frames
		import cv2
		vidcap = cv2.VideoCapture(video_path)
		frames = []
		frames_number_to_read = []
		for i in range(int((end_seconds - start_seconds) * 25) + 1):
			frame_number = int(start_frame + i * (fps / 25))
			if start_frame <= frame_number < end_frame:
				frames_number_to_read.append(frame_number)
		
		current_frame_number = start_frame
		vidcap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_number)
		ret, current_frame = vidcap.read()
		if ret and current_frame is not None:
			print("Using OpenCV to extract frames...")
			count = 0
			for p in frames_number_to_read:
				if p == current_frame_number:
					frames.append(current_frame)
					count += 1
				else:
					new_frame_number = current_frame_number
					while new_frame_number < p:
						ret, new_frame = vidcap.read()
						if not ret:
							break
						new_frame_number += 1
					if ret:
						current_frame_number = p
						current_frame = new_frame
					frames.append(current_frame)
			vidcap.release()
		else:
			print("Using imageio to extract frames...")
			vidcap.release()
			import imageio

			frames = []
			cap = imageio.get_reader(video_path)
			try:
				if start_frame != 0:
					cap.set_image_index(start_frame)
				print(f"load & seek time: {round(time.time() - t, 2)}s")
				current_frame = cap.get_next_data()
				current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)
			except IndexError:
				raise ValueError(f"Could not read frame at index {current_frame_number}")
			count = 0
			for p in frames_number_to_read:
				frame_to_process = None
				if p == current_frame_number:
					frame_to_process = current_frame
					frame_to_process = cv2.cvtColor(frame_to_process, cv2.COLOR_RGB2BGR)
				else:
					new_frame_number = p
					if new_frame_number != current_frame_number + 1:
						cap.set_image_index(new_frame_number)
					try:
						new_frame = cap.get_next_data()
					except IndexError:
						break
					if new_frame is not None:
						current_frame_number = p
						current_frame = new_frame
						frame_to_process = current_frame
						frame_to_process = cv2.cvtColor(frame_to_process, cv2.COLOR_RGB2BGR)
					else:
						break
				if frame_to_process is not None:
					frames.append(frame_to_process)
					count += 1
			print(f"Total frames processed: {count}")

			cap.close()
	else:
		# Temporarily commenting out the original code for extracting video frames
		command = ("ffmpeg -y -i '%s' -vf fps=25 -ss %s -t %s -threads %d -f image2 %s -loglevel panic" % \
			(video_path, start, duration, data_loader_thread, os.path.join(pyframesPath, '%06d.jpg')))
		subprocess.call(command, shell=True, stdout=None)
		# sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the frames and save in %s \r\n" %(pyframesPath))
	
	print("Video frames extracted in %.3f seconds."%(time.time() - t))
	# Scene detection for the video frames
	print("Detecting scenes...")
	t = time.time()
	scene = scene_detect(video_path, save = True, start_frame = start_frame, end_frame = int(end_seconds * fps))
	# convert scenes to 25 fps
	new_scenes = []
	for scene_data in scene:
		new_scenes.append((FrameTimecode(timecode=int(scene_data[0].frame_num * (25 / fps)), fps=25), FrameTimecode(timecode=int(scene_data[1].frame_num * (25 / fps)), fps=25)))
	print("Scenes detected in %.3f seconds."%(time.time() - t))
	# sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Scene detection and save in %s \r\n" %(pyworkPath))
	# # Face detection for the video frames
	print("Detecting faces...")
	t = time.time()
	if face_boxes:
		face_boxes = face_boxes.strip().split("\n")
		faces = []
		for face_box in face_boxes:
			frame_data = face_box.split(",")
			frame_number = int(frame_data[0])
			frame_number -= start_frame
			# print(frame_number)
			frame_faces = []
			for i in range(1, len(frame_data), 5):
				bbox = [float(frame_data[i]), float(frame_data[i+1]), float(frame_data[i+2]), float(frame_data[i+3])]
				conf = float(frame_data[i+4])
				frame_faces.append({"frame": frame_number, "bbox": bbox, "conf": conf})
			faces.append(frame_faces)
		
		# these face boxes were from the original frames, so we need to convert them to the 25 fps frames
		# if the original fps is greater than 25, we need to convert the frame numbers to 25 fps and can sample the face boxes
		# if the original fps is less than 25, we need to interpolate the face boxes to 25 fps
		flist = glob.glob(os.path.join(pyframesPath, '*.jpg'))
		if (end_seconds - start_seconds) * fps < STORE_FRAMES_IN_MEMORY_THRESHOLD:
			target_num_frames = len(frames)
		else:
			target_num_frames = len(flist)
		# print(f"Interpolating faces from {len(faces)} frames to {target_num_frames} frames...")
		new_faces = []
		for i in range(target_num_frames):
			target_frame = int(i * (fps / 25))
			# print(i, target_frame, start_frame, end_frame)
			if target_frame >= len(faces):
				new_faces.append(frame_faces)
				continue
			frame_faces = []
			for face in faces[target_frame]:
				frame_faces.append({
					"frame": i, #+ (start_seconds * 25),
					"bbox": face["bbox"],
					"conf": face["conf"]
				})
			new_faces.append(frame_faces)
		faces = new_faces
	else:
		if (end_seconds - start_seconds) * fps < STORE_FRAMES_IN_MEMORY_THRESHOLD:
			faces = predict_faces_from_frames(DET, frames)
		else:
			faces = predict_faces(DET)

	print("Faces detected in %.3f seconds."%(time.time() - t))
	# sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face detection and save in %s \r\n" %(pyworkPath))

	# Face tracking
	print("Tracking faces...")
	t = time.time()
	allTracks, vidTracks = [], []
	first_frame = max(int(start_seconds * 25), 0)
	for shot in new_scenes:
		if shot[1].frame_num - shot[0].frame_num >= min_track: # Discard the shot frames less than minTrack frames
			allTracks.extend(track_shot(faces[max(0, shot[0].frame_num - first_frame):shot[1].frame_num - first_frame])) # 'frames' to present this tracks' timestep, 'bbox' presents the location of the faces
	print("Faces tracked in %.3f seconds."%(time.time() - t))

	print("Cropping faces...")
	for ii, track in enumerate(allTracks):
		if (end_seconds - start_seconds) * fps < STORE_FRAMES_IN_MEMORY_THRESHOLD:
			vidTracks.append(crop_video_from_frames(track, os.path.join(pycropPath, '%05d'%ii), frames))
		else:
			vidTracks.append(crop_video(track, os.path.join(pycropPath, '%05d'%ii)))
	savePath = os.path.join(pyworkPath, 'tracks.pckl')
	with open(savePath, 'wb') as fil:
		pickle.dump(vidTracks, fil)
	# sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face Crop and saved in %s tracks \r\n" %pycropPath)
	print("Faces cropped in %.3f seconds."%(time.time() - t))
	fil = open(savePath, 'rb')
	vidTracks = pickle.load(fil)

	# Active Speaker Detection by TalkNet
	print("Detecting active speakers...")
	t = time.time()
	files = glob.glob("%s/*.avi"%pycropPath)
	files.sort()
	scores = evaluate_network(s, files)
	savePath = os.path.join(pyworkPath, 'scores.pckl')
	with open(savePath, 'wb') as fil:
		pickle.dump(scores, fil)
	# sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Scores extracted and saved in %s \r\n" %pyworkPath)

	if (end_seconds - start_seconds) * fps < STORE_FRAMES_IN_MEMORY_THRESHOLD:
		faces = [{'frame_number': i, 'faces': []} for i in range(len(frames))]
	else:
		flist = glob.glob(os.path.join(pyframesPath, '*.jpg'))
		flist.sort()
		faces = [{'frame_number': i, 'faces': []} for i in range(len(flist))]

	def get_scene_by_frame_number(frame_number):
		for scene_data in new_scenes:
			if scene_data[0].frame_num <= frame_number and scene_data[1].frame_num > frame_number:
				return scene_data
		return None
	
	# print('vidtracks', vidTracks)
	faces_by_scene = {}
	faces_by_frame = {}
	for tidx, track in enumerate(vidTracks):
		score = scores[tidx]
		for fidx, frame in enumerate(track['track']['frame'].tolist()):
			s = score[max(fidx - 3, 0): min(fidx + 3, len(score) - 1)] # average smoothing
			s = numpy.mean(s)
			x1 = int(track['proc_track']['x'][fidx] - track['proc_track']['s'][fidx])
			y1 = int(track['proc_track']['y'][fidx] - track['proc_track']['s'][fidx])
			x2 = int(track['proc_track']['x'][fidx] + track['proc_track']['s'][fidx])
			y2 = int(track['proc_track']['y'][fidx] + track['proc_track']['s'][fidx])
			# print(frame, fidx, len(faces))
			faces[frame]['faces'].append({'track_id': tidx, 'raw_score': float(s), 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'speaking': bool(s >= 0)})
			if frame not in faces_by_frame:
				faces_by_frame[frame] = []
			faces_by_frame[frame].append({'track_id': tidx, 'raw_score': float(s), 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'speaking': bool(s >= 0)})
			# print(frame, frame + int(25 * start_seconds))
			scene_data = get_scene_by_frame_number(frame + int(25 * start_seconds))
			if scene_data is not None:
				scene_num = scene_data[0].frame_num
				if scene_num not in faces_by_scene:
					faces_by_scene[scene_num] = {}
				# for each scene, keep a list of faces organized by frame number
				if frame not in faces_by_scene[scene_num]:
					faces_by_scene[scene_num][frame] = []
				faces_by_scene[scene_num][frame].append({'track_id': tidx, 'raw_score': float(s), 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'speaking': bool(s >= 0)})

	target_num_frames = (int(end_seconds * fps) - int(start_seconds * fps)) + 1
	print(f"Interpolating faces from {len(faces)} frames to {target_num_frames} frames...")

	interpolated_faces = []

	for scene_data in new_scenes:
		scene_num = scene_data[0].frame_num
		num_frames = scene_data[1].frame_num - scene_data[0].frame_num
		target_start_frame = int(scene_data[0].frame_num * (fps / 25))
		target_num_frames = int(num_frames * (fps / 25))
		# interpolate the faces in this scene
		interpolated_frames = []
		for i in range(target_start_frame, target_start_frame + target_num_frames + 1):
			frame_num = -1
			if target_num_frames > 0:
				frame_num = int(i * (num_frames / target_num_frames)) - new_scenes[0][0].frame_num
			if frame_num not in faces_by_frame:
				frame_num = int(i * (25 / fps)) - new_scenes[0][0].frame_num
			interpolated_frames.append({
				'frame_number': i,
				'faces': [] if frame_num not in faces_by_frame else faces_by_frame[frame_num]
			})
		interpolated_faces.extend(interpolated_frames)
			
	faces = interpolated_faces
	
	print("Active speakers detected in %.3f seconds."%(time.time() - t))
	if return_visualization:
		print("Visualizing the result...")
		t = time.time()

		if (end_seconds - start_seconds) * fps < STORE_FRAMES_IN_MEMORY_THRESHOLD:
			visualization_from_frames(vidTracks, scores, frames)
		else:
			visualization(vidTracks, scores)
		print("Result visualized in %.3f seconds."%(time.time() - t))
		# ffmpeg convert avi to mp4
		# subprocess.call(["ffmpeg", "-y", "-i", os.path.join(pyaviPath,'video_out.avi'), os.path.join(pyaviPath,'video_out.mp4')])
		return interpolated_faces, os.path.join(pyaviPath,'video_out.mp4')
	
	if 'frames' in locals():
		del frames
	
	return interpolated_faces
