from lib import *

if __name__ == "__main__":
    import argparse
    import os
    from scipy.stats import mode
    
    parser = argparse.ArgumentParser(description='')

    parser.add_argument(
        '--video_folder', dest='video_folder', action='store',
        type=str,help='Folder containing videos.')
    parser.add_argument(
        '--mp3_path', dest='mp3_path', action='store',
        type=str,help='Path to MP3 file.')
    parser.add_argument(
        '--fps', dest='fps', action='store',
        default=24,
        type=int,help='FPS for the output.')
    parser.add_argument(
        '--output_path', dest='output_path', action='store',
        type=str,help='Path to output video.')
    parser.add_argument(
        '--height', dest='height', action='store',
        default=720,
        type=int,help='Output height.')
    parser.add_argument(
        '--width', dest='width', action='store',
        default=1280,
        type=int,help='Output width.')
    parser.add_argument(
        '--no_freak', dest='no_freak', action='store_false',
        default=True,
        help='No effects are added.')
    parser.add_argument(
        '--no_shift', dest='no_shift', action='store_false',
        default=True,
        help='No translation happens.')
    parser.add_argument(
        '--bass_response', dest='bass_response', 
        action='store',
        type=float,
        default=0.7,
        help='Range for bass response.')    

    args = parser.parse_args()
    
    a = time.time()

    all_video_paths = glob(os.path.join(args.video_folder,'*'))
    music,sr = librosa.load(args.mp3_path)
    audioclip = AudioFileClip(args.mp3_path)

    onset_frames = librosa.onset.onset_detect(music, sr=sr, wait=0)
    onset_times = librosa.frames_to_time(onset_frames)
    
    tempo_at_onset_times = tempo_at_times(music,sr,onset_times)
    IF = np.abs(librosa.stft(music,sr,hop_length=int(sr/24),win_length=int(sr/24)))
    IF = np.sum(IF[:500,:],axis=0)
    intensity_frame = (IF-IF.min())/(IF.max() - IF.min())
    intensity_frame = np.int16((1-intensity_frame)*(255*args.bass_response))

    n_sec = 1
    fps = args.fps
    max_clip_duration = fps * n_sec
    D = audioclip.duration 
    sh = (args.height,args.width)

    V = VideoFrameGenerator(all_video_paths,
                            onset_times,sh,
                            max_clip_duration,
                            fps,tempo_at_onset_times,D,
                            intensity_frame,
                            freak=args.no_freak,
                            shift=args.no_shift)

    v = VideoClip(V.get,duration=D)
    v = v.set_audio(audioclip.subclip(0,float(D)))
    print("Writing video...")
    v.write_videofile(args.output_path,
                      fps=fps,
                      audio=True)
    V.empty_queue()
    print(time.time()-a)
