import glob
import os
import cv2
import image_undistortion

# Permanent paths
IN_DIR = '/home/ubuntu/lens-correction/cpu/distorted'
OUT_DIR = '/home/ubuntu/lens-correction/cpu/new'
CAMERA_PARAM = '/home/ubuntu/lens-correction/cpu/config/camera_intrinsic_parameters.json'

def main():
    video_list = glob.glob(os.path.join(IN_DIR, '*'))

    for video_path in video_list:
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        orig_frame_dir = os.path.join(OUT_DIR, video_name, 'orig_frames')
        os.makedirs(orig_frame_dir, exist_ok=True)
        undist_frame_dir = os.path.join(OUT_DIR, video_name, 'undist_frames')
        os.makedirs(undist_frame_dir, exist_ok=True)

        vid_cap = cv2.VideoCapture(video_path)
        width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vid_cap.get(cv2.CAP_PROP_FPS)
        undistorted_video_path = os.path.join(OUT_DIR, f'{video_name}_undist.mp4')
        vid_writer = cv2.VideoWriter(
            undistorted_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)
        )

        # Generate undistortion maps once
        map1, map2 = image_undistortion.get_undistort_maps((height, width), CAMERA_PARAM)

        frame_id = 0
        while True:
            ret, frame = vid_cap.read()
            if not ret:
                break

            # Perform undistortion on CPU
            undistorted_frame = image_undistortion.undistortion(frame, map1, map2, resize=False)

            # Save frames
            undistorted_frame_path = os.path.join(undist_frame_dir, f'{frame_id}.jpg')
            orig_frame_path = os.path.join(orig_frame_dir, f'{frame_id}.jpg')
            cv2.imwrite(undistorted_frame_path, undistorted_frame)
            cv2.imwrite(orig_frame_path, frame)
            vid_writer.write(undistorted_frame)
            frame_id += 1

        vid_cap.release()
        vid_writer.release()

    print('Video undistorting is done!')

if __name__ == "__main__":
    main()
