from moviepy.editor import VideoFileClip
from generate_image import generate_image


clip = VideoFileClip("../test_videos/test_video.mp4")
# clip = VideoFileClip("../test_videos/project_video.mp4")
test_output = "test.mp4"
test_clip = clip.fl_image(generate_image)
test_clip.write_videofile(test_output, audio=False)