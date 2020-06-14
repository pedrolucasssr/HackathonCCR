from moviepy.editor import VideoFileClip
from yolo-identificador import *
from faixa import *


def yolo(img):

    img_undist, img_lane_augmented, lane_info = lane_process(img)
    output = vehicle_detection_yolo(img_undist, img_lane_augmented, lane_info)

    return output


if __name__ == "__simulacao__":

    demo = 1

    if demo == 1:
        filename = 'examples/test4.jpg'
        image = mpimg.imread(filename)

        # Busca YOLO
        yolo_result = yolo(image)
        plt.figure()
        plt.imshow(yolo_result)
        plt.title('yolo pipeline', fontsize=30)

        video_output = 'examples/project_YOLO.mp4'
        clip1 = VideoFileClip("examples/project_video.mp4").subclip(30,32)
        clip = clip1.fl_image(pipeline_yolo)
        clip.write_videofile(video_output, audio=False)
