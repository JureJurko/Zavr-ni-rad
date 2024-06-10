import cv2
import sys

from ultralytics import YOLO
import matplotlib.pyplot as plt


class Model:
    IMAGE = 'IMAGE'
    VIDEO = 'VIDEO'

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def predict_media(self, media, media_path, output_path):
        if media.upper() == Model.VIDEO:
            self.predict_video(media_path, output_path)
        elif media.upper() == Model.IMAGE:
            self.predict_image(media_path)
        else:
            print("Wrong media argument: <image|video>")
            sys.exit(1)

    def predict_image(self, image_path):
        image = cv2.imread(image_path)

        results = self.model.predict(image, show=False)

        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    def predict_video(self, video_path, output_path):
        video = cv2.VideoCapture(video_path)

        if not video.isOpened():
            print("Error: Could not open video.")
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(video.get(3)), int(video.get(4))))

            while video.isOpened():
                ret, frame = video.read()
                if ret:
                    results = self.model.predict(source=frame, show=False)

                    processed_frame = results[0].plot()
                    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
                    out.write(processed_frame)
                else:
                    break

            video.release()
            out.release()


def parse_args():
    if len(sys.argv) != 5:
        print("Usage: python model.py <model_path> <image|video> <image_path|video_path> <output_path>")
        sys.exit(1)
    else:
        model_path = sys.argv[1]
        media_type = sys.argv[2]
        media_path = sys.argv[3]
        output_path = sys.argv[4]

    return model_path, media_type, media_path, output_path


def main():
    model_path, media_type, media_path, output_path = parse_args()

    model = Model(model_path)

    model.predict_media(media_type, media_path, output_path)

    print("Predicting done!")


if __name__ == '__main__':
    main()
