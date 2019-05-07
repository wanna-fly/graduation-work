"""
video -> frames
"""
import os
import cv2

from opts import parser
args = parser.parse_args()

class video_preprocessor():
    def __init__(self,
                 source_data_path=args.source_path,
                 dataset=args.dataset,
                 out_path=args.out_path,
                 split_file_path=args.split_file_path,
                 split_mode=args.split_mode
    ):
        self.dataset = dataset
        self.out_path = out_path
        self.split_file_path = split_file_path
        self.split_mode = split_mode
        self.source_data_path=source_data_path

        if not os.path.exists(self.out_path):
            os.mkdir(self.out_path)
        if not os.path.exists(os.path.join(self.out_path, self.dataset)):
            os.mkdir(os.path.join(self.out_path, self.dataset))
            for i in range(3):
                os.mkdir(os.path.join(self.out_path, self.dataset, "split_"+str(i+1)))
                os.mkdir(os.path.join(self.out_path, self.dataset, "split_" + str(i+1), "train"))
                os.mkdir(os.path.join(self.out_path, self.dataset, "split_" + str(i+1), "valid"))
                os.mkdir(os.path.join(self.out_path, self.dataset, "split_" + str(i+1), "test"))

        self.root_path = os.path.join(self.out_path, self.dataset, "split_"+str(split_mode))

    def get_split_strategy(self):
        split_files = os.listdir(self.split_file_path)
        # split_1 = [file for file in split_files if file[-5] == '1']
        # split_2 = [file for file in split_files if file[-5] == '2']
        # split_3 = [file for file in split_files if file[-5] == '3']
        splits = [file for file in split_files if file[-5] == str(self.split_mode)]

        train_num=0; test_num=0; valid_num=0;
        for split_file in splits:
            class_name = '_'.join(split_file.split('.')[0].split('_')[:-2])
            contents = [x.strip().split() for x in open(os.path.join(self.split_file_path, split_file)).readlines()]
            train_videos = [line[0] for line in contents if line[1] == '1']
            test_videos = [line[0] for line in contents if line[1] == '2']
            valid_videos = [line[0] for line in contents if line[1] == '0']
            train_num+=len(train_videos)
            test_num+=len(test_videos)
            valid_num+=len(valid_videos)

            source_path = os.path.join(self.source_data_path, class_name)
            for video in train_videos:
                target_path = os.path.join(self.root_path, "train", class_name)
                if not os.path.exists(target_path):
                    os.mkdir(target_path)
                self.extract_frames(video, source_path, target_path)
            for video in test_videos:
                target_path = os.path.join(self.root_path, "test", class_name)
                if not os.path.exists(target_path):
                    os.mkdir(target_path)
                self.extract_frames(video, source_path, target_path)
            for video in valid_videos:
                target_path = os.path.join(self.root_path, "valid", class_name)
                if not os.path.exists(target_path):
                    os.mkdir(target_path)
                self.extract_frames(video, source_path, target_path)

        print("train videos num:", train_num)
        print("test videos num:", test_num)
        print("valid videos num:", valid_num)

    def extract_frames(self, video, source_path, target_path):
        video_filename = video.split('.')[0]
        if not os.path.exists(os.path.join(target_path, video_filename)):
            os.mkdir(os.path.join(target_path, video_filename))

        capture = cv2.VideoCapture(os.path.join(source_path, video))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        i = 0
        success = True
        while success:
            success, frame = capture.read()
            if frame is None:
                continue
            cv2.imwrite(os.path.join(target_path, video_filename, '00{}.jpg'.format(str(i))), frame)
            i += 1
        capture.release()


if __name__ =="__main__":
    vd = video_preprocessor()
    vd.get_split_strategy()