from lavis.datasets.datasets.read_video import read_video

video, info = read_video("/home/tom/orbslam3_docker/Datasets/cameractrl/realestate/c4b2915ba803f481/video.mp4")

print(video.shape)
print(info)